#!/usr/bin/env python3
"""Cross-fit difficulty removal, truth geometry, and difficulty transfer analysis."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analyze_rq_a import evaluate, fit_ids, load_language
from study_truth_transport_v2.src.data import LANGUAGES
from study_truth_transport_v2.src.difficulty import (
    difficulty_probe, extreme_indices, pooled_difficulty_direction, stable_fold)
from study_truth_transport_v2.src.metrics import auroc, cosine
from study_truth_transport_v2.src.probes import fit_mass_mean
from study_truth_transport_v2.src.residualization import project_out


def behavior_maps(path: Path) -> dict[str, dict[str, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {language: {row["group_id"]: row for row in payload["languages"][language]["groups"]
                       if not row["mixed_label"]} for language in LANGUAGES}


def residual_lookup(table: dict, directions_by_fold: dict[int, np.ndarray]) -> dict:
    result = {}
    for index, gid in enumerate(table["ids"]):
        vector = project_out(table["x"][index:index + 1], directions_by_fold[stable_fold(gid)])[0]
        result[gid] = (vector, int(table["y"][index]))
    return result


def test_extremes(table: dict, behavior: dict) -> tuple[np.ndarray, np.ndarray]:
    ease = np.asarray([behavior[gid]["ease_composite"] for gid in table["ids"]])
    easy, hard = extreme_indices(table["y"], ease)
    indices = np.concatenate([hard, easy])
    return table["x"][indices], np.concatenate([np.zeros(len(hard), dtype=int), np.ones(len(easy), dtype=int)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--layer-selection", required=True, type=Path)
    parser.add_argument("--behavior", required=True, type=Path)
    parser.add_argument("--allocation-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    selection = json.loads(args.layer_selection.read_text(encoding="utf-8"))
    block_index = int(selection["selected_cache_index"])
    data = {language: load_language(args.prompt_root, args.cache_root, language, block_index)
            for language in LANGUAGES}
    behavior = behavior_maps(args.behavior)
    train_tables = {}
    for language in LANGUAGES:
        table = data[language]["train"]
        missing = set(table["ids"]) - set(behavior[language])
        if missing:
            raise ValueError(f"Missing behavioral groups for {language}: {len(missing)}")
        train_tables[language] = {**table, "behavior": behavior[language]}

    all_groups = set().union(*(set(table["ids"]) for table in train_tables.values()))
    fold_directions = {}
    for fold in range(5):
        allowed = {gid for gid in all_groups if stable_fold(gid) != fold}
        fold_directions[fold], _ = pooled_difficulty_direction(train_tables, allowed)
    final_direction, language_difficulty_directions = pooled_difficulty_direction(train_tables)

    original_probes, residual_probes = {}, {}
    residual_train_lookups, residual_tests = {}, {}
    geometry = []
    for language in LANGUAGES:
        original_probes[language] = fit_mass_mean(data[language]["train"]["x"], data[language]["train"]["y"])
        residual_train_lookups[language] = residual_lookup(data[language]["train"], fold_directions)
        rx = np.asarray([residual_train_lookups[language][gid][0] for gid in data[language]["train"]["ids"]])
        residual_probes[language] = fit_mass_mean(rx, data[language]["train"]["y"])
        residual_tests[language] = {**data[language]["test"],
                                    "x": project_out(data[language]["test"]["x"], final_direction)}
        geometry.append({
            "language": language,
            "truth_difficulty_cosine": cosine(original_probes[language].direction,
                                                language_difficulty_directions[language]),
        })

    pair_results = []
    difficulty_transfer = []
    for source in LANGUAGES:
        source_behavior = behavior[source]
        train = data[source]["train"]
        source_ease = np.asarray([source_behavior[gid]["ease_composite"] for gid in train["ids"]])
        difficulty_source_probe = difficulty_probe(train["x"], train["y"], source_ease)
        for target in LANGUAGES:
            original_eval = evaluate(original_probes[source], data[target]["test"])
            residual_eval = evaluate(residual_probes[source], residual_tests[target])
            pair_results.append({
                "source": source, "target": target,
                "truth_cosine_original": cosine(original_probes[source].direction, original_probes[target].direction),
                "truth_cosine_residual": cosine(residual_probes[source].direction, residual_probes[target].direction),
                "original_transfer": original_eval,
                "residual_transfer": residual_eval,
            })
            target_x, target_easy = test_extremes(data[target]["test"], behavior[target])
            difficulty_transfer.append({
                "source": source, "target": target,
                "auroc": auroc(target_easy, difficulty_source_probe.score(target_x)),
                "n_groups": len(target_easy),
            })

    difficulty_cosines = []
    for i, left in enumerate(LANGUAGES):
        for right in LANGUAGES[i + 1:]:
            difficulty_cosines.append({"left": left, "right": right,
                                       "cosine": cosine(language_difficulty_directions[left],
                                                        language_difficulty_directions[right])})

    overlap_rows = []
    for i, left in enumerate(LANGUAGES):
        for right in LANGUAGES[i + 1:]:
            allocation_path = args.allocation_root / f"{left}__{right}.json"
            allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
            for plan in allocation["plans"]:
                left_probe = fit_ids(residual_train_lookups[left], plan["source_group_ids"])
                right_probe = fit_ids(residual_train_lookups[right], plan["target_group_ids"])
                overlap_rows.append({
                    "left": left, "right": right, "repetition": plan["repetition"],
                    "overlap_fraction": plan["requested_overlap_fraction"],
                    "residual_cosine": cosine(left_probe.direction, right_probe.direction),
                    "left_to_right_auroc": evaluate(left_probe, residual_tests[right])["auroc"],
                    "right_to_left_auroc": evaluate(right_probe, residual_tests[left])["auroc"],
                })
    grouped = defaultdict(list)
    for row in overlap_rows:
        grouped[row["overlap_fraction"]].append(row)
    overlap_summary = [{
        "overlap_fraction": overlap,
        "mean_residual_cosine": float(np.mean([row["residual_cosine"] for row in rows])),
        "mean_residual_transfer_auroc": float(np.mean(
            [value for row in rows for value in (row["left_to_right_auroc"], row["right_to_left_auroc"])])),
        "n_unordered_pair_allocations": len(rows),
    } for overlap, rows in sorted(grouped.items())]

    result = {
        "schema_version": 1, "model_id": args.model_id,
        "selected_block_number": block_index + 1,
        "fold_rule": "sha256(group_id) modulo 5; shared across languages",
        "difficulty_direction": "label-balanced easy-minus-hard tercile contrast, then equal mean across six languages",
        "training_residualization": "out-of-fold direction for every training group",
        "validation_test_residualization": "one final direction estimated from all training groups",
        "geometry_by_language": geometry,
        "difficulty_direction_cosines": difficulty_cosines,
        "difficulty_transfer": difficulty_transfer,
        "truth_pair_results": pair_results,
        "overlap_summary": overlap_summary,
        "overlap_allocation_rows": overlap_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pair_results": len(pair_results),
                      "overlap_rows": len(overlap_rows)}))


if __name__ == "__main__":
    main()
