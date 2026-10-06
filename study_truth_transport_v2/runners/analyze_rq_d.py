#!/usr/bin/env python3
"""Knowledge-quadrant transfer using allocation-averaged zero-overlap directions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze_rq_a import fit_ids, load_language
from .analyze_rq_b import behavior_maps, residual_lookup
from study_truth_transport_v2.src.data import LANGUAGES
from study_truth_transport_v2.src.difficulty import pooled_difficulty_direction, stable_fold
from study_truth_transport_v2.src.metrics import auroc
from study_truth_transport_v2.src.probes import fit_mass_mean
from study_truth_transport_v2.src.residualization import project_out


def stratified_auc_interval(y: np.ndarray, score: np.ndarray, draws: int, rng) -> dict:
    if len(y) < 2 or len(np.unique(y)) < 2:
        return {"auroc": None, "p025": None, "p975": None}
    observed = auroc(y, score)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    boot = []
    for _ in range(draws):
        ix = np.concatenate([rng.choice(positive, len(positive), replace=True),
                             rng.choice(negative, len(negative), replace=True)])
        boot.append(auroc(y[ix], score[ix]))
    return {"auroc": observed, "p025": float(np.quantile(boot, 0.025)),
            "p975": float(np.quantile(boot, 0.975))}


def permutation_test(y: np.ndarray, score: np.ndarray, draws: int, rng) -> dict:
    if len(np.unique(y)) < 2:
        return {"null_mean": None, "p_greater": None}
    observed = auroc(y, score)
    null = [auroc(rng.permutation(y), score) for _ in range(draws)]
    return {"null_mean": float(np.mean(null)),
            "p_greater": float((1 + sum(value >= observed for value in null)) / (draws + 1))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--layer-selection", required=True, type=Path)
    parser.add_argument("--behavior", required=True, type=Path)
    parser.add_argument("--allocation-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--permutation-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    block_index = int(json.loads(args.layer_selection.read_text())["selected_cache_index"])
    data = {language: load_language(args.prompt_root, args.cache_root, language, block_index)
            for language in LANGUAGES}
    behavior = behavior_maps(args.behavior)
    train_tables = {language: {**data[language]["train"], "behavior": behavior[language]}
                    for language in LANGUAGES}
    all_groups = set().union(*(set(table["ids"]) for table in train_tables.values()))
    fold_directions = {}
    for fold in range(5):
        fold_directions[fold], _ = pooled_difficulty_direction(
            train_tables, {gid for gid in all_groups if stable_fold(gid) != fold})
    final_direction, _ = pooled_difficulty_direction(train_tables)
    residual_train = {language: residual_lookup(data[language]["train"], fold_directions)
                      for language in LANGUAGES}
    residual_test = {language: project_out(data[language]["test"]["x"], final_direction)
                     for language in LANGUAGES}
    native_original = {language: fit_mass_mean(data[language]["train"]["x"], data[language]["train"]["y"])
                       for language in LANGUAGES}
    native_residual = {}
    for language in LANGUAGES:
        x = np.asarray([residual_train[language][gid][0] for gid in data[language]["train"]["ids"]])
        native_residual[language] = fit_mass_mean(x, data[language]["train"]["y"])

    rng = np.random.default_rng(args.seed)
    pairs = []
    for source in LANGUAGES:
        for target in LANGUAGES:
            if source == target:
                continue
            left, right = ((source, target) if (args.allocation_root / f"{source}__{target}.json").exists()
                           else (target, source))
            allocation = json.loads((args.allocation_root / f"{left}__{right}.json").read_text())
            zero = [plan for plan in allocation["plans"] if plan["shared_groups"] == 0]
            original_scores, residual_scores = [], []
            for plan in zero:
                ids = plan["source_group_ids"] if source == left else plan["target_group_ids"]
                original_probe = fit_ids(data[source]["train"]["lookup"], ids)
                residual_probe = fit_ids(residual_train[source], ids)
                original_scores.append(original_probe.score(data[target]["test"]["x"]))
                residual_scores.append(residual_probe.score(residual_test[target]))
            mean_original = np.mean(np.asarray(original_scores), axis=0)
            mean_residual = np.mean(np.asarray(residual_scores), axis=0)
            target_native_score = native_original[target].score(data[target]["test"]["x"])
            target_native_residual_score = native_residual[target].score(residual_test[target])
            source_known = np.asarray([behavior[source][gid]["known"] for gid in data[target]["test"]["ids"]])
            target_known = np.asarray([behavior[target][gid]["known"] for gid in data[target]["test"]["ids"]])
            labels = data[target]["test"]["y"]
            quadrants = []
            for name, mask in {
                "K/K": source_known & target_known,
                "K/U": source_known & ~target_known,
                "U/K": ~source_known & target_known,
                "U/U": ~source_known & ~target_known,
            }.items():
                y = labels[mask]
                entry = {"quadrant": name, "n_groups": int(mask.sum()),
                         "n_true": int(y.sum()), "n_false": int((y == 0).sum()),
                         "small_cell": bool(mask.sum() < 30)}
                for metric_name, score in {
                    "zero_overlap_transfer": mean_original,
                    "zero_overlap_transfer_residual": mean_residual,
                    "target_native": target_native_score,
                    "target_native_residual": target_native_residual_score,
                }.items():
                    entry[metric_name] = stratified_auc_interval(y, score[mask], args.bootstrap_draws, rng)
                entry["zero_overlap_permutation"] = permutation_test(
                    y, mean_original[mask], args.permutation_draws, rng)
                quadrants.append(entry)
            pairs.append({"source": source, "target": target, "zero_overlap_allocations": len(zero),
                          "quadrants": quadrants})
    result = {
        "schema_version": 1, "model_id": args.model_id,
        "selected_block_number": block_index + 1,
        "known_definition": ">=2/3 correct and prediction sign identical across all three frozen templates; dependency group requires all member claims known",
        "zero_overlap_score": "mean target score across 50 independently fitted zero-overlap source directions",
        "uncertainty": "stratified dependency-group bootstrap conditional on the allocation-averaged direction",
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "ordered_pairs": len(pairs)}))


if __name__ == "__main__":
    main()
