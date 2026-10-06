#!/usr/bin/env python3
"""Run fixed-n fact-overlap dose response at one pre-frozen model layer."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .cache_io import assert_row_alignment, load_complete_cache
from study_truth_transport_v2.src.data import LANGUAGES, group_lookup, pure_group_activations
from study_truth_transport_v2.src.metrics import cosine, score_transport
from study_truth_transport_v2.src.probes import fit_mass_mean


def load_language(prompt_root: Path, cache_root: Path, language: str, block_index: int) -> dict:
    prompt_path = prompt_root / f"{language}.activation.json"
    rows = json.loads(prompt_path.read_text(encoding="utf-8"))
    cache, metadata = load_complete_cache(cache_root / language, prompt_path)
    assert_row_alignment(rows, metadata)
    if block_index >= cache.shape[1]:
        raise ValueError(f"Selected block is missing for {language}")
    layer = cache[:, block_index, :]
    result = {"metadata": metadata, "rows": rows}
    for partition in ("train", "validation", "test"):
        x, y, ids, excluded = pure_group_activations(layer, rows, partition)
        result[partition] = {"x": x, "y": y, "ids": ids,
                             "lookup": group_lookup(x, y, ids), "excluded": excluded}
    return result


def fit_ids(lookup: dict, ids: list[str]):
    try:
        x = np.asarray([lookup[group][0] for group in ids])
        y = np.asarray([lookup[group][1] for group in ids], dtype=int)
    except KeyError as error:
        raise ValueError(f"Allocation contains unavailable group {error.args[0]}") from error
    return fit_mass_mean(x, y)


def evaluate(probe, partition: dict) -> dict:
    scores = probe.score(partition["x"])
    result = score_transport(partition["y"], scores, source_threshold=0.0)
    result["n_groups"] = len(partition["ids"])
    result["n_true"] = int(partition["y"].sum())
    result["n_false"] = int((partition["y"] == 0).sum())
    return result


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--layer-selection", required=True, type=Path)
    parser.add_argument("--allocation-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    selection = json.loads(args.layer_selection.read_text(encoding="utf-8"))
    if selection["model_id"] != args.model_id:
        raise ValueError("Layer-selection model ID mismatch")
    block_index = int(selection["selected_cache_index"])
    language_data = {language: load_language(args.prompt_root, args.cache_root, language, block_index)
                     for language in LANGUAGES}
    rows = []
    reliabilities = {}

    for left_index, source_language in enumerate(LANGUAGES):
        for target_language in LANGUAGES[left_index + 1:]:
            allocation_path = args.allocation_root / f"{source_language}__{target_language}.json"
            if not allocation_path.exists():
                allocation_path = args.allocation_root / f"{target_language}__{source_language}.json"
            allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
            pair = tuple(allocation["language_pair"])
            left, right = pair
            zero_plans = [plan for plan in allocation["plans"] if plan["shared_groups"] == 0]
            for language in pair:
                reliability_values = []
                lookup = language_data[language]["train"]["lookup"]
                for plan in zero_plans:
                    first = fit_ids(lookup, plan["source_group_ids"])
                    second = fit_ids(lookup, plan["target_group_ids"])
                    reliability_values.append(cosine(first.direction, second.direction))
                reliabilities[f"{left}__{right}::{language}"] = {
                    "mean": float(np.mean(reliability_values)),
                    "allocation_p025": percentile(reliability_values, 0.025),
                    "allocation_p975": percentile(reliability_values, 0.975),
                    "values": reliability_values,
                    "interpretation": "Monte Carlo split reliability over pair-matched disjoint allocations; not a population CI"
                }
            r_left = reliabilities[f"{left}__{right}::{left}"]["mean"]
            r_right = reliabilities[f"{left}__{right}::{right}"]["mean"]
            denominator = float(np.sqrt(r_left * r_right)) if r_left > 0 and r_right > 0 else None

            for plan in allocation["plans"]:
                left_probe = fit_ids(language_data[left]["train"]["lookup"], plan["source_group_ids"])
                right_probe = fit_ids(language_data[right]["train"]["lookup"], plan["target_group_ids"])
                raw_cosine = cosine(left_probe.direction, right_probe.direction)
                common = {
                    "model_id": args.model_id,
                    "block_number": block_index + 1,
                    "pair": [left, right],
                    "repetition": int(plan["repetition"]),
                    "overlap_fraction": float(plan["requested_overlap_fraction"]),
                    "shared_groups": int(plan["shared_groups"]),
                    "raw_cosine": raw_cosine,
                    "reliability_left": r_left,
                    "reliability_right": r_right,
                    "reliability_denominator": denominator,
                    "adjusted_cosine_unclipped": raw_cosine / denominator if denominator else None,
                }
                rows.append({**common, "source": left, "target": right,
                             "transfer": evaluate(left_probe, language_data[right]["test"])})
                rows.append({**common, "source": right, "target": left,
                             "transfer": evaluate(right_probe, language_data[left]["test"])})

    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_path = args.output_dir / "allocation_results.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    grouped = defaultdict(list)
    for row in rows:
        key = (row["source"], row["target"], row["overlap_fraction"])
        grouped[key].append(row)
    cells = []
    for (source, target, overlap), values in sorted(grouped.items()):
        metrics = {
            "raw_cosine": [v["raw_cosine"] for v in values],
            "adjusted_cosine_unclipped": [v["adjusted_cosine_unclipped"] for v in values
                                           if v["adjusted_cosine_unclipped"] is not None],
            "auroc": [v["transfer"]["auroc"] for v in values],
            "balanced_accuracy": [v["transfer"]["balanced_accuracy"] for v in values],
            "global_offset": [v["transfer"]["global_offset"] for v in values],
            "standardized_separation": [v["transfer"]["standardized_separation"] for v in values],
        }
        summary_metrics = {}
        for name, observed in metrics.items():
            summary_metrics[name] = ({"mean": float(np.mean(observed)),
                                      "allocation_p025": percentile(observed, 0.025),
                                      "allocation_p975": percentile(observed, 0.975)}
                                     if observed else None)
        cells.append({"source": source, "target": target, "overlap_fraction": overlap,
                      "n_allocations": len(values), "metrics": summary_metrics})

    slopes = []
    for source in LANGUAGES:
        for target in LANGUAGES:
            if source == target:
                continue
            relevant = [cell for cell in cells if cell["source"] == source and cell["target"] == target]
            if not relevant:
                continue
            x = np.asarray([cell["overlap_fraction"] for cell in relevant])
            slopes.append({
                "source": source,
                "target": target,
                "raw_cosine_slope_per_full_overlap": float(np.polyfit(x, [c["metrics"]["raw_cosine"]["mean"] for c in relevant], 1)[0]),
                "auroc_slope_per_full_overlap": float(np.polyfit(x, [c["metrics"]["auroc"]["mean"] for c in relevant], 1)[0]),
            })
    summary = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_id": args.model_id,
        "selected_block_number": block_index + 1,
        "layer_selection": str(args.layer_selection.resolve()),
        "n_directional_rows": len(rows),
        "reliability": reliabilities,
        "cells": cells,
        "descriptive_pair_slopes": slopes,
        "uncertainty_note": "Cell ranges are allocation-sensitivity percentiles. Hierarchical fact-group bootstrap is a separate inferential stage.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_dir), "rows": len(rows),
                      "selected_block_number": block_index + 1}))


if __name__ == "__main__":
    main()
