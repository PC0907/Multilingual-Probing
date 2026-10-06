#!/usr/bin/env python3
"""Inferential layer for RQ-A with allocation and dependency-group uncertainty."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analyze_rq_a import fit_ids, load_language
from study_truth_transport_v2.src.data import LANGUAGES
from study_truth_transport_v2.src.metrics import auroc, balanced_accuracy


def stratified_indices(y: np.ndarray, draws: int, rng) -> np.ndarray:
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    pos = rng.choice(positive, size=(draws, len(positive)), replace=True)
    neg = rng.choice(negative, size=(draws, len(negative)), replace=True)
    return np.concatenate([pos, neg], axis=1)


def bootstrap_metrics(y: np.ndarray, score: np.ndarray, draws: int, rng) -> dict:
    indices = stratified_indices(y, draws, rng)
    auc, ba = [], []
    for ix in indices:
        auc.append(auroc(y[ix], score[ix]))
        ba.append(balanced_accuracy(y[ix], score[ix]))
    return {"auroc": np.asarray(auc), "balanced_accuracy": np.asarray(ba)}


def sign_flip_p(difference: np.ndarray, draws: int, rng) -> float:
    difference = np.asarray(difference, dtype=float)
    observed = abs(float(difference.mean()))
    signs = rng.choice((-1.0, 1.0), size=(draws, len(difference)))
    null = np.abs((signs * difference).mean(axis=1))
    return float((1 + np.sum(null >= observed)) / (draws + 1))


def holm(rows: list[dict], p_key: str = "p_two_sided") -> None:
    order = sorted(range(len(rows)), key=lambda i: rows[i][p_key])
    running = 0.0
    m = len(rows)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (m - rank) * rows[index][p_key])
        running = max(running, adjusted)
        rows[index]["p_holm"] = running


def interval(values: np.ndarray) -> dict:
    return {"mean": float(np.mean(values)), "p025": float(np.quantile(values, 0.025)),
            "p975": float(np.quantile(values, 0.975)), "draws": int(len(values))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--layer-selection", required=True, type=Path)
    parser.add_argument("--allocation-root", required=True, type=Path)
    parser.add_argument("--allocation-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--permutation-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()

    block_index = int(json.loads(args.layer_selection.read_text())["selected_cache_index"])
    data = {language: load_language(args.prompt_root, args.cache_root, language, block_index)
            for language in LANGUAGES}
    allocation_rows = [json.loads(line) for line in args.allocation_results.read_text().splitlines()
                       if line.strip()]
    observed = defaultdict(list)
    for row in allocation_rows:
        observed[(row["source"], row["target"], float(row["overlap_fraction"]))].append(row)
    rng = np.random.default_rng(args.seed)
    cells = []
    bootstrap_by_cell = {}
    for source in LANGUAGES:
        for target in LANGUAGES:
            if source == target:
                continue
            allocation_path = (args.allocation_root / f"{source}__{target}.json")
            source_side = "source_group_ids"
            if not allocation_path.exists():
                allocation_path = args.allocation_root / f"{target}__{source}.json"
                source_side = "target_group_ids"
            plan = json.loads(allocation_path.read_text())
            for overlap in (0.0, 0.25, 0.5, 0.75, 1.0):
                matching = [row for row in plan["plans"]
                            if float(row["requested_overlap_fraction"]) == overlap]
                score_rows = []
                for row in matching:
                    probe = fit_ids(data[source]["train"]["lookup"], row[source_side])
                    score_rows.append(probe.score(data[target]["test"]["x"]))
                ensemble_score = np.mean(np.asarray(score_rows), axis=0)
                boot = bootstrap_metrics(data[target]["test"]["y"], ensemble_score,
                                         args.bootstrap_draws, rng)
                bootstrap_by_cell[(source, target, overlap)] = boot
                values = observed[(source, target, overlap)]
                cosine_values = np.asarray([row["raw_cosine"] for row in values])
                allocation_mean_draw = cosine_values[
                    rng.integers(0, len(cosine_values), size=(args.bootstrap_draws, len(cosine_values)))
                ].mean(axis=1)
                cells.append({
                    "source": source, "target": target, "overlap_fraction": overlap,
                    "raw_cosine_allocation_mean": float(cosine_values.mean()),
                    "raw_cosine_allocation_range": {
                        "p025": float(np.quantile(cosine_values, 0.025)),
                        "p975": float(np.quantile(cosine_values, 0.975))},
                    "raw_cosine_monte_carlo_mean_interval": interval(allocation_mean_draw),
                    "ensemble_score_group_bootstrap_auroc": interval(boot["auroc"]),
                    "ensemble_score_group_bootstrap_balanced_accuracy": interval(
                        boot["balanced_accuracy"]),
                    "n_allocations": len(matching),
                    "n_test_groups": len(ensemble_score),
                })

    pooled_draws = []
    pair_keys = [(left, right) for i, left in enumerate(LANGUAGES)
                 for right in LANGUAGES[i + 1:]]
    for draw in range(args.bootstrap_draws):
        sampled_pairs = [pair_keys[i] for i in rng.integers(0, len(pair_keys), size=len(pair_keys))]
        curve = []
        for overlap in (0.0, 0.25, 0.5, 0.75, 1.0):
            cosine_values, auc_values, ba_values = [], [], []
            for left, right in sampled_pairs:
                for source, target in ((left, right), (right, left)):
                    rows = observed[(source, target, overlap)]
                    cosine_values.append(rows[int(rng.integers(0, len(rows)))]["raw_cosine"])
                    boot = bootstrap_by_cell[(source, target, overlap)]
                    auc_values.append(boot["auroc"][draw])
                    ba_values.append(boot["balanced_accuracy"][draw])
            curve.append((overlap, np.mean(cosine_values), np.mean(auc_values), np.mean(ba_values)))
        x = np.asarray([row[0] for row in curve])
        pooled_draws.append({
            "cosine_slope": float(np.polyfit(x, [row[1] for row in curve], 1)[0]),
            "auroc_slope": float(np.polyfit(x, [row[2] for row in curve], 1)[0]),
            "balanced_accuracy_slope": float(np.polyfit(x, [row[3] for row in curve], 1)[0]),
            "zero_cosine": float(curve[0][1]), "full_cosine": float(curve[-1][1]),
            "zero_auroc": float(curve[0][2]), "full_auroc": float(curve[-1][2]),
        })

    cosine_tests, transfer_tests = [], []
    for left, right in pair_keys:
        zero = np.asarray([row["raw_cosine"] for row in observed[(left, right, 0.0)]])
        full = np.asarray([row["raw_cosine"] for row in observed[(left, right, 1.0)]])
        cosine_tests.append({"left": left, "right": right, "mean_full_minus_zero": float((full-zero).mean()),
                             "p_two_sided": sign_flip_p(full-zero, args.permutation_draws, rng)})
        for source, target in ((left, right), (right, left)):
            zero_auc = np.asarray([row["transfer"]["auroc"] for row in observed[(source, target, 0.0)]])
            full_auc = np.asarray([row["transfer"]["auroc"] for row in observed[(source, target, 1.0)]])
            transfer_tests.append({"source": source, "target": target,
                                   "mean_full_minus_zero": float((full_auc-zero_auc).mean()),
                                   "p_two_sided": sign_flip_p(full_auc-zero_auc,
                                                               args.permutation_draws, rng)})
    holm(cosine_tests)
    holm(transfer_tests)
    pooled = {key: interval(np.asarray([row[key] for row in pooled_draws]))
              for key in pooled_draws[0]}
    result = {
        "schema_version": 1, "model_id": args.model_id,
        "selected_block_number": block_index + 1,
        "cells": cells,
        "pooled_hierarchical_bootstrap": pooled,
        "cosine_full_vs_zero_tests_holm_15": cosine_tests,
        "transfer_full_vs_zero_tests_holm_30": transfer_tests,
        "uncertainty_notes": [
            "Cosine allocation ranges describe the frozen repeated allocations; Monte Carlo mean intervals resample allocations.",
            "Predictive intervals bootstrap dependency groups on the allocation-averaged score ensemble.",
            "The pooled bootstrap resamples language pairs, allocations, and target dependency groups.",
            "Pair endpoint p-values use paired sign-flip tests over 50 allocations and Holm adjustment."
        ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cells": len(cells),
                      "bootstrap_draws": args.bootstrap_draws}))


if __name__ == "__main__":
    main()
