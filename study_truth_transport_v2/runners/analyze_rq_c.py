#!/usr/bin/env python3
"""Compute full score-transport matrices and unlabeled moment adaptation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze_rq_a import load_language
from study_truth_transport_v2.src.data import LANGUAGES
from study_truth_transport_v2.src.metrics import (
    auroc, balanced_accuracy, calibrated_probabilities, calibration_metrics, cosine,
    fit_logistic_calibrator, optimal_balanced_accuracy_threshold, score_transport)
from study_truth_transport_v2.src.probes import fit_mass_mean


def recenter_rescale(target_score: np.ndarray, target_calibration: np.ndarray,
                     source_calibration: np.ndarray) -> np.ndarray:
    target_sd = float(np.std(target_calibration, ddof=1))
    source_sd = float(np.std(source_calibration, ddof=1))
    if target_sd <= 0 or source_sd <= 0:
        raise ValueError("Moment adaptation requires positive standard deviations")
    return ((target_score - float(np.mean(target_calibration))) / target_sd * source_sd +
            float(np.mean(source_calibration)))


def prior_indices(y: np.ndarray, positive_fraction: float, n: int, rng) -> np.ndarray:
    positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    n_positive = int(round(n * positive_fraction))
    n_negative = n - n_positive
    if n_positive > len(positives) or n_negative > len(negatives):
        raise ValueError("Prior subsample exceeds available class")
    return np.concatenate([rng.choice(positives, n_positive, replace=False),
                           rng.choice(negatives, n_negative, replace=False)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--layer-selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prior-repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    selection = json.loads(args.layer_selection.read_text(encoding="utf-8"))
    block_index = int(selection["selected_cache_index"])
    data = {language: load_language(args.prompt_root, args.cache_root, language, block_index)
            for language in LANGUAGES}
    probes = {language: fit_mass_mean(data[language]["train"]["x"], data[language]["train"]["y"])
              for language in LANGUAGES}
    cells = []
    rng = np.random.default_rng(args.seed)
    for source in LANGUAGES:
        probe = probes[source]
        source_val_score = probe.score(data[source]["validation"]["x"])
        source_test_score = probe.score(data[source]["test"]["x"])
        slope, intercept = fit_logistic_calibrator(source_val_score, data[source]["validation"]["y"])
        source_transport = score_transport(data[source]["test"]["y"], source_test_score)
        source_spread = float(source_transport["pooled_sd"])
        for target in LANGUAGES:
            target_val_score = probe.score(data[target]["validation"]["x"])
            target_test_score = probe.score(data[target]["test"]["x"])
            y_val = data[target]["validation"]["y"]
            y_test = data[target]["test"]["y"]
            transport = score_transport(y_test, target_test_score)
            threshold, validation_ba = optimal_balanced_accuracy_threshold(y_val, target_val_score)
            probability = calibrated_probabilities(target_test_score, slope, intercept)
            adapted = recenter_rescale(target_test_score, target_val_score, source_val_score)
            prior_results = {}
            prior_n = min(280, len(y_val), len(y_test))
            for fraction in (0.3, 0.7):
                draws = []
                for _ in range(args.prior_repetitions):
                    calibration_ix = prior_indices(y_val, fraction, prior_n, rng)
                    evaluation_ix = prior_indices(y_test, fraction, prior_n, rng)
                    adapted_draw = recenter_rescale(target_test_score[evaluation_ix],
                                                    target_val_score[calibration_ix], source_val_score)
                    draws.append({
                        "zero_shot_balanced_accuracy": balanced_accuracy(y_test[evaluation_ix], target_test_score[evaluation_ix]),
                        "adapted_balanced_accuracy": balanced_accuracy(y_test[evaluation_ix], adapted_draw),
                        "zero_shot_auroc": auroc(y_test[evaluation_ix], target_test_score[evaluation_ix]),
                        "adapted_auroc": auroc(y_test[evaluation_ix], adapted_draw),
                    })
                prior_results[str(fraction)] = {
                    key: {"mean": float(np.mean([draw[key] for draw in draws])),
                          "p025": float(np.quantile([draw[key] for draw in draws], 0.025)),
                          "p975": float(np.quantile([draw[key] for draw in draws], 0.975))}
                    for key in draws[0]
                }
            cells.append({
                "source": source, "target": target,
                "direction_cosine": cosine(probe.direction, probes[target].direction),
                **transport,
                "scale_ratio": float(transport["pooled_sd"] / source_spread),
                "target_validation_optimal_threshold": threshold,
                "threshold_displacement": threshold,
                "target_validation_balanced_accuracy_at_optimal_threshold": validation_ba,
                "calibration": {"source_validation_slope": slope, "source_validation_intercept": intercept,
                                **calibration_metrics(y_test, probability)},
                "unlabeled_adaptation": {
                    "calibration_partition": "target validation inputs with labels hidden from transformation",
                    "balanced_accuracy": balanced_accuracy(y_test, adapted),
                    "auroc": auroc(y_test, adapted),
                    "balanced_accuracy_change": balanced_accuracy(y_test, adapted) - transport["balanced_accuracy"],
                    "auroc_change": auroc(y_test, adapted) - transport["auroc"],
                },
                "prior_shift": prior_results,
            })
    result = {"schema_version": 1, "model_id": args.model_id,
              "selected_block_number": block_index + 1, "languages": list(LANGUAGES),
              "cells": cells,
              "notes": [
                  "Target-optimal thresholds use validation labels for diagnosis only.",
                  "Unlabeled adaptation estimates target moments on validation inputs and evaluates on disjoint test groups.",
                  "Prior-shift calibration and evaluation samples are drawn from disjoint validation and test partitions.",
                  "Any AUROC change after affine adaptation should be numerical zero up to ties."
              ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cells": len(cells),
                      "selected_block_number": block_index + 1}))


if __name__ == "__main__":
    main()
