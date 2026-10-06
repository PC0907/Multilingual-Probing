#!/usr/bin/env python3
"""All-layer sensitivity for native and cross-language mass-mean probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .cache_io import assert_row_alignment, load_complete_cache
from study_truth_transport_v2.src.data import LANGUAGES, pure_group_activations
from study_truth_transport_v2.src.metrics import auroc, balanced_accuracy, cosine
from study_truth_transport_v2.src.probes import fit_mass_mean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--layer-selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    selection = json.loads(args.layer_selection.read_text(encoding="utf-8"))
    caches, rows_by_language = {}, {}
    for language in LANGUAGES:
        prompt_path = args.prompt_root / f"{language}.activation.json"
        rows = json.loads(prompt_path.read_text(encoding="utf-8"))
        cache, metadata = load_complete_cache(args.cache_root / language, prompt_path)
        assert_row_alignment(rows, metadata)
        caches[language], rows_by_language[language] = cache, rows
    layer_counts = {cache.shape[1] for cache in caches.values()}
    if len(layer_counts) != 1:
        raise ValueError("Languages have different decoder-block counts")

    layer_results = []
    for block_index in range(next(iter(layer_counts))):
        probes, tests = {}, {}
        for language in LANGUAGES:
            train_x, train_y, _, _ = pure_group_activations(
                caches[language][:, block_index, :], rows_by_language[language], "train")
            test_x, test_y, _, _ = pure_group_activations(
                caches[language][:, block_index, :], rows_by_language[language], "test")
            probes[language] = fit_mass_mean(train_x, train_y)
            tests[language] = (test_x, test_y)

        pair_cells = []
        for source in LANGUAGES:
            for target in LANGUAGES:
                target_x, target_y = tests[target]
                score = probes[source].score(target_x)
                pair_cells.append({
                    "source": source,
                    "target": target,
                    "direction_cosine": cosine(probes[source].direction, probes[target].direction),
                    "auroc": auroc(target_y, score),
                    "balanced_accuracy": balanced_accuracy(target_y, score),
                })
        cross = [cell for cell in pair_cells if cell["source"] != cell["target"]]
        native = [cell for cell in pair_cells if cell["source"] == cell["target"]]
        layer_results.append({
            "block_number": block_index + 1,
            "selected_primary": block_index == int(selection["selected_cache_index"]),
            "mean_native_test_auroc": float(np.mean([cell["auroc"] for cell in native])),
            "mean_cross_language_cosine": float(np.mean([cell["direction_cosine"] for cell in cross])),
            "mean_cross_language_auroc": float(np.mean([cell["auroc"] for cell in cross])),
            "mean_cross_language_balanced_accuracy": float(
                np.mean([cell["balanced_accuracy"] for cell in cross])),
            "pair_cells": pair_cells,
        })

    result = {
        "schema_version": 1,
        "model_id": args.model_id,
        "selected_block_number": int(selection["selected_block_number"]),
        "fit": "full training partition, mass-mean, per language and block",
        "evaluation": "held-out test dependency groups",
        "layers": layer_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "layers": len(layer_results)}))


if __name__ == "__main__":
    main()
