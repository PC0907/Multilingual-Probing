#!/usr/bin/env python3
"""Per-language and multidimensional difficulty-erasure sensitivity analyses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .analyze_rq_a import evaluate, load_language
from .analyze_rq_b import behavior_maps
from study_truth_transport_v2.src.data import LANGUAGES
from study_truth_transport_v2.src.difficulty import difficulty_probe, stable_fold
from study_truth_transport_v2.src.metrics import cosine
from study_truth_transport_v2.src.probes import fit_mass_mean


COMPONENTS = ("correct_fraction", "aligned_margin", "consistent_prediction",
              "negative_entropy", "negative_surprisal")


def component_direction(table: dict, behavior: dict, component: str,
                        allowed: set[str] | None = None) -> np.ndarray:
    indices = np.asarray([i for i, gid in enumerate(table["ids"])
                          if allowed is None or gid in allowed], dtype=int)
    values = np.asarray([behavior[table["ids"][i]]["components"][component] for i in indices])
    return difficulty_probe(table["x"][indices], table["y"][indices], values).direction


def orthonormal_basis(directions: list[np.ndarray]) -> np.ndarray:
    matrix = np.column_stack(directions).astype(np.float64)
    q, r = np.linalg.qr(matrix, mode="reduced")
    keep = np.abs(np.diag(r)) > 1e-10
    if not keep.any():
        raise ValueError("Difficulty subspace has zero rank")
    return q[:, keep]


def project_out_basis(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    array = np.asarray(x, dtype=np.float64)
    return array - (array @ basis) @ basis.T


def bases(data: dict, behavior: dict, mode: str, allowed: set[str] | None = None) -> dict[str, np.ndarray]:
    if mode == "pooled_components":
        directions = []
        for component in COMPONENTS:
            per_language = [component_direction(data[l]["train"], behavior[l], component, allowed)
                            for l in LANGUAGES]
            directions.append(np.mean(np.asarray(per_language), axis=0))
        basis = orthonormal_basis(directions)
        return {language: basis for language in LANGUAGES}
    if mode == "per_language_components":
        return {language: orthonormal_basis([
            component_direction(data[language]["train"], behavior[language], component, allowed)
            for component in COMPONENTS]) for language in LANGUAGES}
    if mode == "per_language_single":
        return {language: orthonormal_basis([
            component_direction(data[language]["train"], behavior[language], "ease_composite", allowed)
        ]) for language in LANGUAGES}
    raise ValueError(mode)


def single_direction(table: dict, behavior: dict, allowed: set[str] | None = None) -> np.ndarray:
    indices = np.asarray([i for i, gid in enumerate(table["ids"])
                          if allowed is None or gid in allowed], dtype=int)
    values = np.asarray([behavior[table["ids"][i]]["ease_composite"] for i in indices])
    return difficulty_probe(table["x"][indices], table["y"][indices], values).direction


def variant_bases(data: dict, behavior: dict, mode: str, allowed: set[str] | None = None):
    if mode == "per_language_single":
        return {language: orthonormal_basis([single_direction(
            data[language]["train"], behavior[language], allowed)]) for language in LANGUAGES}
    return bases(data, behavior, mode, allowed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--layer-selection", required=True, type=Path)
    parser.add_argument("--behavior", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    block_index = int(json.loads(args.layer_selection.read_text())["selected_cache_index"])
    data = {language: load_language(args.prompt_root, args.cache_root, language, block_index)
            for language in LANGUAGES}
    behavior = behavior_maps(args.behavior)
    all_groups = set().union(*(set(data[l]["train"]["ids"]) for l in LANGUAGES))
    output = []
    for mode in ("per_language_single", "pooled_components", "per_language_components"):
        fold_bases = {fold: variant_bases(
            data, behavior, mode, {gid for gid in all_groups if stable_fold(gid) != fold})
                      for fold in range(5)}
        final_bases = variant_bases(data, behavior, mode)
        probes, tests = {}, {}
        ranks = {}
        for language in LANGUAGES:
            train = data[language]["train"]
            residual_rows = [project_out_basis(train["x"][i:i + 1],
                                                fold_bases[stable_fold(gid)][language])[0]
                             for i, gid in enumerate(train["ids"])]
            probes[language] = fit_mass_mean(np.asarray(residual_rows), train["y"])
            tests[language] = {**data[language]["test"],
                               "x": project_out_basis(data[language]["test"]["x"],
                                                      final_bases[language])}
            ranks[language] = int(final_bases[language].shape[1])
        cells = []
        for source in LANGUAGES:
            for target in LANGUAGES:
                cells.append({"source": source, "target": target,
                              "direction_cosine": cosine(probes[source].direction,
                                                         probes[target].direction),
                              **evaluate(probes[source], tests[target])})
        cross = [row for row in cells if row["source"] != row["target"]]
        output.append({"mode": mode, "subspace_rank_by_language": ranks,
                       "mean_cross_language_cosine": float(np.mean(
                           [row["direction_cosine"] for row in cross])),
                       "mean_cross_language_auroc": float(np.mean([row["auroc"] for row in cross])),
                       "mean_cross_language_balanced_accuracy": float(np.mean(
                           [row["balanced_accuracy"] for row in cross])),
                       "cells": cells})
    result = {
        "schema_version": 1, "model_id": args.model_id,
        "selected_block_number": block_index + 1,
        "fold_rule": "sha256(group_id) modulo 5; train-only cross-fitting",
        "components": list(COMPONENTS),
        "method": "orthogonal projection of cross-fitted easy-minus-hard component subspaces",
        "leace_note": "This is a multidimensional orthogonal erasure sensitivity, not exact LEACE whitening.",
        "variants": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "variants": len(output)}))


if __name__ == "__main__":
    main()
