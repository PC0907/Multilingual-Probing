#!/usr/bin/env python3
"""RQ-A isotropic, covariance-matched, label, and fact-identity controls."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .analyze_rq_a import fit_ids, load_language
from study_truth_transport_v2.src.data import LANGUAGES
from study_truth_transport_v2.src.metrics import auroc, cosine


def summarize(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "p025": float(np.quantile(values, 0.025)),
            "p50": float(np.quantile(values, 0.5)), "p975": float(np.quantile(values, 0.975)),
            "n": len(values)}


def random_label_directions(x: np.ndarray, y: np.ndarray, draws: int, rng) -> np.ndarray:
    n_positive = int(y.sum())
    weights = np.empty((draws, len(y)), dtype=np.float32)
    for draw in range(draws):
        positive = rng.choice(len(y), n_positive, replace=False)
        row = np.full(len(y), -1 / (len(y) - n_positive), dtype=np.float32)
        row[positive] = 1 / n_positive
        weights[draw] = row
    return weights @ np.asarray(x, dtype=np.float32)


def shuffled_stratified_labels(ids: list[str], labels: np.ndarray, annotations: dict, rng) -> np.ndarray:
    result = labels.copy()
    buckets = defaultdict(list)
    for index, gid in enumerate(ids):
        buckets[annotations[gid]["stratum"]].append(index)
    for indices in buckets.values():
        result[indices] = rng.permutation(result[indices])
    return result


def fit_with_labels(lookup: dict, ids: list[str], labels: np.ndarray) -> np.ndarray:
    x = np.asarray([lookup[gid][0] for gid in ids])
    return x[labels == 1].mean(axis=0) - x[labels == 0].mean(axis=0)


def identity_mapping(pool_ids: list[str], labels: dict[str, int], annotations: dict, rng) -> tuple[dict, int]:
    buckets = defaultdict(list)
    for gid in pool_ids:
        buckets[(labels[gid], annotations[gid]["stratum"])].append(gid)
    mapping, fixed = {}, 0
    for members in buckets.values():
        ordered = list(members)
        if len(ordered) == 1:
            mapping[ordered[0]] = ordered[0]
            fixed += 1
            continue
        shuffled = list(rng.permutation(ordered))
        shift = int(rng.integers(1, len(shuffled)))
        mapped = shuffled[shift:] + shuffled[:shift]
        for source, target in zip(shuffled, mapped):
            mapping[source] = target
    return mapping, fixed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--layer-selection", required=True, type=Path)
    parser.add_argument("--allocation-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--random-draws", type=int, default=1000)
    parser.add_argument("--permutation-draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    block_index = int(json.loads(args.layer_selection.read_text())["selected_cache_index"])
    data = {language: load_language(args.prompt_root, args.cache_root, language, block_index)
            for language in LANGUAGES}
    rng = np.random.default_rng(args.seed)
    width = data["en"]["train"]["x"].shape[1]

    isotropic = rng.normal(size=(args.random_draws, width)).astype(np.float32)
    isotropic /= np.linalg.norm(isotropic, axis=1, keepdims=True)
    isotropic_by_language = {}
    for language in LANGUAGES:
        scores = isotropic @ np.asarray(data[language]["test"]["x"], dtype=np.float32).T
        isotropic_by_language[language] = summarize(
            [auroc(data[language]["test"]["y"], row) for row in scores])
    independent_a = rng.normal(size=(args.random_draws, width)).astype(np.float32)
    independent_b = rng.normal(size=(args.random_draws, width)).astype(np.float32)
    independent_a /= np.linalg.norm(independent_a, axis=1, keepdims=True)
    independent_b /= np.linalg.norm(independent_b, axis=1, keepdims=True)
    isotropic_cosine = summarize(np.sum(independent_a * independent_b, axis=1))

    covariance_directions = {}
    covariance_auroc = {}
    for language in LANGUAGES:
        directions = random_label_directions(data[language]["train"]["x"],
                                             data[language]["train"]["y"], args.random_draws, rng)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        covariance_directions[language] = directions
        scores = directions @ np.asarray(data[language]["test"]["x"], dtype=np.float32).T
        covariance_auroc[language] = summarize(
            [auroc(data[language]["test"]["y"], row) for row in scores])
    covariance_cosines = []
    for i, left in enumerate(LANGUAGES):
        for right in LANGUAGES[i + 1:]:
            covariance_cosines.append({"left": left, "right": right,
                                       **summarize(np.sum(covariance_directions[left] *
                                                          covariance_directions[right], axis=1))})

    pair_controls = []
    for i, left in enumerate(LANGUAGES):
        for right in LANGUAGES[i + 1:]:
            allocation = json.loads((args.allocation_root / f"{left}__{right}.json").read_text())
            full = next(plan for plan in allocation["plans"]
                        if plan["repetition"] == 0 and plan["shared_groups"] == 400)
            ids_left, ids_right = full["source_group_ids"], full["target_group_ids"]
            y_left = np.asarray([data[left]["train"]["lookup"][gid][1] for gid in ids_left])
            y_right = np.asarray([data[right]["train"]["lookup"][gid][1] for gid in ids_right])
            label_null = []
            for _ in range(args.permutation_draws):
                shuffled_left = shuffled_stratified_labels(ids_left, y_left, allocation["group_annotations"], rng)
                shuffled_right = shuffled_stratified_labels(ids_right, y_right, allocation["group_annotations"], rng)
                d_left = fit_with_labels(data[left]["train"]["lookup"], ids_left, shuffled_left)
                d_right = fit_with_labels(data[right]["train"]["lookup"], ids_right, shuffled_right)
                label_null.append(cosine(d_left, d_right))
            labels_right = {gid: value[1] for gid, value in data[right]["train"]["lookup"].items()}
            pool = [gid for gid in allocation["group_annotations"] if gid in labels_right]
            source_probe = fit_ids(data[left]["train"]["lookup"], ids_left)
            identity_null, fixed_counts = [], []
            for _ in range(args.permutation_draws):
                mapping, fixed = identity_mapping(pool, labels_right, allocation["group_annotations"], rng)
                mapped = [mapping[gid] for gid in ids_left]
                target_probe = fit_ids(data[right]["train"]["lookup"], mapped)
                identity_null.append(cosine(source_probe.direction, target_probe.direction))
                fixed_counts.append(fixed)
            observed_target = fit_ids(data[right]["train"]["lookup"], ids_right)
            pair_controls.append({
                "left": left, "right": right,
                "observed_full_overlap_cosine_rep0": cosine(source_probe.direction, observed_target.direction),
                "independent_stratified_label_permutation": summarize(label_null),
                "fact_identity_permutation": summarize(identity_null),
                "fact_identity_unavoidable_fixed_pool_groups": summarize(fixed_counts),
            })
    result = {
        "schema_version": 1, "model_id": args.model_id,
        "selected_block_number": block_index + 1,
        "isotropic_random_direction_cosine": isotropic_cosine,
        "isotropic_random_direction_test_auroc": isotropic_by_language,
        "covariance_matched_random_label_test_auroc": covariance_auroc,
        "covariance_matched_cross_language_cosine": covariance_cosines,
        "pair_controls": pair_controls,
        "notes": [
            "Covariance-matched directions are mass-mean contrasts after exact-count random relabeling of training activations.",
            "Label permutations are independent by language and preserve non-label allocation strata.",
            "Fact-identity permutations rotate the eligible pool within label and non-label stratum; unavoidable singleton strata are counted."
        ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pair_controls": len(pair_controls)}))


if __name__ == "__main__":
    main()
