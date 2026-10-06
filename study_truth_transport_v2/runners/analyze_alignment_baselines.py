#!/usr/bin/env python3
"""Compare fixed-layer cross-lingual alignment baselines on frozen splits.

The transformations are fitted without labels on parallel training groups.  A
source-language mass-mean probe is then evaluated on unchanged target test
groups.  RoSh is the standard full orthogonal-Procrustes rotation plus centroid
shift at the one preregistered layer; it is not the paper's greedy three-layer
recipe.  The PCA and mean-shift arms are raw-space LSI ablations, not the LSI
autoencoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .analyze_rq_a import load_language
from study_truth_transport_v2.src.data import LANGUAGES
from study_truth_transport_v2.src.metrics import score_transport
from study_truth_transport_v2.src.probes import fit_mass_mean
from study_truth_transport_v2.src.bootstrap import stratified_group_bootstrap_indices


def paired(data_a: dict, data_b: dict, partition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    a, b = data_a[partition], data_b[partition]
    ids = sorted(set(a["ids"]) & set(b["ids"]))
    if not ids:
        raise ValueError("No aligned dependency groups")
    xa = np.asarray([a["lookup"][gid][0] for gid in ids], dtype=np.float32)
    xb = np.asarray([b["lookup"][gid][0] for gid in ids], dtype=np.float32)
    ya = np.asarray([a["lookup"][gid][1] for gid in ids], dtype=int)
    yb = np.asarray([b["lookup"][gid][1] for gid in ids], dtype=int)
    if not np.array_equal(ya, yb):
        raise ValueError("Aligned languages disagree on labels")
    return xa, xb, ya, ids


def score_result(y: np.ndarray, score: np.ndarray) -> dict:
    out = score_transport(y, score, source_threshold=0.0)
    return {key: (float(value) if value is not None else None) for key, value in out.items()}


def derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    if n < 2:
        raise ValueError("Derangement needs at least two rows")
    order = np.arange(n)
    for _ in range(1000):
        candidate = rng.permutation(n)
        if np.all(candidate != order):
            return candidate
    return np.roll(order, 1)


def top_difference_pc(a: np.ndarray, b: np.ndarray, device: str = "cpu") -> np.ndarray:
    difference = np.asarray(a - b, dtype=np.float64)
    difference -= difference.mean(axis=0, keepdims=True)
    # The right singular vector is exact and avoids a d x d covariance matrix.
    # CUDA matters here: a 1,200 x 4,096 CPU SVD dominates this analysis.
    if device == "cuda":
        import torch

        matrix = torch.as_tensor(difference, dtype=torch.float32, device="cuda:0")
        _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
        vector = vh[0].float().cpu().numpy()
        del matrix, vh
        torch.cuda.empty_cache()
    else:
        _, _, vh = np.linalg.svd(difference, full_matrices=False)
        vector = vh[0]
    return vector / np.linalg.norm(vector)


def ridge_score_vector(target_centered: np.ndarray, source_centered: np.ndarray,
                       source_direction: np.ndarray, ridge_fraction: float,
                       device: str = "cpu") -> np.ndarray:
    """Return W @ direction for W minimizing ||XW-Y||^2 + lambda||W||^2.

    The dual form avoids materializing a hidden_size x hidden_size map.
    """
    x = np.asarray(target_centered, dtype=np.float64)
    y = np.asarray(source_centered, dtype=np.float64)
    gram = x @ x.T
    scale = float(np.trace(gram) / len(gram))
    lam = max(scale * ridge_fraction, np.finfo(np.float64).eps)
    rhs = y @ np.asarray(source_direction, dtype=np.float64)
    if device == "cuda":
        import torch

        xt = torch.as_tensor(x, dtype=torch.float32, device="cuda:0")
        gram_t = xt @ xt.T
        rhs_t = torch.as_tensor(rhs, dtype=torch.float32, device="cuda:0")
        dual = torch.linalg.solve(
            gram_t + float(lam) * torch.eye(len(gram_t), device="cuda:0"), rhs_t)
        answer = (xt.T @ dual).float().cpu().numpy()
        del xt, gram_t, rhs_t, dual
        torch.cuda.empty_cache()
        return answer
    dual = np.linalg.solve(gram + lam * np.eye(len(gram)), rhs)
    return x.T @ dual


def procrustes_score_vectors(a: np.ndarray, b: np.ndarray, directions: list[np.ndarray],
                             device: str) -> tuple[list[np.ndarray], dict]:
    """Fit R=UV^T for min ||A R-B|| and return R @ each column direction."""
    import torch

    if device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if not visible or "," in visible or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one GPU for CUDA Procrustes")
        torch_device = torch.device("cuda:0")
    else:
        torch_device = torch.device("cpu")
    at = torch.as_tensor(a, dtype=torch.float32, device=torch_device)
    bt = torch.as_tensor(b, dtype=torch.float32, device=torch_device)
    cross = at.T @ bt
    u, singular, vh = torch.linalg.svd(cross, full_matrices=True)
    projected = []
    for direction in directions:
        w = torch.as_tensor(direction, dtype=torch.float32, device=torch_device)
        projected.append((u @ (vh @ w)).float().cpu().numpy())
    diagnostics = {
        "cross_covariance_rank_tolerance": int((singular > singular.max() * 1e-6).sum().item()),
        "largest_singular_value": float(singular.max().item()),
        "smallest_returned_singular_value": float(singular.min().item()),
    }
    del at, bt, cross, u, singular, vh
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()
    return projected, diagnostics


def evaluate_direction(x: np.ndarray, y: np.ndarray, direction: np.ndarray,
                       intercept: float) -> dict:
    return score_result(y, np.asarray(x, dtype=np.float64) @ direction + intercept)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted, running, count = {}, 0.0, len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted


def paired_macro_bootstrap(score_records: list[dict], draws: int, seed: int) -> dict:
    """Synchronous dependency-group bootstrap of macro pair-level changes."""
    method_names = list(score_records[0]["scores"])
    baseline = "raw_source_mass_mean"
    outcomes = ("auroc", "balanced_accuracy", "standardized_separation")
    reference_ids = score_records[0]["ids"]
    if any(record["ids"] != reference_ids for record in score_records[1:]):
        raise ValueError("Ordered test dependency groups differ across language pairs")
    observed_metrics = [
        {method: score_result(record["y"], record["scores"][method])
         for method in method_names}
        for record in score_records
    ]
    observed = {
        method: {
            outcome: float(np.mean([
                metrics[method][outcome] - metrics[baseline][outcome]
                for metrics in observed_metrics
            ]))
            for outcome in outcomes
        }
        for method in method_names if method != baseline
    }
    first = score_records[0]
    indices = list(stratified_group_bootstrap_indices(
        first["y"], np.asarray(first["ids"]), draws, seed))
    samples = {
        method: {outcome: [] for outcome in outcomes}
        for method in observed
    }
    for ix in indices:
        resampled = []
        for record in score_records:
            y = record["y"][ix]
            resampled.append({
                method: score_result(y, record["scores"][method][ix])
                for method in method_names
            })
        for method in observed:
            deltas = {outcome: [] for outcome in outcomes}
            for metrics in resampled:
                current, raw = metrics[method], metrics[baseline]
                for outcome in outcomes:
                    deltas[outcome].append(current[outcome] - raw[outcome])
            for outcome in outcomes:
                samples[method][outcome].append(float(np.mean(deltas[outcome])))
    result = {}
    raw_p = {outcome: {} for outcome in outcomes}
    for method in observed:
        result[method] = {}
        for outcome in outcomes:
            values = np.asarray(samples[method][outcome])
            lower = (np.count_nonzero(values <= 0) + 1) / (len(values) + 1)
            upper = (np.count_nonzero(values >= 0) + 1) / (len(values) + 1)
            p_value = min(1.0, 2 * min(lower, upper))
            raw_p[outcome][method] = p_value
            result[method][outcome] = {
                "mean_change": observed[method][outcome],
                "ci95": [float(np.quantile(values, .025)), float(np.quantile(values, .975))],
                "p_two_sided": float(p_value),
            }
    for outcome in outcomes:
        adjusted = holm_adjust(raw_p[outcome])
        for method, value in adjusted.items():
            result[method][outcome]["p_holm"] = float(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--layer-selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--ridge-fraction", type=float, default=0.01)
    parser.add_argument("--lsi-mean-shift-alpha", type=float, default=0.6)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Refusing existing output: {args.output}")

    selection = json.loads(args.layer_selection.read_text(encoding="utf-8"))
    block_index = int(selection["selected_cache_index"])
    data = {lang: load_language(args.prompt_root, args.cache_root, lang, block_index)
            for lang in LANGUAGES}
    probes = {lang: fit_mass_mean(data[lang]["train"]["x"], data[lang]["train"]["y"])
              for lang in LANGUAGES}
    cells, score_records = [], []
    rng = np.random.default_rng(args.seed)

    for left_index, left in enumerate(LANGUAGES):
        for right in LANGUAGES[left_index + 1:]:
            xl, xr, _, train_ids = paired(data[left], data[right], "train")
            mu_l, mu_r = xl.mean(axis=0), xr.mean(axis=0)
            al, ar = xl - mu_l, xr - mu_r
            permutation = derangement(len(train_ids), rng)

            # R maps left -> right.  Its transpose maps right -> left.
            directions = [probes[right].direction, probes[left].direction]
            mapped, rosh_diag = procrustes_score_vectors(al, ar, directions, args.device)
            r_w_right, rt_w_left = mapped
            scrambled, scrambled_diag = procrustes_score_vectors(
                al[permutation], ar, directions, args.device)
            scrambled_r_w_right, scrambled_rt_w_left = scrambled

            pc_lr = top_difference_pc(al, ar, args.device)
            for source, target, xt, xs, mu_t, mu_s, centered_t, centered_s, rosh_w, scrambled_w, pc in (
                (left, right, data[right]["test"], data[left]["test"], mu_r, mu_l, ar, al,
                 rt_w_left, scrambled_rt_w_left, pc_lr),
                (right, left, data[left]["test"], data[right]["test"], mu_l, mu_r, al, ar,
                 r_w_right, scrambled_r_w_right, pc_lr),
            ):
                probe = probes[source]
                x_test, y_test = xt["x"], xt["y"]
                raw = probe.score(x_test)
                method_scores = {"raw_source_mass_mean": raw}

                # Target-to-source centroid shift.
                shift_intercept = float((mu_s - mu_t) @ probe.direction - probe.threshold)
                method_scores["centroid_shift"] = np.asarray(x_test, dtype=np.float64) @ probe.direction + shift_intercept

                # Orthogonal rotation plus centroid shift.
                rosh_intercept = float(mu_s @ probe.direction - mu_t @ rosh_w - probe.threshold)
                method_scores["rosh_fixed_layer"] = np.asarray(x_test, dtype=np.float64) @ rosh_w + rosh_intercept
                scrambled_intercept = float(
                    mu_s @ probe.direction - mu_t @ scrambled_w - probe.threshold)
                method_scores["rosh_scrambled_correspondence"] = (
                    np.asarray(x_test, dtype=np.float64) @ scrambled_w + scrambled_intercept)

                # Unconstrained ridge map through its induced score vector.
                ridge_w = ridge_score_vector(centered_t, centered_s, probe.direction,
                                             args.ridge_fraction, args.device)
                ridge_intercept = float(mu_s @ probe.direction - mu_t @ ridge_w - probe.threshold)
                method_scores["ridge_unconstrained"] = (
                    np.asarray(x_test, dtype=np.float64) @ ridge_w + ridge_intercept)

                # Raw-space LSI paper ablations.
                pca_w = probe.direction - pc * float(pc @ probe.direction)
                pca_intercept = float(-probe.threshold)
                method_scores["lsi_raw_pca1"] = (
                    np.asarray(x_test, dtype=np.float64) @ pca_w + pca_intercept)
                alpha = args.lsi_mean_shift_alpha
                mean_shift_intercept = float(alpha * (mu_s - mu_t) @ probe.direction - probe.threshold)
                method_scores["lsi_raw_mean_shift"] = (
                    np.asarray(x_test, dtype=np.float64) @ probe.direction + mean_shift_intercept)

                native = probes[target]
                method_scores["target_native_mass_mean"] = native.score(x_test)
                methods = {name: score_result(y_test, score) for name, score in method_scores.items()}
                cells.append({
                    "source": source,
                    "target": target,
                    "n_parallel_train_groups": len(train_ids),
                    "n_test_groups": len(y_test),
                    "methods": methods,
                    "rosh_diagnostics": rosh_diag,
                    "scrambled_rosh_diagnostics": scrambled_diag,
                })
                score_records.append({
                    "source": source, "target": target,
                    "ids": list(xt["ids"]), "y": np.asarray(y_test, dtype=int),
                    "scores": {name: np.asarray(score, dtype=float)
                               for name, score in method_scores.items()},
                })

    methods = list(cells[0]["methods"])
    summaries = {}
    for method in methods:
        summaries[method] = {
            metric: float(np.mean([cell["methods"][method][metric] for cell in cells]))
            for metric in ("auroc", "balanced_accuracy", "standardized_separation")
        }
    result = {
        "schema_version": 1,
        "analysis": "post-core X1 same-split alignment baselines",
        "model_id": args.model_id,
        "selected_block_number": block_index + 1,
        "languages": list(LANGUAGES),
        "n_ordered_pairs": len(cells),
        "fit_partition": "dependency-group-disjoint train; transformations are label blind",
        "test_partition": "unchanged dependency-group-disjoint test",
        "ridge_fraction_of_mean_gram_diagonal": args.ridge_fraction,
        "lsi_raw_mean_shift_alpha": args.lsi_mean_shift_alpha,
        "method_scope": {
            "rosh_fixed_layer": "exact orthogonal Procrustes plus shift at one frozen layer; not greedy three-layer RoSh",
            "lsi_raw_pca1": "raw-space one-component inconsistency removal; not AE-based LSI",
            "lsi_raw_mean_shift": "raw-space centroid shift with lambda 0.6; not AE-based LSI",
        },
        "summary_off_diagonal_means": summaries,
        "paired_macro_bootstrap_vs_raw": paired_macro_bootstrap(
            score_records, args.bootstrap_draws,
            int(hashlib.sha256(f"{args.seed}|{args.model_id}".encode()).hexdigest()[:8], 16)),
        "bootstrap_draws": args.bootstrap_draws,
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "cells": len(cells), "methods": methods}))


if __name__ == "__main__":
    main()
