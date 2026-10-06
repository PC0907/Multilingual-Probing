#!/usr/bin/env python3
"""X13: probe-family robustness (mass mean, shrinkage LDA, L2 logistic regression)
at the frozen layer (protocol/X10_X11_PREREGISTRATION.md)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .analyze_transfer_predictors import PAIRS, TargetGeometry, auroc_matrix, cosine, load_groups, spearman
from study_truth_transport_v2.src.data import LANGUAGES
from study_truth_transport_v2.src.metrics import balanced_accuracy

C_GRID = (0.001, 0.01, 0.1, 1.0)
REPS = 10


def lda_direction(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    geo = TargetGeometry(x, y)
    if x.shape[1] <= len(x) or geo.b <= 0:
        # Few dimensions (or no shrinkage): solve the shrunk covariance directly.
        sigma = geo.a * geo.z.T @ geo.z / geo.n + geo.b * np.eye(x.shape[1])
        w = np.linalg.solve(sigma, geo.delta)
    else:
        gram = geo.z @ geo.z.T
        zd = geo.z @ geo.delta
        alpha = np.linalg.solve((geo.n * geo.b / geo.a) * np.eye(geo.n) + gram, zd)
        w = (geo.delta - geo.z.T @ alpha) / geo.b  # Sigma^{-1} delta via Woodbury
    pos, neg = x[y == 1].mean(axis=0), x[y == 0].mean(axis=0)
    return w, float((pos + neg) @ w / 2)


def logistic(x: np.ndarray, y: np.ndarray, c: float) -> tuple[np.ndarray, float]:
    """L2 logistic regression, sklearn convention: C * sum(logloss) + 0.5 ||w||^2 (intercept unpenalized)."""
    import torch
    device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() and torch.cuda.is_available() else "cpu"
    xt = torch.as_tensor(x, dtype=torch.float64, device=device)
    yt = torch.as_tensor(y, dtype=torch.float64, device=device)
    scale = xt.std(dim=0).mean().clamp_min(1e-8)  # one global scale keeps C comparable across models
    xs = xt / scale
    w = torch.zeros(xs.shape[1], dtype=torch.float64, device=device, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=200, tolerance_grad=1e-9, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = xs @ w + b
        loss = c * torch.nn.functional.binary_cross_entropy_with_logits(logits, yt, reduction="sum") + 0.5 * (w @ w)
        loss.backward()
        return loss
    opt.step(closure)
    weight = (w.detach() / scale).cpu().numpy()
    return weight, float(-b.detach().cpu())  # score = x.w - threshold


def fit(family: str, x: np.ndarray, y: np.ndarray, c: float | None):
    x = np.asarray(x, dtype=np.float64)
    if family == "mm":
        pos, neg = x[y == 1].mean(axis=0), x[y == 0].mean(axis=0)
        return pos - neg, float((pos + neg) @ (pos - neg) / 2)
    if family == "lda":
        return lda_direction(x, y)
    return logistic(x, y, c)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--layer-selection", required=True, type=Path)
    parser.add_argument("--allocation-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reps", type=int, default=REPS)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Refusing existing output: {args.output}")
    block = int(json.loads(args.layer_selection.read_text())["selected_cache_index"])
    raw = {l: load_groups(args.prompt_root, args.cache_root, l) for l in LANGUAGES}
    d = {l: {p: {"x": raw[l][p]["x"][:, block].astype(np.float64), "y": raw[l][p]["y"], "ids": raw[l][p]["ids"]}
             for p in ("train", "validation", "test")} for l in LANGUAGES}
    del raw
    # C chosen on validation only.
    c_scores = {}
    for c in C_GRID:
        vals = []
        for l in LANGUAGES:
            w, _ = fit("lr", d[l]["train"]["x"], d[l]["train"]["y"], c)
            vals.append(float(auroc_matrix((d[l]["validation"]["x"] @ w)[None], d[l]["validation"]["y"])[0]))
        c_scores[c] = float(np.mean(vals))
    c_best = max(c_scores, key=c_scores.get)
    geometry = {l: TargetGeometry(d[l]["train"]["x"], d[l]["train"]["y"]) for l in LANGUAGES}
    result = {"schema_version": 1, "model_id": args.model_id, "frozen_block_number": block + 1,
              "protocol": "protocol/X10_X11_PREREGISTRATION.md (X13)", "lr_c_validation": c_scores, "lr_c": c_best,
              "families": {}}
    for family in ("mm", "lda", "lr"):
        c = c_best if family == "lr" else None
        probes = {l: fit(family, d[l]["train"]["x"], d[l]["train"]["y"], c) for l in LANGUAGES}
        rows = []
        for s, t in PAIRS:
            w, thr = probes[s]
            scores = d[t]["test"]["x"] @ w
            rows.append({"source": s, "target": t, "auroc": float(auroc_matrix(scores[None], d[t]["test"]["y"])[0]),
                         "balanced_accuracy": float(balanced_accuracy(d[t]["test"]["y"], scores - thr)),
                         "cosine": cosine(w, probes[t][0]), "fisher": geometry[t].fisher_efficiency(w)})
        native = [float(auroc_matrix((d[l]["test"]["x"] @ probes[l][0])[None], d[l]["test"]["y"])[0]) for l in LANGUAGES]
        auc = [r["auroc"] for r in rows]
        # Overlap dose response on the first `reps` allocations per level.
        lookup = {l: {g: i for i, g in enumerate(d[l]["train"]["ids"])} for l in LANGUAGES}
        dose = []
        for path in sorted(p for p in args.allocation_root.glob("*__*.json") if not p.name.startswith(".")):
            allocation = json.loads(path.read_text(encoding="utf-8"))
            left, right = allocation["language_pair"]
            for plan in allocation["plans"]:
                if plan["repetition"] >= args.reps:
                    continue
                fits = {}
                for lang, ids in ((left, plan["source_group_ids"]), (right, plan["target_group_ids"])):
                    ix = np.asarray([lookup[lang][g] for g in ids])
                    fits[lang] = fit(family, d[lang]["train"]["x"][ix], d[lang]["train"]["y"][ix], c)
                cos_lr = cosine(fits[left][0], fits[right][0])
                for s, t in ((left, right), (right, left)):
                    scores = d[t]["test"]["x"] @ fits[s][0]
                    dose.append({"source": s, "target": t, "overlap": float(plan["requested_overlap_fraction"]),
                                 "cosine": cos_lr, "auroc": float(auroc_matrix(scores[None], d[t]["test"]["y"])[0])})
        slopes = {"cosine": [], "auroc": []}
        for s, t in PAIRS:
            sub = [r for r in dose if r["source"] == s and r["target"] == t]
            levels = sorted({r["overlap"] for r in sub})
            for key in slopes:
                means = [np.mean([r[key] for r in sub if r["overlap"] == o]) for o in levels]
                slopes[key].append(float(np.polyfit(levels, means, 1)[0]))
        summary = {}
        for key, values in slopes.items():
            v = np.asarray(values)
            half = 2.045 * v.std(ddof=1) / np.sqrt(len(v))  # t(29) 97.5%
            summary[key] = {"mean": float(v.mean()), "ci95": [float(v.mean() - half), float(v.mean() + half)]}
        result["families"][family] = {
            "mean_cross_auroc": float(np.mean(auc)), "mean_native_auroc": float(np.mean(native)),
            "mean_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in rows])),
            "mean_cosine": float(np.mean([r["cosine"] for r in rows])),
            "spearman_cosine": spearman([r["cosine"] for r in rows], auc),
            "spearman_fisher": spearman([r["fisher"] for r in rows], auc),
            "overlap_slopes": summary, "cells": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "lr_c": c_best,
                      "families": {f: {k: v for k, v in r.items() if k in ("mean_cross_auroc", "overlap_slopes", "spearman_cosine", "spearman_fisher")}
                                   for f, r in result["families"].items()}}))


if __name__ == "__main__":
    main()
