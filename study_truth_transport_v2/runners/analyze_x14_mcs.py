#!/usr/bin/env python3
"""X14: Mahalanobis-cosine baselines, parameter-free Proposition-1 prediction, and an
outlier-dimension failure analysis (protocol/X14_PREREGISTRATION.md)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .analyze_transfer_predictors import PAIRS, TargetGeometry, auroc_matrix, cosine, fit_direction, load_groups, spearman
from .analyze_x10_x12 import LEVELS, pool_statistics, predicted_cosine
from study_truth_transport_v2.src.data import LANGUAGES

OUTLIER_FACTOR = 20.0
REPS = 10


def lda_direction(geo: TargetGeometry) -> np.ndarray:
    """Sigma_pool^{-1} Delta via Woodbury (Sigma = a S + b I, S = Z'Z/n)."""
    if geo.a <= 0:
        return geo.delta / geo.b
    gram = geo.z @ geo.z.T
    alpha = np.linalg.solve((geo.n * geo.b / geo.a) * np.eye(geo.n) + gram, geo.z @ geo.delta)
    return (geo.delta - geo.z.T @ alpha) / geo.b


class TotalCovariance:
    """Quadratic forms under the (unshrunk) total sample covariance of x."""

    def __init__(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        self.z = x - x.mean(axis=0)
        self.n = len(x)

    def inner(self, u: np.ndarray, v: np.ndarray) -> float:
        return float((self.z @ u) @ (self.z @ v) / self.n)

    def cosine(self, u: np.ndarray, v: np.ndarray) -> float:
        return float(self.inner(u, v) / math.sqrt(self.inner(u, u) * self.inner(v, v)))


def phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def block_configs(data: dict, block: int, keep: np.ndarray | None = None) -> list[dict]:
    def X(lang, part):
        x = data[lang][part]["x"][:, block].astype(np.float64)
        return x if keep is None else x[:, keep]
    directions, geometry, total, optimal = {}, {}, {}, {}
    for lang in LANGUAGES:
        x, y = X(lang, "train"), data[lang]["train"]["y"]
        directions[lang] = fit_direction(x, y)
        geometry[lang] = TargetGeometry(x, y)
        total[lang] = TotalCovariance(x)
        optimal[lang] = lda_direction(geometry[lang])
    tests = {lang: X(lang, "test") for lang in LANGUAGES}
    rows = []
    for s, t in PAIRS:
        w = directions[s]
        auc = float(auroc_matrix((tests[t] @ w)[None], data[t]["test"]["y"])[0])
        f = geometry[t].fisher_efficiency(w)
        d_star = math.sqrt(max(geometry[t].q, 0.0))
        rows.append({"block_number": block + 1, "source": s, "target": t, "auroc": auc,
                     "p1_euclidean_cosine": cosine(w, directions[t]),
                     "p3_fisher_efficiency": f,
                     "p4a_mcs_total_vs_target_probe": total[t].cosine(w, directions[t]),
                     "p4b_mcs_total_vs_target_optimal": total[t].cosine(w, optimal[t]),
                     "p5_prop1_predicted_auroc": phi(f * d_star / math.sqrt(2.0)),
                     "target_d_star": d_star})
    return rows


STATS = ("p1_euclidean_cosine", "p3_fisher_efficiency", "p4a_mcs_total_vs_target_probe",
         "p4b_mcs_total_vs_target_optimal", "p5_prop1_predicted_auroc")


def rhos(rows: list[dict]) -> dict:
    y = [r["auroc"] for r in rows]
    return {k: spearman([r[k] for r in rows], y) for k in STATS}


def overlap_slopes(data: dict, block: int, allocation_root: Path, keep: np.ndarray | None, reps: int) -> list[dict]:
    """Observed (refit, first `reps` allocations per level) and X10-predicted cosine slopes per pair."""
    out = []
    for path in sorted(p for p in allocation_root.glob("*__*.json") if not p.name.startswith(".")):
        allocation = json.loads(path.read_text(encoding="utf-8"))
        s, t = allocation["language_pair"]
        index = {l: {g: i for i, g in enumerate(data[l]["train"]["ids"])} for l in (s, t)}
        def X(lang, ids):
            x = data[lang]["train"]["x"][[index[lang][g] for g in ids], block].astype(np.float64)
            return x if keep is None else x[:, keep]
        def Y(lang, ids):
            return data[lang]["train"]["y"][[index[lang][g] for g in ids]]
        by_level = {}
        for plan in allocation["plans"]:
            if plan["repetition"] >= reps:
                continue
            a, b = plan["source_group_ids"], plan["target_group_ids"]
            c = cosine(fit_direction(X(s, a), Y(s, a)), fit_direction(X(t, b), Y(t, b)))
            by_level.setdefault(float(plan["requested_overlap_fraction"]), []).append(c)
        levels = sorted(by_level)
        observed = [float(np.mean(by_level[f])) for f in levels]
        pool = sorted({g for plan in allocation["plans"] for g in plan["source_group_ids"] + plan["target_group_ids"]})
        pool = [g for g in pool if g in index[s] and g in index[t]]
        stats = pool_statistics(X(s, pool), X(t, pool), Y(s, pool))
        predicted = [predicted_cosine(stats, f) for f in LEVELS]
        out.append({"pair": [s, t], "observed_slope": float(np.polyfit(levels, observed, 1)[0]),
                    "predicted_slope": float(np.polyfit(LEVELS, predicted, 1)[0])})
    return out


def corr(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--prompt-root", required=True, type=Path)
    ap.add_argument("--cache-root", required=True, type=Path)
    ap.add_argument("--layer-selection", required=True, type=Path)
    ap.add_argument("--allocation-root", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--reps", type=int, default=REPS)
    args = ap.parse_args()
    if args.output.exists():
        raise ValueError(f"Refusing existing output: {args.output}")
    frozen = int(json.loads(args.layer_selection.read_text())["selected_cache_index"])
    data = {l: load_groups(args.prompt_root, args.cache_root, l) for l in LANGUAGES}
    n_blocks = data["en"]["train"]["x"].shape[1]
    configs = [row for block in range(n_blocks) for row in block_configs(data, block)]
    for c in configs:
        c["frozen_block"] = c["block_number"] == frozen + 1

    # H14d: label-free outlier dimensions at the frozen block.
    pooled = np.concatenate([data[l]["train"]["x"][:, frozen].astype(np.float64) for l in LANGUAGES])
    var = pooled.var(axis=0)
    outliers = np.flatnonzero(var > OUTLIER_FACTOR * np.median(var))
    keep = np.setdiff1d(np.arange(pooled.shape[1]), outliers)
    reduced_rows = block_configs(data, frozen, keep) if len(outliers) else [c for c in configs if c["frozen_block"]]
    raw_slopes = overlap_slopes(data, frozen, args.allocation_root, None, args.reps)
    red_slopes = overlap_slopes(data, frozen, args.allocation_root, keep, args.reps) if len(outliers) else raw_slopes
    frozen_rows = [c for c in configs if c["frozen_block"]]
    result = {
        "schema_version": 1, "protocol": "protocol/X14_PREREGISTRATION.md", "model_id": args.model_id,
        "n_blocks": n_blocks, "frozen_block_number": frozen + 1, "n_configurations": len(configs),
        "spearman_all_blocks": rhos(configs), "spearman_frozen_block": rhos(frozen_rows),
        "outlier_analysis": {
            "factor": OUTLIER_FACTOR, "n_outlier_dims": int(len(outliers)), "outlier_dims": outliers.tolist(),
            "outlier_variance_share": float(var[outliers].sum() / var.sum()) if len(outliers) else 0.0,
            "spearman_frozen_raw": rhos(frozen_rows), "spearman_frozen_reduced": rhos(reduced_rows),
            "x10_slope_r_raw": corr([p["predicted_slope"] for p in raw_slopes], [p["observed_slope"] for p in raw_slopes]),
            "x10_slope_r_reduced": corr([p["predicted_slope"] for p in red_slopes], [p["observed_slope"] for p in red_slopes]),
            "pairs_raw": raw_slopes, "pairs_reduced": red_slopes, "reps": args.reps},
        "configurations": configs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    oa = result["outlier_analysis"]
    print(json.dumps({"output": str(args.output), "rho_all": result["spearman_all_blocks"],
                      "n_outliers": oa["n_outlier_dims"], "x10_r_raw": oa["x10_slope_r_raw"], "x10_r_reduced": oa["x10_slope_r_reduced"]}))


if __name__ == "__main__":
    main()
