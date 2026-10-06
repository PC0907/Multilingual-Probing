#!/usr/bin/env python3
"""X10 (estimator-noise model of RQ-A), X11 (translation-quality sensitivity),
X12 (few-shot target labels), frozen in protocol/X10_X11_PREREGISTRATION.md.

Reads one model's verified all-layer cache. X10 uses only pool-level training
statistics at the frozen layer; X11 uses label-blind LaBSE flags; X12 uses
every decoder block and label-balanced draws of k target validation groups.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .analyze_transfer_predictors import (
    PAIRS, TargetGeometry, auroc_matrix, cosine, fit_direction, load_groups, pearson_rows, spearman)
from study_truth_transport_v2.src.data import LANGUAGES
from study_truth_transport_v2.src.metrics import balanced_accuracy

LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
N_PER_CLASS = 200
KS = (8, 16, 32, 64)
DRAWS = 20


class UnlabeledGeometry:
    """Ledoit-Wolf-shrunk total covariance of unlabeled activations via the n x n Gram matrix."""

    def __init__(self, x: np.ndarray):
        z = np.asarray(x, dtype=np.float64)
        z = z - z.mean(axis=0)
        n, d = z.shape
        gram = z @ z.T
        trace_s = np.trace(gram) / n
        nu = trace_s / d
        frob_s2 = float((gram * gram).sum()) / n**2
        delta2 = frob_s2 - nu**2 * d
        quad = (gram * gram).sum(axis=1) / n
        beta2 = float(np.sum(np.diag(gram) ** 2 - 2 * quad + frob_s2)) / n**2
        rho = float(min(beta2, delta2) / delta2)
        self.a, self.b, self.n, self.z = 1 - rho, rho * nu, n, z
        evals, evecs = np.linalg.eigh(gram)
        self.evecs = evecs
        self.inv_eig = 1.0 / (n * self.b / self.a + evals)

    def quadratic(self, w: np.ndarray) -> float:
        zw = self.z @ w
        return float(self.a * (zw @ zw) / self.n + self.b * (w @ w))

    def inverse_quadratic(self, v: np.ndarray) -> float:
        zv = self.z @ v
        proj = self.evecs.T @ zv
        return float((v @ v - (proj * self.inv_eig) @ proj) / self.b)


# ------------------------------------------------------------------- X10
def predicted_cosine(stats: dict, f: float) -> float:
    cross = stats["delta_dot"]
    var_s, var_t = stats["delta_s2"], stats["delta_t2"]
    for y in (0, 1):
        n_y, big_n, c = N_PER_CLASS, stats["N"][y], stats["C"][y]
        k = f * n_y
        cross += (k * c - (n_y**2 - k) * c / (big_n - 1)) / n_y**2
        var_s += stats["V_s"][y] * (big_n - n_y) / (n_y * (big_n - 1))
        var_t += stats["V_t"][y] * (big_n - n_y) / (n_y * (big_n - 1))
    return float(cross / np.sqrt(var_s * var_t))


def pool_statistics(xs: np.ndarray, xt: np.ndarray, y: np.ndarray) -> dict:
    stats = {"N": {}, "V_s": {}, "V_t": {}, "C": {}}
    means_s, means_t = {}, {}
    for cls in (0, 1):
        a, b = xs[y == cls], xt[y == cls]
        means_s[cls], means_t[cls] = a.mean(axis=0), b.mean(axis=0)
        ra, rb = a - means_s[cls], b - means_t[cls]
        stats["N"][cls] = int(len(a))
        stats["V_s"][cls] = float((ra * ra).sum(axis=1).mean())
        stats["V_t"][cls] = float((rb * rb).sum(axis=1).mean())
        stats["C"][cls] = float((ra * rb).sum(axis=1).mean())
    ds, dt = means_s[1] - means_s[0], means_t[1] - means_t[0]
    stats.update(delta_dot=float(ds @ dt), delta_s2=float(ds @ ds), delta_t2=float(dt @ dt))
    return stats


def x10(data: dict, block: int, allocation_root: Path, rqa_summary: dict) -> dict:
    observed = {}
    for cell in rqa_summary["cells"]:
        key = tuple(sorted((cell["source"], cell["target"])))
        observed.setdefault(key, {}).setdefault(cell["overlap_fraction"], []).append(cell["metrics"]["raw_cosine"]["mean"])
    rows = []
    for path in sorted(p for p in allocation_root.glob("*__*.json") if not p.name.startswith(".")):
        allocation = json.loads(path.read_text(encoding="utf-8"))
        s, t = allocation["language_pair"]
        pool = sorted({g for plan in allocation["plans"] for g in plan["source_group_ids"] + plan["target_group_ids"]})
        index = {l: {g: i for i, g in enumerate(data[l]["train"]["ids"])} for l in (s, t)}
        pool = [g for g in pool if g in index[s] and g in index[t]]
        xs = data[s]["train"]["x"][[index[s][g] for g in pool], block].astype(np.float64)
        xt = data[t]["train"]["x"][[index[t][g] for g in pool], block].astype(np.float64)
        y = data[s]["train"]["y"][[index[s][g] for g in pool]]
        stats = pool_statistics(xs, xt, y)
        predicted = [predicted_cosine(stats, f) for f in LEVELS]
        obs = [float(np.mean(observed[tuple(sorted((s, t)))][f])) for f in LEVELS]
        rows.append({"pair": [s, t], "pool_groups": len(pool), "C": stats["C"], "N": stats["N"],
                     "predicted_cosine": predicted, "observed_cosine": obs,
                     "predicted_slope": float(np.polyfit(LEVELS, predicted, 1)[0]),
                     "observed_slope": float(np.polyfit(LEVELS, obs, 1)[0])})
    return {"levels": list(LEVELS), "pairs": rows}


# ------------------------------------------------------------------- X11
def frozen_statistics(data: dict, block: int, flagged: dict[str, set]) -> dict:
    keep = {l: {p: np.asarray([g not in flagged.get(l, set()) for g in data[l][p]["ids"]]) for p in ("train", "validation", "test")}
            for l in LANGUAGES}
    sub = {l: {p: {"x": data[l][p]["x"][keep[l][p], block].astype(np.float64), "y": data[l][p]["y"][keep[l][p]],
                   "ids": [g for g, k in zip(data[l][p]["ids"], keep[l][p]) if k]} for p in ("train", "validation", "test")}
           for l in LANGUAGES}
    probes = {}
    for l in LANGUAGES:
        tr = sub[l]["train"]
        pos, neg = tr["x"][tr["y"] == 1].mean(axis=0), tr["x"][tr["y"] == 0].mean(axis=0)
        probes[l] = (pos - neg, float((pos + neg) @ (pos - neg) / 2))
    geometry = {l: TargetGeometry(sub[l]["train"]["x"], sub[l]["train"]["y"]) for l in LANGUAGES}
    rows = []
    for s, t in PAIRS:
        w, thr = probes[s]
        te = sub[t]["test"]
        scores = te["x"] @ w
        common = sorted(set(sub[s]["validation"]["ids"]) & set(sub[t]["validation"]["ids"]))
        vs = {g: i for i, g in enumerate(sub[s]["validation"]["ids"])}
        vt = {g: i for i, g in enumerate(sub[t]["validation"]["ids"])}
        a = sub[s]["validation"]["x"][[vs[g] for g in common]] @ w
        b = sub[t]["validation"]["x"][[vt[g] for g in common]] @ w
        rows.append({"source": s, "target": t, "auroc": float(auroc_matrix(scores[None], te["y"])[0]),
                     "balanced_accuracy": float(balanced_accuracy(te["y"], scores - thr)),
                     "p1": cosine(w, probes[t][0]), "p2": float(pearson_rows(a[None], b[None])[0]),
                     "p3": geometry[t].fisher_efficiency(w)})
    y = [r["auroc"] for r in rows]
    rho = {k: spearman([r[k] for r in rows], y) for k in ("p1", "p2", "p3")}
    return {"mean_cross_auroc": float(np.mean(y)), "mean_balanced_accuracy": float(np.mean([r["balanced_accuracy"] for r in rows])),
            "spearman": rho, "best": max(rho, key=rho.get), "n_test_groups": {l: len(sub[l]["test"]["ids"]) for l in LANGUAGES}}


# ------------------------------------------------------------------- X12
def x12(data: dict, model_id: str) -> dict:
    n_blocks = data["en"]["train"]["x"].shape[1]
    outcome, direct, fisher, cos = [], {k: [] for k in KS}, {k: [] for k in KS}, {k: [] for k in KS}
    draws = {}
    for t in LANGUAGES:
        yv = data[t]["validation"]["y"]
        pos, neg = np.flatnonzero(yv == 1), np.flatnonzero(yv == 0)
        for k in KS:
            rng = np.random.default_rng(int(hashlib.sha256(f"x12|{model_id}|{t}|{k}".encode()).hexdigest()[:8], 16))
            draws[(t, k)] = [np.concatenate([rng.choice(pos, k // 2, replace=False), rng.choice(neg, k // 2, replace=False)])
                             for _ in range(DRAWS)]
    for block in range(n_blocks):
        directions = {l: fit_direction(data[l]["train"]["x"][:, block].astype(np.float64), data[l]["train"]["y"]) for l in LANGUAGES}
        for t in LANGUAGES:
            geo = UnlabeledGeometry(data[t]["train"]["x"][:, block])
            xv = data[t]["validation"]["x"][:, block].astype(np.float64)
            yv = data[t]["validation"]["y"]
            xte = data[t]["test"]["x"][:, block].astype(np.float64)
            sources = [s for s in LANGUAGES if s != t]
            quad = {s: geo.quadratic(directions[s]) for s in sources}
            for s in sources:
                outcome.append(float(auroc_matrix((xte @ directions[s])[None], data[t]["test"]["y"])[0]))
            for k in KS:
                dir_rows, fis_rows, cos_rows = [], [], []
                for ix in draws[(t, k)]:
                    xs, ys = xv[ix], yv[ix]
                    dmu = xs[ys == 1].mean(axis=0) - xs[ys == 0].mean(axis=0)
                    inv_q = geo.inverse_quadratic(dmu)
                    scores = np.stack([xs @ directions[s] for s in sources])
                    dir_rows.append(auroc_matrix(scores, ys))
                    fis_rows.append([float(directions[s] @ dmu / np.sqrt(quad[s] * inv_q)) for s in sources])
                    cos_rows.append([cosine(directions[s], dmu) for s in sources])
                direct[k].append(np.asarray(dir_rows).T)
                fisher[k].append(np.asarray(fis_rows).T)
                cos[k].append(np.asarray(cos_rows).T)
    outcome = np.asarray(outcome)
    result = {"n_configurations": len(outcome), "ks": list(KS), "draws": DRAWS, "by_k": {}}
    for k in KS:
        stats = {}
        for name, store in (("direct", direct), ("fisher", fisher), ("cos", cos)):
            matrix = np.concatenate(store[k])  # configs x draws
            per_draw = [spearman(matrix[:, r], outcome) for r in range(DRAWS)]
            stats[name] = {"mean_rho": float(np.mean(per_draw)), "p025": float(np.quantile(per_draw, .025)),
                           "p975": float(np.quantile(per_draw, .975)), "per_draw": [float(v) for v in per_draw]}
        diff = np.asarray(stats["fisher"]["per_draw"]) - np.asarray(stats["direct"]["per_draw"])
        stats["fisher_minus_direct"] = {"mean": float(diff.mean()), "p025": float(np.quantile(diff, .025)),
                                        "p975": float(np.quantile(diff, .975))}
        result["by_k"][str(k)] = stats
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--layer-selection", required=True, type=Path)
    parser.add_argument("--allocation-root", required=True, type=Path)
    parser.add_argument("--rqa-summary", required=True, type=Path)
    parser.add_argument("--qe", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-x12", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Refusing existing output: {args.output}")
    block = int(json.loads(args.layer_selection.read_text())["selected_cache_index"])
    data = {lang: load_groups(args.prompt_root, args.cache_root, lang) for lang in LANGUAGES}
    qe = json.loads(args.qe.read_text())
    flagged = {lang: set(item["flagged_group_ids"]) for lang, item in qe["languages"].items()}
    result = {"schema_version": 1, "model_id": args.model_id, "frozen_block_number": block + 1,
              "protocol": "protocol/X10_X11_PREREGISTRATION.md",
              "x10": x10(data, block, args.allocation_root, json.loads(args.rqa_summary.read_text())),
              "x11": {"unfiltered": frozen_statistics(data, block, {}), "filtered": frozen_statistics(data, block, flagged)}}
    if not args.skip_x12:
        result["x12"] = x12(data, args.model_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    pairs = result["x10"]["pairs"]
    print(json.dumps({"output": str(args.output),
                      "x10_slope_r": float(np.corrcoef([p["predicted_slope"] for p in pairs], [p["observed_slope"] for p in pairs])[0, 1]),
                      "x11": {k: {kk: vv for kk, vv in v.items() if kk in ("mean_cross_auroc", "best")} for k, v in result["x11"].items()},
                      "x12_k16": {n: v["mean_rho"] for n, v in result.get("x12", {}).get("by_k", {}).get("16", {}).items() if n != "fisher_minus_direct"}}))


if __name__ == "__main__":
    main()
