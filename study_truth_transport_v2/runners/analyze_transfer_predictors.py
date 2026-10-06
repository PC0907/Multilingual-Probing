#!/usr/bin/env python3
"""X7: does label-free translation consistency predict cross-lingual truth transfer?

Frozen in protocol/X7_X9_PREREGISTRATION.md. For every decoder block and ordered
language pair, a source mass-mean probe (training partition) is scored on target
test groups (outcome AUROC) and compared with three predictors:

P1 Euclidean cosine between source and target mass-mean directions;
P2 translation consistency: Pearson correlation of the source probe's scores on
   source- and target-language versions of the same validation groups (no labels);
P3 Fisher efficiency of the source direction in the target's within-class
   geometry (Ledoit-Wolf shrinkage; secondary, uses target training labels).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .cache_io import assert_row_alignment, load_complete_cache
from study_truth_transport_v2.src.data import LANGUAGES, group_index

PAIRS = [(s, t) for s in LANGUAGES for t in LANGUAGES if s != t]


# ----------------------------------------------------------------- primitives
def average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with ties, along the last axis."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, axis=-1, kind="mergesort")
    sorted_values = np.take_along_axis(values, order, axis=-1)
    ranks_sorted = np.broadcast_to(np.arange(1, values.shape[-1] + 1, dtype=np.float64), values.shape).copy()
    # Average tied blocks.
    for index in np.ndindex(values.shape[:-1]):
        row, rank_row = sorted_values[index], ranks_sorted[index]
        start = 0
        while start < len(row):
            stop = start + 1
            while stop < len(row) and row[stop] == row[start]:
                stop += 1
            if stop - start > 1:
                rank_row[start:stop] = (start + 1 + stop) / 2
            start = stop
    ranks = np.empty_like(ranks_sorted)
    np.put_along_axis(ranks, order, ranks_sorted, axis=-1)
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = average_ranks(np.asarray(a)), average_ranks(np.asarray(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def auroc_matrix(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """AUROC for each row of scores (configs x samples); ties count one half."""
    labels = np.asarray(labels, dtype=bool)
    positive, negative = scores[:, labels], scores[:, ~labels]
    greater = (positive[:, :, None] > negative[:, None, :]).sum(axis=(1, 2))
    ties = (positive[:, :, None] == negative[:, None, :]).sum(axis=(1, 2))
    return (greater + 0.5 * ties) / (positive.shape[1] * negative.shape[1])


def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    denominator = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
    return np.where(denominator > 0, (a * b).sum(axis=-1) / np.where(denominator > 0, denominator, 1), np.nan)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


class TargetGeometry:
    """Within-class covariance with analytic Ledoit-Wolf shrinkage via an n x n Gram matrix."""

    def __init__(self, x: np.ndarray, y: np.ndarray, shrinkage: float | None = None):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=int)
        mu_pos, mu_neg = x[y == 1].mean(axis=0), x[y == 0].mean(axis=0)
        self.delta = mu_pos - mu_neg
        z = x - np.where(y[:, None] == 1, mu_pos, mu_neg)
        n, d = z.shape
        gram = z @ z.T
        trace_s = np.trace(gram) / n
        nu = trace_s / d
        frob_s2 = float((gram * gram).sum()) / n**2
        delta2 = frob_s2 - nu**2 * d
        sq_norms = np.diag(gram)
        quad = (gram * gram).sum(axis=1) / n  # z_i' S z_i
        beta2 = float(np.sum(sq_norms**2 - 2 * quad + frob_s2)) / n**2
        rho = float(min(beta2, delta2) / delta2) if shrinkage is None else float(shrinkage)
        self.rho, self.nu, self.n, self.z = rho, nu, n, z
        self.a, self.b = 1 - rho, rho * nu  # Sigma = a S + b I
        zd = z @ self.delta
        system = (n * self.b / self.a) * np.eye(n) + gram if self.a > 0 else None
        if self.a > 0:
            self.q = float((self.delta @ self.delta - zd @ np.linalg.solve(system, zd)) / self.b)
        else:
            self.q = float(self.delta @ self.delta / self.b)

    def quadratic(self, w: np.ndarray) -> float:
        zw = self.z @ w
        return float(self.a * (zw @ zw) / self.n + self.b * (w @ w))

    def fisher_efficiency(self, w: np.ndarray) -> float:
        return float(w @ self.delta / np.sqrt(self.quadratic(w) * self.q))


# --------------------------------------------------------------- data loading
def load_groups(prompt_root: Path, cache_root: Path, language: str) -> dict:
    prompt_path = prompt_root / f"{language}.activation.json"
    rows = json.loads(prompt_path.read_text(encoding="utf-8"))
    cache, metadata = load_complete_cache(cache_root / language, prompt_path)
    assert_row_alignment(rows, metadata)
    cache = np.asarray(cache, dtype=np.float32)  # rows x blocks x hidden, in RAM
    out = {}
    groups = group_index(rows)
    for partition in ("train", "validation", "test"):
        ids, labels, members = [], [], []
        for gid, ix in groups.items():
            if rows[int(ix[0])]["partition"] != partition:
                continue
            group_labels = {int(rows[int(i)]["label"]) for i in ix}
            if len(group_labels) != 1:
                continue
            ids.append(gid); labels.append(group_labels.pop()); members.append(ix)
        x = np.stack([cache[ix].mean(axis=0) for ix in members])  # groups x blocks x hidden
        out[partition] = {"ids": ids, "y": np.asarray(labels, dtype=int), "x": x}
    return out


def fit_direction(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x[y == 1].mean(axis=0) - x[y == 0].mean(axis=0)


# ----------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--layer-selection", required=True, type=Path)
    parser.add_argument("--allocation-root", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--with-x1-methods", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"Refusing existing output: {args.output}")
    selection = json.loads(args.layer_selection.read_text(encoding="utf-8"))
    frozen_index = int(selection["selected_cache_index"])
    data = {lang: load_groups(args.prompt_root, args.cache_root, lang) for lang in LANGUAGES}
    n_blocks = data["en"]["train"]["x"].shape[1]

    # Validation groups pure in all six languages; parallel by construction.
    common_val = sorted(set.intersection(*(set(data[l]["validation"]["ids"]) for l in LANGUAGES)))
    val_index = {l: np.asarray([data[l]["validation"]["ids"].index(g) for g in common_val]) for l in LANGUAGES}
    test_y = {l: data[l]["test"]["y"] for l in LANGUAGES}

    configs, test_scores, val_pairs = [], {t: [] for t in LANGUAGES}, []
    for block in range(n_blocks):
        directions, geometry = {}, {}
        for lang in LANGUAGES:
            tr = data[lang]["train"]
            directions[lang] = fit_direction(tr["x"][:, block].astype(np.float64), tr["y"])
            geometry[lang] = TargetGeometry(tr["x"][:, block], tr["y"])
        for source, target in PAIRS:
            w = directions[source]
            scores = data[target]["test"]["x"][:, block].astype(np.float64) @ w
            a = data[source]["validation"]["x"][val_index[source], block].astype(np.float64) @ w
            b = data[target]["validation"]["x"][val_index[target], block].astype(np.float64) @ w
            native = data[target]["test"]["x"][:, block].astype(np.float64) @ directions[target]
            configs.append({
                "block_number": block + 1, "source": source, "target": target,
                "frozen_block": block == frozen_index,
                "auroc": float(auroc_matrix(scores[None], test_y[target])[0]),
                "native_auroc": float(auroc_matrix(native[None], test_y[target])[0]),
                "p1_euclidean_cosine": cosine(w, directions[target]),
                "p2_translation_consistency": float(pearson_rows(a[None], b[None])[0]),
                "p2_spearman": spearman(a, b),
                "p3_fisher_efficiency": geometry[target].fisher_efficiency(w),
                "target_shrinkage": geometry[target].rho,
            })
            test_scores[target].append((len(configs) - 1, scores))
            val_pairs.append((a, b))

    def summarize(rows: list[dict]) -> dict:
        y = np.asarray([r["auroc"] for r in rows])
        return {name: spearman([r[name] for r in rows], y) for name in
                ("p1_euclidean_cosine", "p2_translation_consistency", "p2_spearman", "p3_fisher_efficiency")}

    observed = summarize(configs)
    frozen = summarize([c for c in configs if c["frozen_block"]])

    # Paired bootstrap: resample validation groups and label-stratified test groups.
    rng = np.random.default_rng(int(hashlib.sha256(f"{args.seed}|{args.model_id}".encode()).hexdigest()[:8], 16))
    p1 = np.asarray([c["p1_euclidean_cosine"] for c in configs])
    p3 = np.asarray([c["p3_fisher_efficiency"] for c in configs])
    va = np.stack([pair[0] for pair in val_pairs]); vb = np.stack([pair[1] for pair in val_pairs])
    target_blocks = {t: (np.asarray([i for i, _ in test_scores[t]]), np.stack([s for _, s in test_scores[t]]))
                     for t in LANGUAGES}
    deltas, rho_p1, rho_p2 = [], [], []
    for _ in range(args.bootstrap_draws):
        vix = rng.integers(0, len(common_val), size=len(common_val))
        p2 = pearson_rows(va[:, vix], vb[:, vix])
        auc = np.empty(len(configs))
        for t, (index, matrix) in target_blocks.items():
            y = test_y[t]
            pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
            tix = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            auc[index] = auroc_matrix(matrix[:, tix], y[tix])
        r1, r2 = spearman(p1, auc), spearman(p2, auc)
        rho_p1.append(r1); rho_p2.append(r2); deltas.append(r2 - r1)
    deltas = np.asarray(deltas)
    lower = (np.count_nonzero(deltas <= 0) + 1) / (len(deltas) + 1)
    upper = (np.count_nonzero(deltas >= 0) + 1) / (len(deltas) + 1)

    result = {
        "schema_version": 1,
        "analysis": "post-core X7 label-free transfer prediction (protocol/X7_X9_PREREGISTRATION.md)",
        "model_id": args.model_id, "n_blocks": n_blocks, "frozen_block_number": frozen_index + 1,
        "n_configurations": len(configs), "n_common_validation_groups": len(common_val),
        "spearman_with_auroc_all_blocks": observed,
        "spearman_with_auroc_frozen_block": frozen,
        "h1_delta_rho_p2_minus_p1": {
            "observed": observed["p2_translation_consistency"] - observed["p1_euclidean_cosine"],
            "ci95": [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))],
            "p_two_sided": float(min(1.0, 2 * min(lower, upper))),
            "bootstrap_rho_p1_ci95": [float(np.quantile(rho_p1, .025)), float(np.quantile(rho_p1, .975))],
            "bootstrap_rho_p2_ci95": [float(np.quantile(rho_p2, .025)), float(np.quantile(rho_p2, .975))],
            "draws": args.bootstrap_draws,
        },
        "configurations": configs,
    }

    if args.allocation_root is not None:
        result["overlap_secondary"] = overlap_secondary(data, frozen_index, args.allocation_root, val_index)
    if args.with_x1_methods:
        result["x1_methods_secondary"] = x1_secondary(data, frozen_index, val_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "configs": len(configs),
                      "rho_all_blocks": observed, "rho_frozen": frozen,
                      "delta": result["h1_delta_rho_p2_minus_p1"]}))


def overlap_secondary(data: dict, block: int, allocation_root: Path, val_index: dict) -> dict:
    """RQ-A allocations: overlap slope of P1 (probe-probe cosine) versus P2."""
    lookup = {l: {gid: i for i, gid in enumerate(data[l]["train"]["ids"])} for l in LANGUAGES}
    rows = []
    for path in sorted(p for p in allocation_root.glob("*__*.json") if not p.name.startswith(".")):
        allocation = json.loads(path.read_text(encoding="utf-8"))
        left, right = allocation["language_pair"]
        for plan in allocation["plans"]:
            def fit(lang, ids):
                ix = np.asarray([lookup[lang][g] for g in ids])
                return fit_direction(data[lang]["train"]["x"][ix, block].astype(np.float64), data[lang]["train"]["y"][ix])
            wl, wr = fit(left, plan["source_group_ids"]), fit(right, plan["target_group_ids"])
            for source, target, w in ((left, right, wl), (right, left, wr)):
                a = data[source]["validation"]["x"][val_index[source], block].astype(np.float64) @ w
                b = data[target]["validation"]["x"][val_index[target], block].astype(np.float64) @ w
                scores = data[target]["test"]["x"][:, block].astype(np.float64) @ w
                rows.append({"overlap": float(plan["requested_overlap_fraction"]), "source": source, "target": target,
                             "p1": cosine(wl, wr), "p2": float(pearson_rows(a[None], b[None])[0]),
                             "auroc": float(auroc_matrix(scores[None], data[target]["test"]["y"])[0])})
    levels = sorted({r["overlap"] for r in rows})
    means = {name: [float(np.mean([r[name] for r in rows if r["overlap"] == o])) for o in levels]
             for name in ("p1", "p2", "auroc")}
    return {"overlap_levels": levels, "means_by_level": means,
            "slopes_per_full_overlap": {name: float(np.polyfit(levels, values, 1)[0]) for name, values in means.items()},
            "n_rows": len(rows)}


def x1_secondary(data: dict, block: int, val_index: dict) -> dict:
    """P1/P2/AUROC for X1 transforms using each method's effective target score vector."""
    from .analyze_alignment_baselines import procrustes_score_vectors, ridge_score_vector, top_difference_pc, derangement
    import os
    device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() else "cpu"
    rng = np.random.default_rng(20261003)
    tr = {l: data[l]["train"] for l in LANGUAGES}
    directions = {l: fit_direction(tr[l]["x"][:, block].astype(np.float64), tr[l]["y"]) for l in LANGUAGES}
    rows = []
    for li, left in enumerate(LANGUAGES):
        for right in LANGUAGES[li + 1:]:
            ids = sorted(set(tr[left]["ids"]) & set(tr[right]["ids"]))
            xl = np.asarray([tr[left]["x"][tr[left]["ids"].index(g), block] for g in ids], dtype=np.float32)
            xr = np.asarray([tr[right]["x"][tr[right]["ids"].index(g), block] for g in ids], dtype=np.float32)
            al, ar = xl - xl.mean(0), xr - xr.mean(0)
            perm = derangement(len(ids), rng)
            (r_w_right, rt_w_left), _ = procrustes_score_vectors(al, ar, [directions[right], directions[left]], device)
            (s_w_right, st_w_left), _ = procrustes_score_vectors(al[perm], ar, [directions[right], directions[left]], device)
            pc = top_difference_pc(al, ar, device)
            for source, target, rosh, scr, centered_t, centered_s in (
                    (left, right, rt_w_left, st_w_left, ar, al), (right, left, r_w_right, s_w_right, al, ar)):
                w = directions[source]
                methods = {"raw": w, "rosh_fixed_layer": rosh, "rosh_scrambled": scr,
                           "ridge": ridge_score_vector(centered_t, centered_s, w, 0.01, device),
                           "pca1": w - pc * float(pc @ w)}
                a = data[source]["validation"]["x"][val_index[source], block].astype(np.float64) @ w
                for name, v in methods.items():
                    v = np.asarray(v, dtype=np.float64)
                    b = data[target]["validation"]["x"][val_index[target], block].astype(np.float64) @ v
                    scores = data[target]["test"]["x"][:, block].astype(np.float64) @ v
                    rows.append({"method": name, "source": source, "target": target,
                                 "p1": cosine(v, directions[target]), "p2": float(pearson_rows(a[None], b[None])[0]),
                                 "auroc": float(auroc_matrix(scores[None], data[target]["test"]["y"])[0])})
    summary = {}
    for name in sorted({r["method"] for r in rows}):
        sub = [r for r in rows if r["method"] == name]
        summary[name] = {k: float(np.mean([r[k] for r in sub])) for k in ("p1", "p2", "auroc")}
    return {"method_means": summary, "rows": rows,
            "spearman_over_methods_and_pairs": {"p1": spearman([r["p1"] for r in rows], [r["auroc"] for r in rows]),
                                                "p2": spearman([r["p2"] for r in rows], [r["auroc"] for r in rows])}}


if __name__ == "__main__":
    main()
