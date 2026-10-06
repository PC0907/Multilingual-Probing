#!/usr/bin/env python3
"""Translation-system robustness: key per-model statistics under v2 (NLLB) and v3 (LLM) claim translations.

Reads two results roots with the same layout and writes a JSON summary plus a LaTeX table body.
Only models present (and complete) in both roots are compared.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from study_truth_transport_v2.reporting.compose_report import rq_a_stats


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(root: Path, model: str) -> dict | None:
    res = root / model
    needed = [res / "rq_c.json", res / "rq_a/inference.json", res / "extensions/transfer_predictors.json"]
    if not all(p.exists() for p in needed):
        return None
    cells = [c for c in read(res / "rq_c.json")["cells"] if c["source"] != c["target"]]
    a = rq_a_stats(root, model)
    tp = read(res / "extensions/transfer_predictors.json")
    out = {
        "block": int(read(res / "layer_selection.json")["selected_block_number"]),
        "cross_auroc": float(np.mean([c["auroc"] for c in cells])),
        "cosine": float(np.mean([c["direction_cosine"] for c in cells])),
        "cos_slope": a["cos_slope"]["mean"], "cos_slope_lo": a["cos_slope"]["p025"],
        "auc_slope": a["auc_slope"]["mean"], "auc_slope_lo": a["auc_slope"]["p025"], "auc_slope_hi": a["auc_slope"]["p975"],
        "rho_cos": tp["spearman_with_auroc_all_blocks"]["p1_euclidean_cosine"],
        "rho_fisher": tp["spearman_with_auroc_all_blocks"]["p3_fisher_efficiency"],
    }
    for name, key in (("mmmlu", "mmmlu_transfer.json"), ("include", "include_transfer.json")):
        p = res / "extensions" / key
        out[name] = read(p)["off_diagonal_means"]["paired_candidate_accuracy"] if p.exists() else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2-root", required=True, type=Path)
    ap.add_argument("--v3-root", required=True, type=Path)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--table", type=Path)
    args = ap.parse_args()
    rows = {}
    for m in args.models:
        a, b = stats(args.v2_root, m), stats(args.v3_root, m)
        if a and b:
            rows[m] = {"v2": a, "v3": b}
    keys = ("cross_auroc", "cosine", "cos_slope", "auc_slope", "rho_cos", "rho_fisher", "mmmlu", "include")
    agg = {}
    for k in keys:
        pairs = [(r["v2"][k], r["v3"][k]) for r in rows.values() if r["v2"][k] is not None and r["v3"][k] is not None]
        if pairs:
            d = np.array([b - a for a, b in pairs])
            agg[k] = {"n": len(pairs), "mean_v2": float(np.mean([a for a, _ in pairs])), "mean_v3": float(np.mean([b for _, b in pairs])),
                      "mean_diff": float(d.mean()), "max_abs_diff": float(np.abs(d).max())}
    dissoc = {v: sum(1 for r in rows.values() if r[v]["cos_slope_lo"] > 0 and r[v]["auc_slope_lo"] <= 0 <= r[v]["auc_slope_hi"]) for v in ("v2", "v3")}
    fisher_wins = {v: sum(1 for r in rows.values() if r[v]["rho_fisher"] > r[v]["rho_cos"]) for v in ("v2", "v3")}
    result = {"models": list(rows), "per_model": rows, "aggregate": agg,
              "n_models_overlap_dissociation": dissoc, "n_models_fisher_beats_cosine": fisher_wins,
              "same_block": sum(r["v2"]["block"] == r["v3"]["block"] for r in rows.values())}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.table:
        lines = []
        for m, r in rows.items():
            f = lambda x, d=3: "--" if x is None else f"{x:.{d}f}"
            lines.append(" & ".join([m, f"{r['v2']['block']}/{r['v3']['block']}",
                                     f(r["v2"]["cross_auroc"]), f(r["v3"]["cross_auroc"]),
                                     f"{r['v2']['cos_slope']:+.3f}", f"{r['v3']['cos_slope']:+.3f}",
                                     f"{r['v2']['auc_slope']:+.4f}", f"{r['v3']['auc_slope']:+.4f}",
                                     f(r["v2"]["rho_fisher"], 2), f(r["v3"]["rho_fisher"], 2)]) + r" \\")
        args.table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"models": len(rows), "dissociation": dissoc, "fisher_beats_cosine": fisher_wins, "same_block": result["same_block"]}))


if __name__ == "__main__":
    main()
