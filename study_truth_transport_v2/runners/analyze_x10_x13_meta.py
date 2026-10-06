#!/usr/bin/env python3
"""Cross-model summaries and frozen decision rules for X10-X13 (protocol/X10_X11_PREREGISTRATION.md)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def sign_test(successes: int, n: int) -> float:
    tail = sum(math.comb(n, k) for k in range(0, min(successes, n - successes) + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    x = {m: json.loads((args.results_root / m / "extensions/x10_x12.json").read_text()) for m in args.models}
    p = {m: json.loads((args.results_root / m / "extensions/x13_probes.json").read_text()) for m in args.models}

    # X10
    units = [(m, pair) for m in args.models for pair in x[m]["x10"]["pairs"]]
    pred = np.asarray([u[1]["predicted_slope"] for u in units]); obs = np.asarray([u[1]["observed_slope"] for u in units])
    r_slope = float(np.corrcoef(pred, obs)[0, 1])
    ratio = obs / pred
    through_origin = float((pred @ obs) / (pred @ pred))
    lev = {}
    for i, f in enumerate(x[args.models[0]]["x10"]["levels"]):
        pl = np.asarray([u[1]["predicted_cosine"][i] for u in units]); ol = np.asarray([u[1]["observed_cosine"][i] for u in units])
        lev[str(f)] = {"r": float(np.corrcoef(pl, ol)[0, 1]), "mean_abs_error": float(np.mean(np.abs(pl - ol)))}
    median_ratio = float(np.median(ratio))
    verdict = ("supported" if r_slope >= 0.7 and 0.8 <= median_ratio <= 1.25 else
               "partially supported" if r_slope >= 0.5 else "not supported")
    per_model_r = {m: float(np.corrcoef([q["predicted_slope"] for q in x[m]["x10"]["pairs"]],
                                        [q["observed_slope"] for q in x[m]["x10"]["pairs"]])[0, 1]) for m in args.models}
    x10 = {"n_units": len(units), "slope_r": r_slope, "median_ratio_observed_over_predicted": median_ratio,
           "ratio_iqr": [float(np.quantile(ratio, .25)), float(np.quantile(ratio, .75))],
           "through_origin_slope": through_origin, "levels": lev, "per_model_slope_r": per_model_r, "verdict": verdict}

    # X11
    x11_rows = {}
    for m in args.models:
        u, f = x[m]["x11"]["unfiltered"], x[m]["x11"]["filtered"]
        x11_rows[m] = {"delta_cross_auroc": f["mean_cross_auroc"] - u["mean_cross_auroc"],
                       "delta_balanced_accuracy": f["mean_balanced_accuracy"] - u["mean_balanced_accuracy"],
                       "best_unfiltered": u["best"], "best_filtered": f["best"],
                       "spearman_unfiltered": u["spearman"], "spearman_filtered": f["spearman"]}
    robust = all(abs(v["delta_cross_auroc"]) < 0.02 and v["best_unfiltered"] == v["best_filtered"] for v in x11_rows.values())
    x11 = {"per_model": x11_rows, "max_abs_delta_cross_auroc": float(max(abs(v["delta_cross_auroc"]) for v in x11_rows.values())),
           "n_best_unchanged": sum(v["best_unfiltered"] == v["best_filtered"] for v in x11_rows.values()),
           "verdict": "robust" if robust else "not robust"}

    # X12
    ks = x[args.models[0]]["x12"]["ks"]
    curves = {str(k): {name: float(np.mean([x[m]["x12"]["by_k"][str(k)][name]["mean_rho"] for m in args.models]))
                       for name in ("direct", "fisher", "cos")} for k in ks}
    diffs16 = {m: x[m]["x12"]["by_k"]["16"]["fisher_minus_direct"] for m in args.models}
    positive = sum(v["mean"] > 0 for v in diffs16.values())
    x12 = {"curves_mean_rho_over_models": curves, "k16_fisher_minus_direct": diffs16, "k16_models_positive": positive,
           "k16_sign_test_p": sign_test(positive, len(args.models)),
           "verdict": "supported" if positive >= 8 else "not supported"}

    # X13
    fam = {}
    for f in ("mm", "lda", "lr"):
        ok = [m for m in args.models if p[m]["families"][f]["overlap_slopes"]["cosine"]["ci95"][0] > 0
              and abs(p[m]["families"][f]["overlap_slopes"]["auroc"]["mean"]) < 0.01]
        fam[f] = {"n_models_dissociation": len(ok), "replicates": len(ok) >= 8,
                  "mean_cross_auroc": float(np.mean([p[m]["families"][f]["mean_cross_auroc"] for m in args.models])),
                  "mean_native_auroc": float(np.mean([p[m]["families"][f]["mean_native_auroc"] for m in args.models])),
                  "mean_balanced_accuracy": float(np.mean([p[m]["families"][f]["mean_balanced_accuracy"] for m in args.models])),
                  "median_spearman_cosine": float(np.median([p[m]["families"][f]["spearman_cosine"] for m in args.models])),
                  "median_spearman_fisher": float(np.median([p[m]["families"][f]["spearman_fisher"] for m in args.models])),
                  "mean_cosine_slope": float(np.mean([p[m]["families"][f]["overlap_slopes"]["cosine"]["mean"] for m in args.models])),
                  "max_abs_auroc_slope": float(max(abs(p[m]["families"][f]["overlap_slopes"]["auroc"]["mean"]) for m in args.models)),
                  "per_model": {m: {k: p[m]["families"][f][k] for k in ("mean_cross_auroc", "mean_balanced_accuracy", "mean_cosine",
                                                                       "spearman_cosine", "spearman_fisher", "overlap_slopes")}
                                for m in args.models}}
    x13 = {"families": fam, "lr_c": {m: p[m]["lr_c"] for m in args.models}}
    result = {"schema_version": 1, "protocol": "protocol/X10_X11_PREREGISTRATION.md", "models": args.models,
              "x10": x10, "x11": x11, "x12": x12, "x13": x13}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"x10": {k: x10[k] for k in ("slope_r", "median_ratio_observed_over_predicted", "verdict")},
                      "x11": {k: x11[k] for k in ("max_abs_delta_cross_auroc", "n_best_unchanged", "verdict")},
                      "x12": {k: x12[k] for k in ("k16_models_positive", "verdict")},
                      "x13": {f: {k: v[k] for k in ("n_models_dissociation", "replicates", "mean_cross_auroc")} for f, v in fam.items()}}))


if __name__ == "__main__":
    main()
