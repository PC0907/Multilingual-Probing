#!/usr/bin/env python3
"""Cross-model X14 summary and frozen decision rules (protocol/X14_PREREGISTRATION.md)."""
import argparse
import json
from pathlib import Path

import numpy as np

LINEAR = ("p1_euclidean_cosine", "p3_fisher_efficiency", "p4a_mcs_total_vs_target_probe", "p4b_mcs_total_vs_target_optimal")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", required=True, type=Path)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    r = {m: json.loads((args.results_root / m / "extensions/x14_mcs.json").read_text()) for m in args.models}
    rho = {m: r[m]["spearman_all_blocks"] for m in args.models}
    n = len(args.models)
    fisher_beats_cos = sum(rho[m]["p3_fisher_efficiency"] > rho[m]["p1_euclidean_cosine"] for m in args.models)
    mcs_ge_fisher = sum(rho[m]["p4b_mcs_total_vs_target_optimal"] >= rho[m]["p3_fisher_efficiency"] for m in args.models)

    # H14b: leave-one-model-out linear calibration.
    lomo = {}
    for stat in LINEAR:
        sse = sst = 0.0
        y_all = np.concatenate([[c["auroc"] for c in r[m]["configurations"]] for m in args.models])
        for held in args.models:
            train = [c for m in args.models if m != held for c in r[m]["configurations"]]
            slope, icpt = np.polyfit([c[stat] for c in train], [c["auroc"] for c in train], 1)
            test = r[held]["configurations"]
            y = np.array([c["auroc"] for c in test]); pred = slope * np.array([c[stat] for c in test]) + icpt
            sse += float(((y - pred) ** 2).sum()); sst += float(((y - y_all.mean()) ** 2).sum())
        lomo[stat] = 1 - sse / sst
    pooled = [c for m in args.models for c in r[m]["configurations"]]
    slope_p4b = float(np.polyfit([c["p4b_mcs_total_vs_target_optimal"] for c in pooled], [c["auroc"] for c in pooled], 1)[0])

    # H14c: parameter-free Proposition-1 prediction.
    y = np.array([c["auroc"] for c in pooled]); yh = np.array([c["p5_prop1_predicted_auroc"] for c in pooled])
    r2_prop1 = float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum()); mae_prop1 = float(np.abs(y - yh).mean())
    per_model_prop1 = {m: {"mae": float(np.mean([abs(c["auroc"] - c["p5_prop1_predicted_auroc"]) for c in r[m]["configurations"]])),
                           "mean_bias": float(np.mean([c["p5_prop1_predicted_auroc"] - c["auroc"] for c in r[m]["configurations"]]))}
                       for m in args.models}

    # H14d: outlier dimensions.
    oa = {m: {k: r[m]["outlier_analysis"][k] for k in ("n_outlier_dims", "outlier_variance_share", "x10_slope_r_raw", "x10_slope_r_reduced")}
          | {"rho_cos_raw": r[m]["outlier_analysis"]["spearman_frozen_raw"]["p1_euclidean_cosine"],
             "rho_cos_reduced": r[m]["outlier_analysis"]["spearman_frozen_reduced"]["p1_euclidean_cosine"],
             "rho_fisher_raw": r[m]["outlier_analysis"]["spearman_frozen_raw"]["p3_fisher_efficiency"],
             "rho_fisher_reduced": r[m]["outlier_analysis"]["spearman_frozen_reduced"]["p3_fisher_efficiency"]} for m in args.models}
    ap_ = oa.get("apertus-8b-2509")
    h14d = None
    if ap_:
        h14d = bool((ap_["x10_slope_r_reduced"] or -1) > 0.5 and ap_["rho_cos_reduced"] > ap_["rho_cos_raw"])

    out = {"schema_version": 1, "protocol": "protocol/X14_PREREGISTRATION.md", "models": args.models,
           "within_model_spearman": rho,
           "h14a": {"fisher_beats_cosine": fisher_beats_cos, "mcs_optimal_ge_fisher": mcs_ge_fisher, "n_models": n,
                    "keep_fisher_over_cosine_claim": fisher_beats_cos >= 0.8 * n, "recommend_mcs": mcs_ge_fisher >= 0.8 * n},
           "h14b": {"lomo_r2": lomo, "calibrated": {k: v >= 0.5 for k, v in lomo.items()}, "pooled_slope_p4b": slope_p4b},
           "h14c": {"r2_no_fit": r2_prop1, "mae": mae_prop1, "calibrated": r2_prop1 >= 0.5 and mae_prop1 <= 0.05, "per_model": per_model_prop1},
           "h14d": {"per_model": oa, "apertus_supported": h14d}}
    out["any_calibrated"] = bool(any(out["h14b"]["calibrated"][k] for k in LINEAR[2:]) or out["h14c"]["calibrated"])
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"h14a": out["h14a"], "lomo_r2": lomo, "h14c": {k: out["h14c"][k] for k in ("r2_no_fit", "mae", "calibrated")},
                      "h14d_apertus": h14d}, indent=1))


if __name__ == "__main__":
    main()
