#!/usr/bin/env python3
"""X7 across models: H1 (per-model Δρ, Holm, sign test) and H2 (leave-one-model-out R²)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

PREDICTORS = ("p1_euclidean_cosine", "p2_translation_consistency", "p3_fisher_efficiency")


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted, running = {}, 0.0
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * p_values[name]))
        adjusted[name] = running
    return adjusted


def sign_test(successes: int, n: int) -> float:
    tail = sum(math.comb(n, k) for k in range(0, min(successes, n - successes) + 1)) / 2**n
    return float(min(1.0, 2 * tail))


def lomo_r2(models: dict[str, list[dict]], predictor: str) -> dict[str, float]:
    out = {}
    for held in models:
        train = [c for m, rows in models.items() if m != held for c in rows]
        x = np.asarray([c[predictor] for c in train]); y = np.asarray([c["auroc"] for c in train])
        slope, intercept = np.polyfit(x, y, 1)
        xt = np.asarray([c[predictor] for c in models[held]]); yt = np.asarray([c["auroc"] for c in models[held]])
        residual = yt - (slope * xt + intercept)
        out[held] = float(1 - (residual @ residual) / ((yt - yt.mean()) @ (yt - yt.mean())))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--primary-models", nargs="+", required=True)
    parser.add_argument("--secondary-models", nargs="*", default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    load = lambda m: json.loads((args.results_root / m / "extensions/transfer_predictors.json").read_text())
    primary = {m: load(m) for m in args.primary_models}
    per_model, raw_p = {}, {}
    for m, data in primary.items():
        h1 = data["h1_delta_rho_p2_minus_p1"]
        per_model[m] = {"n_configurations": data["n_configurations"], "n_blocks": data["n_blocks"],
                        "rho_all_blocks": data["spearman_with_auroc_all_blocks"],
                        "rho_frozen_block": data["spearman_with_auroc_frozen_block"],
                        "delta_rho": h1["observed"], "delta_ci95": h1["ci95"], "p_two_sided": h1["p_two_sided"]}
        raw_p[m] = h1["p_two_sided"]
        if "overlap_secondary" in data:
            per_model[m]["overlap_slopes"] = data["overlap_secondary"]["slopes_per_full_overlap"]
        if "x1_methods_secondary" in data:
            per_model[m]["x1_methods"] = data["x1_methods_secondary"]["method_means"]
            per_model[m]["x1_rho"] = data["x1_methods_secondary"]["spearman_over_methods_and_pairs"]
    adjusted = holm(raw_p)
    for m in per_model:
        per_model[m]["p_holm"] = adjusted[m]
    positive_ci = sum(v["delta_ci95"][0] > 0 for v in per_model.values())
    negative_ci = sum(v["delta_ci95"][1] < 0 for v in per_model.values())
    positive_sign = sum(v["delta_rho"] > 0 for v in per_model.values())
    configs = {m: data["configurations"] for m, data in primary.items()}
    lomo = {p: lomo_r2(configs, p) for p in PREDICTORS}
    secondary = {}
    for m in args.secondary_models:
        data = load(m)
        secondary[m] = {"rho_all_blocks": data["spearman_with_auroc_all_blocks"],
                        "delta_rho": data["h1_delta_rho_p2_minus_p1"]["observed"],
                        "delta_ci95": data["h1_delta_rho_p2_minus_p1"]["ci95"]}
    result = {
        "schema_version": 1, "analysis": "X7 meta-analysis (protocol/X7_X9_PREREGISTRATION.md)",
        "primary_models": args.primary_models,
        "h1": {"models_with_delta_ci_above_zero": positive_ci, "models_with_delta_ci_below_zero": negative_ci,
               "models_with_positive_delta": positive_sign, "n_models": len(per_model),
               "sign_test_p": sign_test(positive_sign, len(per_model)),
               "mean_delta_rho": float(np.mean([v["delta_rho"] for v in per_model.values()])),
               "supported": positive_ci > len(per_model) / 2},
        "h2_leave_one_model_out_r2": {p: {"per_model": v, "mean": float(np.mean(list(v.values())))} for p, v in lomo.items()},
        "per_model": per_model,
        "secondary_training_checkpoints": secondary,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"h1": result["h1"], "h2_mean": {p: v["mean"] for p, v in result["h2_leave_one_model_out_r2"].items()}}))


if __name__ == "__main__":
    main()
