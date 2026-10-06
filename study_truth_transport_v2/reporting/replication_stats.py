"""Shared computations for the v3-primary write-up: X14, the v2/v3 translation robustness check, the v5
replication (full and shortcut-free) and translation-quality checks. Every value is read from results JSON."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from .compose_report import rq_a_stats, rq_c_stats

RESULTS = Path(__file__).resolve().parents[1] / "results"
V2, V3, V5 = RESULTS / "remote", RESULTS / "remote_v3", RESULTS / "remote_v5"
MODELS = ["qwen3-0.6b-base", "qwen3-1.7b-base", "qwen3-4b-base", "qwen3-8b-base", "qwen3-14b-base", "qwen3-8b",
          "olmo-2-1124-7b", "gemma-7b", "apertus-8b-2509", "mistral-7b-v0.3"]
NAME = {"qwen3-0.6b-base": "Qwen3-0.6B", "qwen3-1.7b-base": "Qwen3-1.7B", "qwen3-4b-base": "Qwen3-4B",
        "qwen3-8b-base": "Qwen3-8B-Base", "qwen3-14b-base": "Qwen3-14B", "qwen3-8b": "Qwen3-8B (post-trained)",
        "olmo-2-1124-7b": "OLMo-2-7B", "gemma-7b": "Gemma-7B", "apertus-8b-2509": "Apertus-8B",
        "mistral-7b-v0.3": "Mistral-7B-v0.3"}
X14_KEYS = {"p1_euclidean_cosine": "Cosine", "p3_fisher_efficiency": "Fisher efficiency",
            "p4a_mcs_total_vs_target_probe": "Mahalanobis cosine vs target probe",
            "p4b_mcs_total_vs_target_optimal": "Mahalanobis cosine vs target optimum",
            "p5_prop1_predicted_auroc": "Prop. 1 (no fit)"}


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _variant(root: Path, model: str, shortcut_free: bool = False) -> dict:
    base, sub = (root / model, "shortcut_control") if shortcut_free else (root, model)
    a, c = rq_a_stats(base, sub), rq_c_stats(base, sub)
    tp = read((root / model / "shortcut_control" / "transfer_predictors.json") if shortcut_free
              else (root / model / "extensions" / "transfer_predictors.json"))["spearman_with_auroc_all_blocks"]
    cos, auc = a["cos_slope"], a["auc_slope"]
    return {"cross_auroc": c["auc"], "cosine": c["cos"],
            "cos_slope": cos["mean"], "cos_lo": cos["p025"], "cos_hi": cos["p975"],
            "auc_slope": auc["mean"], "auc_lo": auc["p025"], "auc_hi": auc["p975"],
            "dissociation": bool(cos["p025"] > 0 and auc["p025"] <= 0 <= auc["p975"]),
            "auc_flat": bool(auc["p025"] <= 0 <= auc["p975"]), "cos_up": bool(cos["p025"] > 0),
            "rho_cos": tp["p1_euclidean_cosine"], "rho_fisher": tp["p3_fisher_efficiency"],
            "fisher_beats_cos": bool(tp["p3_fisher_efficiency"] > tp["p1_euclidean_cosine"])}


def replication() -> dict:
    rows = {m: {"v2": _variant(V2, m), "v3": _variant(V3, m), "v5": _variant(V5, m), "v5_sf": _variant(V5, m, True)}
            for m in MODELS}
    agg = {}
    for v in ("v2", "v3", "v5", "v5_sf"):
        r = [rows[m][v] for m in MODELS]
        agg[v] = {"n": len(r), "dissociation": sum(x["dissociation"] for x in r), "auc_flat": sum(x["auc_flat"] for x in r),
                  "cos_up": sum(x["cos_up"] for x in r), "fisher_beats_cos": sum(x["fisher_beats_cos"] for x in r),
                  "mean_cross_auroc": float(np.mean([x["cross_auroc"] for x in r])),
                  "mean_cos_slope": float(np.mean([x["cos_slope"] for x in r])),
                  "median_rho_cos": float(np.median([x["rho_cos"] for x in r])),
                  "median_rho_fisher": float(np.median([x["rho_fisher"] for x in r]))}
    shortcut_gain = [rows[m]["v5"]["cross_auroc"] - rows[m]["v5_sf"]["cross_auroc"] for m in MODELS]
    v3_sf_gap = [rows[m]["v5_sf"]["cross_auroc"] - rows[m]["v3"]["cross_auroc"] for m in MODELS]
    slope_ratio = [rows[m]["v3"]["cos_slope"] / rows[m]["v5_sf"]["cos_slope"] for m in MODELS if rows[m]["v5_sf"]["cos_slope"] > 0]
    same_dissoc = [m for m in MODELS if rows[m]["v5"]["dissociation"]]
    return {"rows": rows, "agg": agg, "shortcut_gain": {"min": min(shortcut_gain), "max": max(shortcut_gain),
                                                       "mean": float(np.mean(shortcut_gain))},
            "v5_sf_minus_v3": {"min": min(v3_sf_gap), "max": max(v3_sf_gap), "mean": float(np.mean(v3_sf_gap))},
            "v3_over_v5sf_cos_slope": {"min": min(slope_ratio), "max": max(slope_ratio), "median": float(np.median(slope_ratio))},
            "v5_dissociation_models": same_dissoc,
            "v5_sf_dissociation_models": [m for m in MODELS if rows[m]["v5_sf"]["dissociation"]]}


def x14(root: Path) -> dict:
    meta = read(root / "x14_meta.json")
    return {"within": meta["within_model_spearman"], "h14a": meta["h14a"], "lomo_r2": meta["h14b"]["lomo_r2"],
            "calibrated": meta["h14b"]["calibrated"], "prop1": {k: meta["h14c"][k] for k in ("r2_no_fit", "mae", "calibrated")},
            "prop1_bias": float(np.mean([v["mean_bias"] for v in meta["h14c"]["per_model"].values()])),
            "best_count": Counter(max(v, key=v.get) for v in meta["within_model_spearman"].values())}


def mistral_v5_behavior() -> dict:
    data = read(V5 / "mistral-7b-v0.3" / "behavior_aggregate.json")["languages"]
    out = {}
    for lang, v in data.items():
        preds = [p for f in v["facts"] for p in f["template_predictions"]]
        out[lang] = {"share_true": float(np.mean(preds)),
                     "excluded": [k for k, p in v["standardization"].items() if "excluded" in p]}
    return out


def translation() -> dict:
    comp = read(V3 / "translation_comparison.json")
    qe = read(RESULTS / "translation_qe_three_way.json")["languages"]
    fid = read(RESULTS / "translation_fidelity.json")
    return {"comparison": comp, "labse": {l: {s: qe[l][s]["mean"] for s in qe[l]} for l in qe},
            "labse_share_llm_higher": {l: qe[l]["LLM"]["share_higher_than_ref"] for l in qe},
            "fidelity": {l: {s: (v["digit_mismatch"], v["imperial_to_metric"]) for s, v in r.items()} for l, r in fid["languages"].items()},
            "n_imperial": fid["n_imperial_english_claims"]}
