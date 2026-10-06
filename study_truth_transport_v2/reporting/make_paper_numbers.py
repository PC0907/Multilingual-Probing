#!/usr/bin/env python3
"""Write LaTeX macros and tables for the ACL-format paper from the JSON artifacts.

Every number in the paper body comes from this script's output (paper/generated/).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from study_truth_transport_v2.reporting import replication_stats as rs
from study_truth_transport_v2.reporting.compose_report import rq_a_stats, rq_b_stats, rq_c_stats, rq_d_stats

MODELS = ["qwen3-0.6b-base", "qwen3-1.7b-base", "qwen3-4b-base", "qwen3-8b-base", "qwen3-14b-base",
          "qwen3-8b", "olmo-2-1124-7b", "gemma-7b", "apertus-8b-2509", "mistral-7b-v0.3"]
CORE = ["gemma-7b", "qwen3-8b-base", "apertus-8b-2509", "mistral-7b-v0.3"]
NAME = {"qwen3-0.6b-base": "Qwen3-0.6B", "qwen3-1.7b-base": "Qwen3-1.7B", "qwen3-4b-base": "Qwen3-4B",
        "qwen3-8b-base": "Qwen3-8B", "qwen3-14b-base": "Qwen3-14B", "qwen3-8b": "Qwen3-8B (post)",
        "olmo-2-1124-7b": "OLMo-2-7B", "gemma-7b": "Gemma-7B", "apertus-8b-2509": "Apertus-8B",
        "mistral-7b-v0.3": "Mistral-7B"}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def f(x, d=2, sign=False):
    x = round(float(x), d) + 0.0  # avoid printing negative zero
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def ci(lo, hi, d=3):
    return f"[{lo:+.{d}f}, {hi:+.{d}f}]"


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.results_root
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    ext = lambda m, name: read(root / m / "extensions" / name)

    a = {m: rq_a_stats(root, m) for m in MODELS}
    c = {m: rq_c_stats(root, m) for m in MODELS}
    b = {m: rq_b_stats(root, m) for m in MODELS}
    lsi = {m: (ext(m, "lsi_latent/results.json")["off_diagonal_means"], ext(m, "lsi_latent/inference.json")["comparison"]) for m in MODELS}
    x1 = {m: ext(m, "alignment_baselines.json") for m in MODELS}
    x3 = {m: ext(m, "mmmlu_transfer.json") for m in MODELS}
    x6 = {m: ext(m, "include_transfer.json") for m in MODELS}
    x7 = {m: ext(m, "transfer_predictors.json") for m in MODELS}
    meta = read(root / "x7_meta.json")
    x2 = {m: ext(m, "damage_budgeted_steering.json") for m in CORE}

    # ------------------------------------------------------------- macros
    mac = ""
    mac += macro("nModels", str(len(MODELS)))
    cos_pos = [m for m in MODELS if a[m]["cos_slope"]["p025"] > 0]
    auc_null = [m for m in MODELS if a[m]["auc_slope"]["p025"] <= 0 <= a[m]["auc_slope"]["p975"]]
    mac += macro("nCosSlopePos", str(len(cos_pos)))
    mac += macro("nAucSlopeNull", str(len(auc_null)))
    mac += macro("cosSlopeMin", f(min(a[m]["cos_slope"]["mean"] for m in MODELS), 3, True))
    mac += macro("cosSlopeMax", f(max(a[m]["cos_slope"]["mean"] for m in MODELS), 3, True))
    mac += macro("aucSlopeMaxAbs", f(max(abs(a[m]["auc_slope"]["mean"]) for m in MODELS), 4))
    mac += macro("aucSlopeHalfWidthMax", f(max((a[m]["auc_slope"]["p975"] - a[m]["auc_slope"]["p025"]) / 2 for m in MODELS), 3))
    mac += macro("nCosTestsSig", str(sum(a[m]["cos_sig"] for m in MODELS)))
    mac += macro("nCosTests", str(15 * len(MODELS)))
    mac += macro("nAucTestsSig", str(sum(a[m]["auc_sig"] for m in MODELS)))
    mac += macro("nAucTests", str(30 * len(MODELS)))
    lsi_down = [m for m in MODELS if lsi[m][1]["auroc"]["ci95"][1] < 0]
    lsi_up = [m for m in MODELS if lsi[m][1]["auroc"]["ci95"][0] > 0]
    lsi_cos_up = [m for m in MODELS if lsi[m][0]["direction_cosine"] > c[m]["cos"] + 0.02]
    mac += macro("nLatentDown", str(len(lsi_down)))
    mac += macro("nLatentUp", str(len(lsi_up)))
    mac += macro("nLatentCosUp", str(len(lsi_cos_up)))
    mac += macro("latentWorst", f(min(lsi[m][1]["auroc"]["mean_change"] for m in MODELS), 3, True))
    rosh = {m: x1[m]["paired_macro_bootstrap_vs_raw"]["rosh_fixed_layer"]["auroc"] for m in MODELS}
    cen = {m: x1[m]["paired_macro_bootstrap_vs_raw"]["centroid_shift"]["balanced_accuracy"] for m in MODELS}
    # Holm-adjusted, matching the asterisks in the X1 table.
    mac += macro("nRoshUp", str(sum(1 for v in rosh.values() if v["p_holm"] < .05 and v["mean_change"] > 0)))
    mac += macro("nRoshDown", str(sum(1 for v in rosh.values() if v["p_holm"] < .05 and v["mean_change"] < 0)))
    mac += macro("nCentroidUp", str(sum(1 for v in cen.values() if v["p_holm"] < .05 and v["mean_change"] > 0)))
    mac += macro("centroidMin", f(min(v["mean_change"] for v in cen.values()), 3, True))
    mac += macro("centroidMax", f(max(v["mean_change"] for v in cen.values()), 3, True))
    # transport
    mac += macro("nAucAboveBa", str(sum(1 for m in MODELS if c[m]["auc"] > c[m]["ba"])))
    mac += macro("nRecenterUp", str(sum(1 for m in MODELS if c[m]["recenter_gain"] > 0)))
    # X7
    n_cfg = sum(x7[m]["n_configurations"] for m in MODELS)
    mac += macro("nConfigs", f"{n_cfg:,}".replace(",", "{,}"))
    rho = lambda m, kind, key: x7[m]["spearman_with_auroc_" + kind][key]
    for key, short in (("p1_euclidean_cosine", "Cos"), ("p2_translation_consistency", "Cons"), ("p3_fisher_efficiency", "Fish")):
        vals = [rho(m, "all_blocks", key) for m in MODELS]
        fro = [rho(m, "frozen_block", key) for m in MODELS]
        mac += macro(f"rho{short}Min", f(min(vals))) + macro(f"rho{short}Max", f(max(vals))) + macro(f"rho{short}Med", f(float(np.median(vals))))
        mac += macro(f"rhoFrozen{short}Min", f(min(fro))) + macro(f"rhoFrozen{short}Max", f(max(fro))) + macro(f"rhoFrozen{short}Med", f(float(np.median(fro))))
        mac += macro(f"lomo{short}", f(meta["h2_leave_one_model_out_r2"][key]["mean"]))
    best_frozen = sum(1 for m in MODELS if rho(m, "frozen_block", "p3_fisher_efficiency") >
                      max(rho(m, "frozen_block", "p1_euclidean_cosine"), rho(m, "frozen_block", "p2_translation_consistency")))
    best_all = sum(1 for m in MODELS if rho(m, "all_blocks", "p3_fisher_efficiency") >
                   max(rho(m, "all_blocks", "p1_euclidean_cosine"), rho(m, "all_blocks", "p2_translation_consistency")))
    mac += macro("nFisherBestFrozen", str(best_frozen)) + macro("nFisherBestAll", str(best_all))
    h1 = meta["h1"]
    mac += macro("nHOneAbove", str(h1["models_with_delta_ci_above_zero"])) + macro("nHOneBelow", str(h1["models_with_delta_ci_below_zero"]))
    mac += macro("hOneMeanDelta", f(h1["mean_delta_rho"], 2, True)) + macro("hOneSignP", f(h1["sign_test_p"], 3))
    ov = {k: [x7[m]["overlap_secondary"]["slopes_per_full_overlap"][k] for m in MODELS] for k in ("p1", "p2", "auroc")}
    mac += macro("ovCosMin", f(min(ov["p1"]), 3, True)) + macro("ovCosMax", f(max(ov["p1"]), 3, True))
    mac += macro("ovConsMin", f(min(ov["p2"]), 3, True)) + macro("ovConsMax", f(max(ov["p2"]), 3, True))
    mac += macro("ovAucMaxAbs", f(max(abs(v) for v in ov["auroc"]), 4))
    mac += macro("nXOneConsBetter", str(sum(1 for m in MODELS if x7[m]["x1_methods_secondary"]["spearman_over_methods_and_pairs"]["p2"] >
                                            x7[m]["x1_methods_secondary"]["spearman_over_methods_and_pairs"]["p1"])))
    ridge = {m: x7[m]["x1_methods_secondary"]["method_means"] for m in MODELS}
    mac += macro("ridgeCosMax", f(max(ridge[m]["ridge"]["p1"] for m in MODELS)))
    mac += macro("ridgeAucMin", f(min(ridge[m]["ridge"]["auroc"] for m in MODELS)))
    sec = meta["secondary_training_checkpoints"]
    steps = sorted(sec, key=lambda s: int(s.split("step")[-1]))
    mac += macro("pythiaCosEarly", f(sec[steps[1]]["rho_all_blocks"]["p1_euclidean_cosine"]))
    mac += macro("pythiaCosLate", f(sec[steps[-1]]["rho_all_blocks"]["p1_euclidean_cosine"]))
    mac += macro("pythiaDeltaLate", f(sec[steps[-1]]["delta_rho"], 2, True))
    mac += macro("pythiaDeltaLateCI", ci(*sec[steps[-1]]["delta_ci95"], d=2))
    # external and scale
    p3 = {m: x3[m]["off_diagonal_means"]["paired_candidate_accuracy"] for m in MODELS}
    p6 = {m: x6[m]["off_diagonal_means"]["paired_candidate_accuracy"] for m in MODELS}
    mac += macro("mmmluMin", f(min(p3.values()), 3)) + macro("mmmluMax", f(max(p3.values()), 3))
    mac += macro("includeMin", f(min(p6.values()), 3)) + macro("includeMax", f(max(p6.values()), 3))
    qs = ["qwen3-0.6b-base", "qwen3-1.7b-base", "qwen3-4b-base", "qwen3-8b-base", "qwen3-14b-base"]
    mac += macro("scaleCrossLo", f(c[qs[0]]["auc"], 3)) + macro("scaleCrossHi", f(c[qs[-1]]["auc"], 3))
    mac += macro("scaleMmmluLo", f(p3[qs[0]], 3)) + macro("scaleMmmluHi", f(p3[qs[-1]], 3))
    mac += macro("scaleIncludeLo", f(p6[qs[0]], 3)) + macro("scaleIncludeHi", f(p6[qs[-1]], 3))
    mac += macro("postCross", f(c["qwen3-8b"]["auc"], 3)) + macro("baseCross", f(c["qwen3-8b-base"]["auc"], 3))
    # steering
    for m, short in (("qwen3-8b-base", "Qwen"), ("gemma-7b", "Gemma"), ("apertus-8b-2509", "Apertus"), ("mistral-7b-v0.3", "Mistral")):
        agg = x2[m]["aggregate"]["zero_overlap_source"]
        mac += macro(f"safeSlope{short}", "--" if agg["mean_symmetric_slope"] is None else f(agg["mean_symmetric_slope"], 3, True))
        mac += macro(f"safePairs{short}", f'{agg["pairs_with_safe_nonzero_alpha"]}/{agg["pairs"]}')
    # difficulty (RQ-B)
    drops = {m: b[m]["auc_original"] - b[m]["auc_residual"] for m in MODELS}
    mac += macro("nDiffLowersAuc", str(sum(1 for v in drops.values() if v > 0)))
    lowering = [v for v in drops.values() if v > 0]
    mac += macro("diffDropMin", f(min(lowering), 3)) + macro("diffDropMax", f(max(lowering), 3))
    mac += macro("diffResidMin", f(min(b[m]["auc_residual"] for m in MODELS), 2))
    mac += macro("nDiffLowersCos", str(sum(1 for m in MODELS if b[m]["cos_residual"] < b[m]["cos_original"])))
    # Apertus case
    mac += macro("apertusRhoCos", f(rho("apertus-8b-2509", "all_blocks", "p1_euclidean_cosine")))
    mac += macro("apertusRhoFish", f(rho("apertus-8b-2509", "all_blocks", "p3_fisher_efficiency")))
    mac += macro("apertusFrozenCons", f(rho("apertus-8b-2509", "frozen_block", "p2_translation_consistency")))
    mac += macro("apertusFrozenConsSpearman", f(rho("apertus-8b-2509", "frozen_block", "p2_spearman")))
    # X10-X13 (protocol/X10_X11_PREREGISTRATION.md)
    meta_path = root / "x10_x13_meta.json"
    if meta_path.exists():
        mx = read(meta_path)
        x10m, x11m, x12m, x13m = mx["x10"], mx["x11"], mx["x12"], mx["x13"]
        mac += macro("xTenUnits", str(x10m["n_units"]))
        mac += macro("xTenR", f(x10m["slope_r"])) + macro("xTenRatio", f(x10m["median_ratio_observed_over_predicted"]))
        mac += macro("xTenRatioLo", f(x10m["ratio_iqr"][0])) + macro("xTenRatioHi", f(x10m["ratio_iqr"][1]))
        mac += macro("xTenOrigin", f(x10m["through_origin_slope"]))
        mac += macro("xTenLevelZeroR", f(x10m["levels"]["0.0"]["r"])) + macro("xTenLevelOneR", f(x10m["levels"]["1.0"]["r"]))
        mac += macro("xTenLevelZeroMAE", f(x10m["levels"]["0.0"]["mean_abs_error"], 3)) + macro("xTenLevelOneMAE", f(x10m["levels"]["1.0"]["mean_abs_error"], 3))
        mac += macro("xTenVerdict", x10m["verdict"])
        # Post hoc sensitivity (not the preregistered verdict): Apertus excluded.
        rest = [m for m in MODELS if m != "apertus-8b-2509"]
        pairs = {m: ext(m, "x10_x12.json")["x10"]["pairs"] for m in MODELS}
        pr = np.asarray([q["predicted_slope"] for m in rest for q in pairs[m]])
        ob = np.asarray([q["observed_slope"] for m in rest for q in pairs[m]])
        mac += macro("xTenRExApertus", f(np.corrcoef(pr, ob)[0, 1])) + macro("xTenRatioExApertus", f(np.median(ob / pr)))
        mac += macro("xTenModelRMin", f(min(x10m["per_model_slope_r"][m] for m in rest)))
        mac += macro("xTenModelRMax", f(max(x10m["per_model_slope_r"][m] for m in rest)))
        ratios = {m: np.median([q["observed_slope"] / q["predicted_slope"] for q in pairs[m]]) for m in MODELS}
        mac += macro("xTenModelRatioMin", f(min(ratios[m] for m in rest))) + macro("xTenModelRatioMax", f(max(ratios[m] for m in rest)))
        mac += macro("xTenOverstatePct", str(int(round(100 * (1 - float(np.median(ob / pr)))))))
        mac += macro("xTenApertusR", f(x10m["per_model_slope_r"]["apertus-8b-2509"])) + macro("xTenApertusRatio", f(ratios["apertus-8b-2509"]))
        mac += macro("xElevenMaxDelta", f(x11m["max_abs_delta_cross_auroc"], 3)) + macro("xElevenBestUnchanged", str(x11m["n_best_unchanged"]))
        mac += macro("xElevenVerdict", x11m["verdict"])
        for k in ("8", "16", "64"):
            for name in ("direct", "fisher", "cos"):
                mac += macro(f"xTwelve{name.capitalize()}{'ABC'[['8','16','64'].index(k)]}", f(x12m["curves_mean_rho_over_models"][k][name]))
        mac += macro("xTwelvePositive", str(x12m["k16_models_positive"])) + macro("xTwelveVerdict", x12m["verdict"])
        for fam, short in (("mm", "MM"), ("lda", "LDA"), ("lr", "LR")):
            v = x13m["families"][fam]
            mac += macro(f"xThirteen{short}N", str(v["n_models_dissociation"])) + macro(f"xThirteen{short}Slope", f(v["mean_cosine_slope"], 3, True))
            mac += macro(f"xThirteen{short}AucSlope", f(v["max_abs_auroc_slope"], 4)) + macro(f"xThirteen{short}Cross", f(v["mean_cross_auroc"], 3))
            mac += macro(f"xThirteen{short}BA", f(v["mean_balanced_accuracy"], 3))
            mac += macro(f"xThirteen{short}RhoCos", f(v["median_spearman_cosine"])) + macro(f"xThirteen{short}RhoFish", f(v["median_spearman_fisher"]))
        mm_slope = x13m["families"]["mm"]["mean_cosine_slope"]
        mac += macro("xThirteenLDARatio", f(x13m["families"]["lda"]["mean_cosine_slope"] / mm_slope, 1))
        mac += macro("xThirteenLRRatio", f(x13m["families"]["lr"]["mean_cosine_slope"] / mm_slope, 1))
        mac += macro("nLrCAtEdge", str(sum(1 for c_ in x13m["lr_c"].values() if float(c_) == 0.001)))
        rows = []
        for m in MODELS:
            cells = [m and NAME[m]]
            for fam in ("mm", "lda", "lr"):
                v = x13m["families"][fam]["per_model"][m]
                cells += [f(v["mean_cross_auroc"], 3), f(v["overlap_slopes"]["cosine"]["mean"], 3, True), f(v["overlap_slopes"]["auroc"]["mean"], 4, True)]
            cells.append(f(x10m["per_model_slope_r"][m]))
            rows.append(" & ".join(cells) + r" \\")
        (out / "table_x13.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
        rows = []
        for k, v in x12m["curves_mean_rho_over_models"].items():
            rows.append(" & ".join([k, f(v["direct"]), f(v["fisher"]), f(v["cos"])]) + r" \\")
        (out / "table_x12.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (out / "numbers.tex").write_text(mac, encoding="utf-8")

    # ------------------------------------------------------------- tables
    rows = []
    for m in MODELS:
        s = a[m]
        rows.append(" & ".join([
            NAME[m], str(s["block"]),
            f'{f(s["cos_slope"]["mean"], 3, True)} {{\\scriptsize {ci(s["cos_slope"]["p025"], s["cos_slope"]["p975"])}}}',
            f'{f(s["auc_slope"]["mean"], 4, True)} {{\\scriptsize {ci(s["auc_slope"]["p025"], s["auc_slope"]["p975"])}}}',
            f(c[m]["auc"], 3),
            f(lsi[m][0]["direction_cosine"] - c[m]["cos"], 2, True),
            f'{f(lsi[m][1]["auroc"]["mean_change"], 3, True)}{"$^*$" if lsi[m][1]["auroc"]["ci95"][0] > 0 or lsi[m][1]["auroc"]["ci95"][1] < 0 else ""}',
            f(p3[m], 3), f(p6[m], 3)]) + r" \\")
    (out / "table_main.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for m in MODELS:
        h = x7[m]["h1_delta_rho_p2_minus_p1"]
        rows.append(" & ".join([NAME[m], f'{x7[m]["n_configurations"]:,}'.replace(",", "{,}"),
                                f(rho(m, "all_blocks", "p1_euclidean_cosine")), f(rho(m, "all_blocks", "p2_translation_consistency")),
                                r"\textbf{" + f(rho(m, "all_blocks", "p3_fisher_efficiency")) + "}" if rho(m, "all_blocks", "p3_fisher_efficiency") == max(rho(m, "all_blocks", k) for k in ("p1_euclidean_cosine", "p2_translation_consistency", "p3_fisher_efficiency")) else f(rho(m, "all_blocks", "p3_fisher_efficiency")),
                                f(rho(m, "frozen_block", "p1_euclidean_cosine")), f(rho(m, "frozen_block", "p2_translation_consistency")),
                                r"\textbf{" + f(rho(m, "frozen_block", "p3_fisher_efficiency")) + "}" if rho(m, "frozen_block", "p3_fisher_efficiency") == max(rho(m, "frozen_block", k) for k in ("p1_euclidean_cosine", "p2_translation_consistency", "p3_fisher_efficiency")) else f(rho(m, "frozen_block", "p3_fisher_efficiency")),
                                f'{f(h["observed"], 2, True)} {{\\scriptsize {ci(*h["ci95"], d=2)}}}']) + r" \\")
    (out / "table_x7.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for m in MODELS:
        inf = x1[m]["paired_macro_bootstrap_vs_raw"]
        cell = lambda key, metric: f'{f(inf[key][metric]["mean_change"], 3, True)}{"$^*$" if inf[key][metric]["p_holm"] < .05 else ""}'
        rows.append(" & ".join([NAME[m], f(x1[m]["summary_off_diagonal_means"]["raw_source_mass_mean"]["auroc"], 3),
                                cell("centroid_shift", "balanced_accuracy"), cell("rosh_fixed_layer", "auroc"),
                                cell("rosh_scrambled_correspondence", "auroc"), cell("ridge_unconstrained", "auroc"),
                                cell("lsi_raw_pca1", "auroc"), cell("target_native_mass_mean", "auroc")]) + r" \\")
    (out / "table_x1.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for m in MODELS:
        rows.append(" & ".join([NAME[m], f(c[m]["cos"], 3), f(c[m]["auc"], 3), f(c[m]["ba"], 3), f(c[m]["dprime"], 2),
                                f(c[m]["recenter_gain"], 3, True), f(b[m]["auc_original"], 3) + r"$\to$" + f(b[m]["auc_residual"], 3)]) + r" \\")
    (out / "table_transport.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for m in CORE:
        for direction, label in (("zero_overlap_source", "zero-overlap"), ("target_native", "native")):
            agg = x2[m]["aggregate"][direction]
            rows.append(" & ".join([NAME[m], label, f'{agg["pairs_with_safe_nonzero_alpha"]}/{agg["pairs"]}',
                                    "--" if agg["mean_safe_alpha"] is None else f(agg["mean_safe_alpha"], 2),
                                    "--" if agg["mean_symmetric_slope"] is None else f(agg["mean_symmetric_slope"], 3, True)]) + r" \\")
    (out / "table_x2.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for name in steps:
        v = sec[name]
        rows.append(" & ".join([f'{int(name.split("step")[-1]):,}'.replace(",", "{,}"), f(v["rho_all_blocks"]["p1_euclidean_cosine"]),
                                f(v["rho_all_blocks"]["p2_translation_consistency"]), f(v["rho_all_blocks"]["p3_fisher_efficiency"]),
                                f'{f(v["delta_rho"], 2, True)} {{\\scriptsize {ci(*v["delta_ci95"], d=2)}}}']) + r" \\")
    (out / "table_pythia.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    d = {m: rq_d_stats(root, m) for m in CORE if (root / m / "rq_d.json").exists()}
    rows = []
    for m, v in d.items():
        rows.append(" & ".join([NAME[m], *["--" if v[q]["mean_auc"] is None else f'{f(v[q]["mean_auc"], 2)} ({v[q]["valid"]})' for q in ("K/K", "K/U", "U/K", "U/U")]]) + r" \\")
    (out / "table_rqd.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    # ------------------------------------------------- X14, translation robustness, v5 replication
    mac2 = ""
    for tag, xr in (("Three", rs.x14(rs.V3)), ("Five", rs.x14(rs.V5))):
        w = xr["within"]
        mac2 += macro(f"xFourteenFishBeatsCos{tag}", str(xr["h14a"]["fisher_beats_cosine"]))
        mac2 += macro(f"xFourteenMcsGeFish{tag}", str(xr["h14a"]["mcs_optimal_ge_fisher"]))
        for key, short in (("p1_euclidean_cosine", "Cos"), ("p3_fisher_efficiency", "Fish"),
                           ("p4a_mcs_total_vs_target_probe", "McsProbe"), ("p4b_mcs_total_vs_target_optimal", "McsOpt"),
                           ("p5_prop1_predicted_auroc", "PropOne")):
            vals = [w[m][key] for m in w]
            mac2 += macro(f"xFourteen{short}Med{tag}", f(float(np.median(vals)))) + macro(f"xFourteen{short}Min{tag}", f(min(vals))) + macro(f"xFourteen{short}Max{tag}", f(max(vals)))
            if key in xr["lomo_r2"]:
                mac2 += macro(f"xFourteen{short}Lomo{tag}", f(xr["lomo_r2"][key]))
        mac2 += macro(f"propOneRtwo{tag}", f(xr["prop1"]["r2_no_fit"])) + macro(f"propOneMae{tag}", f(xr["prop1"]["mae"], 3))
        mac2 += macro(f"propOneBias{tag}", f(xr["prop1_bias"], 3, True))
        mac2 += macro(f"xFourteenApertusOpt{tag}", f(w["apertus-8b-2509"]["p4b_mcs_total_vs_target_optimal"]))
    h14d = read(rs.V3 / "x14_meta.json")["h14d"]
    ap = h14d["per_model"]["apertus-8b-2509"]
    others = [v["n_outlier_dims"] for m, v in h14d["per_model"].items() if m != "apertus-8b-2509"]
    mac2 += macro("apertusOutlierDims", str(ap["n_outlier_dims"])) + macro("apertusOutlierShare", str(int(round(100 * ap["outlier_variance_share"]))))
    mac2 += macro("otherOutlierDimsMax", str(max(others)))
    mac2 += macro("apertusXTenRaw", f(ap["x10_slope_r_raw"])) + macro("apertusXTenReduced", f(ap["x10_slope_r_reduced"]))
    mac2 += macro("apertusRhoCosReduced", f(ap["rho_cos_reduced"])) + macro("apertusRhoCosRawFrozen", f(ap["rho_cos_raw"]))
    mac2 += macro("hFourteenDSupported", "supported" if h14d["apertus_supported"] else "not supported")
    rows = []
    x3k, x5k = rs.x14(rs.V3), rs.x14(rs.V5)
    for key, label in rs.X14_KEYS.items():
        cells = [label]
        for xr in (x3k, x5k):
            vals = [xr["within"][m][key] for m in xr["within"]]
            cells += [f"{f(float(np.median(vals)))} {{\\scriptsize [{f(min(vals))}, {f(max(vals))}]}}",
                      f(xr["lomo_r2"][key]) if key in xr["lomo_r2"] else "--"]
        rows.append(" & ".join(cells) + r" \\")
    (out / "table_x14.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rep, tr = rs.replication(), rs.translation()
    for v, tag in (("v2", "Two"), ("v3", "Three"), ("v5", "Five"), ("v5_sf", "FiveSF")):
        g = rep["agg"][v]
        mac2 += macro(f"rep{tag}Dissoc", str(g["dissociation"])) + macro(f"rep{tag}AucFlat", str(g["auc_flat"]))
        mac2 += macro(f"rep{tag}CosUp", str(g["cos_up"])) + macro(f"rep{tag}FishBeats", str(g["fisher_beats_cos"]))
        mac2 += macro(f"rep{tag}Cross", f(g["mean_cross_auroc"], 3)) + macro(f"rep{tag}CosSlope", f(g["mean_cos_slope"], 3, True))
        mac2 += macro(f"rep{tag}RhoCos", f(g["median_rho_cos"])) + macro(f"rep{tag}RhoFish", f(g["median_rho_fisher"]))
    mac2 += macro("shortcutGainMin", f(rep["shortcut_gain"]["min"], 3)) + macro("shortcutGainMax", f(rep["shortcut_gain"]["max"], 3))
    mac2 += macro("vFiveSfMinusThreeMaxAbs", f(max(abs(rep["v5_sf_minus_v3"]["min"]), abs(rep["v5_sf_minus_v3"]["max"])), 2))
    mac2 += macro("vFiveSfMinusThreeMean", f(rep["v5_sf_minus_v3"]["mean"], 3, True))
    mac2 += macro("cosSlopeRatioMed", f(rep["v3_over_v5sf_cos_slope"]["median"], 1))
    mac2 += macro("vFiveDissocModels", ", ".join(NAME[m] for m in rep["v5_sf_dissociation_models"]))
    comp = tr["comparison"]
    mac2 += macro("transCrossDiff", f(comp["aggregate"]["cross_auroc"]["mean_diff"], 3, True)) + macro("transSameBlock", str(comp["same_block"]))
    fid = tr["fidelity"]
    for s, short in (("NLLB", "Nllb"), ("Opus", "Opus"), ("Google", "Google")):
        mac2 += macro(f"digitErr{short}", str(sum(fid[l][s][0] for l in fid))) + macro(f"unitSwap{short}", str(sum(fid[l][s][1] for l in fid)))
    mac2 += macro("nImperial", str(tr["n_imperial"]))
    labse = tr["labse"]
    mac2 += macro("labseNllbWins", str(sum(1 for l in labse if labse[l]["NLLB"] > max(labse[l]["LLM"], labse[l]["Google"]))))
    mac2 += macro("labseMaxGap", f(max(labse[l]["NLLB"] - labse[l]["LLM"] for l in labse), 3))
    mb = rs.mistral_v5_behavior()
    mac2 += macro("mistralHiTrue", str(int(round(100 * mb["hi"]["share_true"])))) + macro("mistralFrFalse", str(int(round(100 * (1 - mb["fr"]["share_true"])))))
    x7v5 = read(rs.V5 / "x7_meta.json")["h1"]
    mac2 += macro("nHOneAboveFive", str(x7v5["models_with_delta_ci_above_zero"]))
    rows = []
    for m in MODELS:
        r = rep["rows"][m]; s = r["v5_sf"]
        rows.append(" & ".join([NAME[m], f(r["v3"]["cross_auroc"], 3), f(r["v5"]["cross_auroc"], 3), f(s["cross_auroc"], 3),
                                f'{f(s["cos_slope"], 3, True)} {{\\scriptsize {ci(s["cos_lo"], s["cos_hi"])}}}',
                                f'{f(s["rho_fisher"])}/{f(s["rho_cos"])}']) + r" \\")
    (out / "table_v5.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    with (out / "numbers.tex").open("a", encoding="utf-8") as handle:
        handle.write(mac2)
    print(json.dumps({"output": str(out), "files": sorted(p.name for p in out.iterdir())}))


if __name__ == "__main__":
    main()
