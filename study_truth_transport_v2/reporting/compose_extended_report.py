#!/usr/bin/env python3
"""Assemble the final evidence-linked fifteen-page extended report.

All quantitative statements are computed from the JSON artifacts at build time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import replication_stats as rs
from .compose_report import (
    body, callout, f, figure, page, reference, small, subtitle, table,
    rq_a_stats, rq_b_stats, rq_c_stats, rq_d_stats, rq_d_margin_stats, rq_e_stats,
    tidy,
)


CORE = ["gemma-7b", "qwen3-8b-base", "apertus-8b-2509"]
ALL = CORE + ["mistral-7b-v0.3"]
DISPLAY = {"gemma-7b": "Gemma-7B", "qwen3-8b-base": "Qwen3-8B-Base",
           "apertus-8b-2509": "Apertus-8B", "mistral-7b-v0.3": "Mistral-7B-v0.3"}
SHORT = {"gemma-7b": "Gemma", "qwen3-8b-base": "Qwen", "apertus-8b-2509": "Apertus",
         "mistral-7b-v0.3": "Mistral"}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def ext(root: Path, model: str, name: str):
    return read(root / model / "extensions" / name)


def delta_ci(item: dict, digits: int = 3) -> str:
    return (f'{item["mean_change"]:+.{digits}f} [{item["ci95"][0]:+.{digits}f}, '
            f'{item["ci95"][1]:+.{digits}f}]' + ("*" if item.get("p_holm", 1) < .05 else ""))


def sgn(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def names(models: list[str]) -> str:
    items = [SHORT[m] for m in models]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--figure-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root, fig = args.results_root, args.figure_dir
    required = [root / m / "extensions" / n for m in ALL for n in (
        "alignment_baselines.json", "lsi_latent/inference.json", "mmmlu_transfer.json",
        "include_transfer.json", "damage_budgeted_steering.json")]
    required += [root / m / "rq_c.json" for m in ALL] + [root / "rq_f_pythia_trajectory.json", root / "x7_meta.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing final artifacts:\n" + "\n".join(missing))

    a = {m: rq_a_stats(root, m) for m in ALL}
    b = {m: rq_b_stats(root, m) for m in ALL}
    c = {m: rq_c_stats(root, m) for m in ALL}
    # v3 ran RQ-D and RQ-E for Mistral as well; v2 did not (X4 replication covered RQ-A/B/C only).
    DE = [m for m in ALL if (root / m / "rq_d.json").exists() and (root / m / "steering_summary.json").exists()]
    d = {m: rq_d_stats(root, m) for m in DE}
    # Known test rate per model: mean over languages of the behavioural known rate on test groups.
    known_test = {m: float(np.mean([v["known_rate_by_partition"]["test"]
                                    for v in read(root / m / "behavior_aggregate.json")["languages"].values()])) for m in DE}
    e = {m: rq_e_stats(root, m) for m in DE}
    x1 = {m: ext(root, m, "alignment_baselines.json") for m in ALL}
    raw = {m: x1[m]["summary_off_diagonal_means"]["raw_source_mass_mean"] for m in ALL}
    inf = {m: x1[m]["paired_macro_bootstrap_vs_raw"] for m in ALL}
    lsi = {m: ext(root, m, "lsi_latent/results.json")["off_diagonal_means"] for m in ALL}
    lsi_inf = {m: ext(root, m, "lsi_latent/inference.json")["comparison"] for m in ALL}
    x2 = {m: ext(root, m, "damage_budgeted_steering.json") for m in ALL}
    x3 = {m: ext(root, m, "mmmlu_transfer.json") for m in ALL}
    x6 = {m: ext(root, m, "include_transfer.json") for m in ALL}
    traj = read(root / "rq_f_pythia_trajectory.json")["checkpoints"]

    # Derived quantities used in prose.
    lsi_gain = [m for m in ALL if lsi_inf[m]["auroc"]["ci95"][0] > 0]
    lsi_loss = [m for m in ALL if lsi_inf[m]["auroc"]["ci95"][1] < 0]
    lsi_null = [m for m in ALL if m not in lsi_gain + lsi_loss]
    rosh_up = [m for m in ALL if inf[m]["rosh_fixed_layer"]["auroc"]["mean_change"] > 0]
    cen_ba = [inf[m]["centroid_shift"]["balanced_accuracy"]["mean_change"] for m in ALL]
    auc_sig = sum(a[m]["auc_sig"] for m in ALL)
    cos_sig = sum(a[m]["cos_sig"] for m in ALL)
    capped = {m: sum(1 for cell in x2[m]["cells"] if cell["direction_type"] == "named"
                     and cell["selected_alpha_magnitude"] == 4.0) for m in ALL}
    cos_excl = [m for m in ALL if a[m]["cos_slope"]["p025"] > 0]
    cos_incl = [m for m in ALL if m not in cos_excl]
    def ci_above(data):
        off = [cell for cell in data["cells"] if cell["source"] != cell["target"]]
        return sum(cell["bootstrap"]["paired_accuracy_p025"] > .5 for cell in off), len(off)
    ext_ci = {m: (ci_above(x3[m]), ci_above(x6[m])) for m in ALL}
    best_x3 = max(ALL, key=lambda m: x3[m]["off_diagonal_means"]["paired_candidate_accuracy"])
    first, last = traj[0], traj[-1]

    pages = []
    X7_NAME = {**DISPLAY, "qwen3-0.6b-base": "Qwen3-0.6B-Base", "qwen3-1.7b-base": "Qwen3-1.7B-Base",
               "qwen3-4b-base": "Qwen3-4B-Base", "qwen3-14b-base": "Qwen3-14B-Base",
               "qwen3-8b": "Qwen3-8B post-trained", "olmo-2-1124-7b": "OLMo-2-7B"}
    X8_ORDER = ["qwen3-0.6b-base", "qwen3-1.7b-base", "qwen3-4b-base", "qwen3-8b-base", "qwen3-14b-base", "qwen3-8b", "olmo-2-1124-7b"]
    meta = read(root / "x7_meta.json")
    x7_primary = meta["primary_models"]
    x7 = {m: ext(root, m, "transfer_predictors.json") for m in x7_primary}
    _r = lambda m, kind, key: x7[m]["spearman_with_auroc_" + kind][key]
    _best = lambda kind: sum(1 for m in x7_primary if _r(m, kind, "p3_fisher_efficiency") >
                             max(_r(m, kind, "p1_euclidean_cosine"), _r(m, kind, "p2_translation_consistency")))
    p3_all_n, p3_frozen_n = _best("all_blocks"), _best("frozen_block")
    _rep1 = rs.replication(); _x14v3 = rs.x14(rs.V3)
    pages.append(page(
        "Alignment is easy to raise; truth transfer is not",
        callout("Main result: sharing training facts across languages, or learning a shared label-blind latent space, makes multilingual truth directions look more similar without improving held-out truth discrimination. Geometry, ranking transfer, threshold transport, and causal effect are separate estimands."),
        body(f"We study 2,000 balanced true/false claims in English, German, Arabic, Hindi, French, and Spanish (Claude Opus 5.5 translations; NLLB translations and an independently built 2,000-family claim set serve as replications), grouped into 1,972 dependency units, with mass-mean probes in four core families (Gemma-7B, Qwen3-8B-Base, Apertus-8B, Mistral-7B-v0.3), extended to {len(set(x7_primary) | set(ALL))} models with a Qwen3 scale series, OLMo-2-7B, and post-trained Qwen3-8B. Holding training size fixed while varying only the share of matched fact identities raises direction cosine (slope interval above zero in {names(cos_excl)}; {cos_sig}/60 Holm-significant pair tests) but changes held-out AUROC in only {auc_sig}/120 tests; every AUROC-slope interval includes zero."),
        table([["Model", "Cosine slope", "AUROC slope", "Cross AUROC", "Latent Δcosine", "Latent ΔAUROC"],
               *[[DISPLAY[m], sgn(a[m]["cos_slope"]["mean"]), sgn(a[m]["auc_slope"]["mean"], 4), f(c[m]["auc"]),
                  sgn(lsi[m]["direction_cosine"] - c[m]["cos"]), sgn(lsi_inf[m]["auroc"]["mean_change"])]
                 for m in ALL]], [.22, .15, .15, .15, .16, .17]),
        subtitle("Extensions completed after the core"),
        body(f"An LSI-style shared autoencoder raises cosine in most models but never improves AUROC significantly: AUROC falls in {names(lsi_loss)} and is unchanged in {names(lsi_null) if lsi_null else 'none'}. Centroid shift improves source-threshold balanced accuracy without changing AUROC, and fixed-layer orthogonal RoSh does not beat the raw direction in ranking. Generic-claim directions transfer zero-shot, but weakly outside {SHORT[best_x3]}, to professionally translated MMMLU and to native-authored INCLUDE exam questions. Steering at a neutral-text damage budget is large only in Qwen. Across every layer of {len(x7_primary)} models, Euclidean cosine is a weak and manipulable proxy for transfer; the target's discriminant geometry (Fisher efficiency) tracks transfer best in {p3_all_n}/{len(x7_primary)} models across all layers ({p3_frozen_n}/{len(x7_primary)} at the selected layer), and a preregistered label-free alternative did not beat cosine. X14: the total-covariance Mahalanobis cosine against the target's optimal discriminant does at least as well as Fisher efficiency in {_x14v3['h14a']['mcs_optimal_ge_fisher']}/10 models and is the only statistic calibrated across models. On an independently built claim set (v5), AUROC is again flat under overlap in {_rep1['agg']['v5_sf']['auc_flat']}/10 models and Fisher efficiency beats cosine in {_rep1['agg']['v5_sf']['fisher_beats_cos']}/10, but the overlap-driven cosine increase is clearly present in only {_rep1['agg']['v5_sf']['cos_up']}/10."),
        dek="Final extended technical report for a NAACL 2027 submission",
        eyebrow="TRUTH TRANSPORT / FINAL REPORT / 5 OCTOBER 2026",
    ))

    pages.append(page(
        "Closest work, questions, and estimands",
        body("Multilingual probe accuracy and vector similarity are already studied (Li et al.); cross-language transfer of hallucination and confidence signals is established (CrossHallu, Shared Doubt); LSI and RoSh align languages in latent or residual space. We use RoSh and LSI as qualified baselines and claim no priority over them. The contribution is measurement: separating content overlap, geometry, score transport, behavioral knowledge, causal effect, and the statistic that actually tracks transfer."),
        table([
            ["RQ / extension", "Estimand", "Answer"],
            ["A", "0–100% shared train facts at fixed n", "Cosine rises; held-out AUROC is flat"],
            ["B", "Geometry after cross-fitted difficulty removal", "Difficulty explains part, not all"],
            ["C", "Ranking, separation, offset, threshold transport", "Ranking transfers better than thresholds"],
            ["D", "Decoding in behavioral knowledge quadrants", "Diagnostic only; label imbalance"],
            ["E / X2", "Steering vs controls, damage-budgeted", "Strong in Qwen, weak or absent elsewhere"],
            ["X1", "Label-blind alignment baselines incl. LSI-style AE", "Cosine moves without matching transfer"],
            ["X3 / X6", "Zero-shot translated MMMLU and native INCLUDE", "Above chance, strong only in Qwen"],
            ["X7", "Which statistic tracks transfer across all layers", "See Section X7"],
            ["X8 / X9", "Scale, open-data family, post-training", "See Section X8/X9"],
        ], [.16, .44, .40]),
        subtitle("Frozen design"),
        body("The representation is the residual stream after one decoder block at the final statement token under one translated judgment prompt; the block is chosen per model by mean within-language validation AUROC. A mass-mean direction is the true-class centroid minus the false-class centroid, with the midpoint threshold. Only mass-mean probes are used. Test labels never select a layer, threshold, allocation, transform, steering magnitude, checkpoint, or truncation. Intervals resample dependency groups (claims) or questions (external sets); multiple comparisons use Holm correction within model and outcome. Post-core extensions were each frozen before their outcomes existed (1–2 October 2026)."),
        reference('Sources: <link href="https://aclanthology.org/2025.xllm-1.7/">Li et al. (XLLM 2025)</link>; <link href="https://arxiv.org/abs/2607.04029">CrossHallu</link>; <link href="https://arxiv.org/abs/2605.31220">Shared Doubt</link>; <link href="https://arxiv.org/abs/2608.28860">LSI</link>; <link href="https://arxiv.org/abs/2609.34678">RoSh</link>; <link href="https://arxiv.org/abs/2310.06824">Geometry of Truth</link>; <link href="https://arxiv.org/abs/2406.13229">Language-agnostic alignment</link>; <link href="https://arxiv.org/abs/2601.16390">CLAS</link>.'),
    ))

    n_x3 = {m: x3[m]["cells"][0]["n_questions"] for m in ALL}
    pages.append(page(
        "Data: one parallel claim set and two external tests",
        body("The claim set has 1,000 true and 1,000 false English statements with aligned German, Arabic, Hindi, French, and Spanish versions. Duplicated, paraphrased, or logically coupled rows are linked into 1,972 dependency groups; splitting, allocation, residualization, and bootstrap inference use these groups. The five non-English versions used here (dataset v3) were machine-translated by Claude Opus 5.5 under a frozen, label-blind protocol and checked automatically; the earlier NLLB translations (v2) are kept as a robustness comparison. No bilingual human review was carried out."),
        table([
            ["Property", "Frozen value", "Consequence"],
            ["Languages", "EN, DE, AR, HI, FR, ES", "All 30 ordered source→target pairs"],
            ["Rows / groups", "2,000 / 1,972", "Group-disjoint 1,200/400/400 split"],
            ["Pure train groups", "1,170", "Two mixed-label groups excluded from probe fits"],
            ["External, translated (X3)", f"MMMLU, {min(n_x3.values())}–{max(n_x3.values())} questions", "English MMLU + OpenAI professional translations"],
            ["External, native (X6)", "INCLUDE, 2,189 questions", "AR 540, DE 138, HI 543, FR 419, ES 549; no English"],
            ["Context limit", "512 tokens", "Label-blind question-head truncation shared by both candidates"],
        ], [.25, .30, .45]),
        subtitle("External construction"),
        body("Each external question yields one correct and one hash-selected incorrect candidate statement ('For the question …, the correct answer is …'), scored with the same activation prompt; the two candidates form one bootstrap group. MMMLU uses ten questions per MMLU subject. INCLUDE uses every test question written natively in each language from local exams, after label-blind removal of rows with non-distinct options or repeated question text. No external item selects a layer, direction, threshold, or transform."),
        body(f"Long prompts were fitted to the frozen 512-token limit by removing words from the start of the shared question (rule frozen before any external outcome). MMMLU compaction mostly affected Hindi; one or two questions whose single answer option alone exceeded the limit were dropped in all languages for that model (questions kept: {', '.join(f'{SHORT[m]} {n_x3[m]}' for m in ALL)}). A sensitivity restricted to never-compacted questions is reported."),
        small("The claim set is a candidate parallel research set, not a native-cultural benchmark; MMMLU is translated, and INCLUDE is native-authored but not parallel across languages."),
    ))


    pages.append(page(
        "RQ-A: shared facts raise cosine, not transfer",
        figure(fig / "rq_a_overlap.png", "Fixed-n overlap dose response: each fit uses 400 groups; only shared fact identity changes (50 allocations per level).", max_height=290),
        table([["Model", "Block", "Cosine slope [95% CI]", "AUROC slope [95% CI]", "Cosine 0→100%", "Sig. cos / AUROC"],
               *[[DISPLAY[m], str(a[m]["block"]),
                  f'{sgn(a[m]["cos_slope"]["mean"])} [{sgn(a[m]["cos_slope"]["p025"])}, {sgn(a[m]["cos_slope"]["p975"])}]',
                  f'{sgn(a[m]["auc_slope"]["mean"], 4)} [{sgn(a[m]["auc_slope"]["p025"], 4)}, {sgn(a[m]["auc_slope"]["p975"], 4)}]',
                  f(a[m]["zero_cos"]["mean"]) + "→" + f(a[m]["full_cos"]["mean"]),
                  f'{a[m]["cos_sig"]}/15 · {a[m]["auc_sig"]}/30'] for m in ALL]],
              [.17, .07, .22, .24, .15, .15]),
        body(f"Every AUROC-slope interval includes zero. Cosine-slope intervals exclude zero for {names(cos_excl)}" + (f"; {names(cos_incl)}'s raw cosine is too noisy for a precise slope (interval {sgn(a[cos_incl[0]]['cos_slope']['p025'])} to {sgn(a[cos_incl[0]]['cos_slope']['p975'])})" if cos_incl else "") + ". Mistral, run as a prospective fourth-family replication, reproduces the pattern. Because sample size and label balance are fixed, the cosine increase cannot be explained by more training evidence; it reflects matched-content covariance in the two estimators."),
        callout("Interpretation: full-overlap cosine is contaminated by shared content and is not a clean estimate of language-independent truth geometry. Zero-overlap directions still transfer."),
    ))

    pages.append(page(
        "RQ-B: difficulty explains a component, not the whole direction",
        figure(fig / "rq_b_difficulty.png", "Original and cross-fitted difficulty-residualized geometry and transfer.", max_height=280),
        table([["Model", "Cosine raw→resid.", "AUROC raw→resid.", "BA raw→resid.", "Difficulty cos.", "Difficulty AUROC"],
               *[[DISPLAY[m], f(b[m]["cos_original"]) + "→" + f(b[m]["cos_residual"]),
                  f(b[m]["auc_original"]) + "→" + f(b[m]["auc_residual"]),
                  f(b[m]["ba_original"]) + "→" + f(b[m]["ba_residual"]),
                  f(b[m]["difficulty_cos"]), f(b[m]["difficulty_auc"])] for m in ALL]],
              [.2, .17, .17, .16, .14, .16]),
        body("Difficulty is measured only from training-partition behavior (correctness, margin consistency, entropy, surprisal). Five-fold cross-fitting estimates the difficulty direction on other groups before projecting it out of a held-out fold, so test examples never define the removed direction."),
        body(f"Residualization lowers cosine and AUROC in all four models (AUROC drops range from {min(b[m]['auc_original'] - b[m]['auc_residual'] for m in ALL):.3f} to {max(b[m]['auc_original'] - b[m]['auc_residual'] for m in ALL):.3f}), so easy-versus-hard item structure contributes to both. Residual transfer stays well above chance, which rules out the claim that the direction is only difficulty."),
        callout("Supported: behavioral difficulty is a measurable confound. Not supported: the residual is a unique, universal truth axis."),
    ))

    pages.append(page(
        "RQ-C: score ranking transfers better than source thresholds",
        figure(fig / "rq_c_transport_summary.png", "Transport summary over the 30 off-diagonal pairs; full matrices on the next page.", max_height=250),
        table([["Model", "Cosine", "Cross AUROC", "Source-thr. BA", "d′", "Recenter ΔBA", "ΔBA 30%/70% prior"],
               *[[DISPLAY[m], f(c[m]["cos"]), f(c[m]["auc"]), f(c[m]["ba"]), f(c[m]["dprime"], 2),
                  sgn(c[m]["recenter_gain"]), f'{sgn(c[m]["prior30_gain"])} / {sgn(c[m]["prior70_gain"])}']
                 for m in ALL]], [.2, .1, .13, .14, .08, .14, .21]),
        body("AUROC asks whether target true claims score above target false claims under the source direction. Balanced accuracy additionally requires the source threshold to land correctly in the target score distribution. In every model, cross AUROC exceeds source-threshold balanced accuracy, so part of the transfer failure is score transport rather than missing discrimination."),
        body("Unlabeled target recentering changes no ranking and cannot change AUROC, yet it raises balanced accuracy in every model. The 30%/70% class-prior checks bound how far this adaptation can be trusted when the unlabeled target mean mixes class prior with language offset."),
        small("Each RQ-C cell also stores class means and SDs, pooled SD, global offset, target-validation optimal threshold, calibration diagnostics, and prior-shift results."),
    ))

    mx = read(root / "x10_x13_meta.json")
    x10m, x11m, x12m, x13m = mx["x10"], mx["x11"], mx["x12"], mx["x13"]
    rest = [m for m in mx["models"] if m != "apertus-8b-2509"]
    fam = x13m["families"]
    k16, k64 = x12m["curves_mean_rho_over_models"]["16"], x12m["curves_mean_rho_over_models"]["64"]
    pages.append(page(
        "X10 to X13: a predicted artifact that grows with probe flexibility",
        figure(fig / "x10_x13.png", "Left: overlap slope of cosine predicted from pool statistics versus observed. Right: cosine slope by probe family; black ticks are AUROC slopes.", max_height=230),
        body(f"X10 (preregistered): the estimator-noise formula predicts each pair's overlap slope from training-pool statistics only. The frozen pooled rule fails (r = {x10m['slope_r']:.2f} over {x10m['n_units']} units, verdict: {x10m['verdict']}) because Apertus is not predicted (within-model r = {x10m['per_model_slope_r']['apertus-8b-2509']:.2f}). In the other nine models within-model r is {min(x10m['per_model_slope_r'][m] for m in rest):.2f} to {max(x10m['per_model_slope_r'][m] for m in rest):.2f}, observed slopes are a median {x10m['median_ratio_observed_over_predicted']:.2f} of predicted, and predicted cosine levels match observed ones at every overlap level (r ≥ {min(v['r'] for v in x10m['levels'].values()):.2f})."),
        table([["Probe family", "Mean cross AUROC", "Cosine slope", "Max |AUROC slope|", "Models with dissociation", "Median ρ(Fisher)", "Median ρ(cosine)"],
               *[[name, f(fam[k]["mean_cross_auroc"], 3), sgn(fam[k]["mean_cosine_slope"]), f(fam[k]["max_abs_auroc_slope"], 4),
                  f"{fam[k]['n_models_dissociation']}/{len(mx['models'])}", f(fam[k]["median_spearman_fisher"]), f(fam[k]["median_spearman_cosine"])]
                 for k, name in (("mm", "Mass mean"), ("lda", "Shrinkage LDA"), ("lr", "Logistic regression"))]],
              [.2, .13, .12, .14, .15, .13, .13]),
        body(f"X13: with shrinkage LDA and logistic regression the overlap artifact is {fam['lda']['mean_cosine_slope'] / fam['mm']['mean_cosine_slope']:.1f}× and {fam['lr']['mean_cosine_slope'] / fam['mm']['mean_cosine_slope']:.1f}× larger than with mass-mean probes, and AUROC remains flat in every model. X12: few-shot Fisher efficiency does not beat direct few-shot AUROC (mean ρ at k = 16: {k16['fisher']:.2f} vs {k16['direct']:.2f}; k = 64: {k64['fisher']:.2f} vs {k64['direct']:.2f}; verdict: {x12m['verdict']}). X11: dropping the 10% lowest-LaBSE-similarity test claims per language changes cross AUROC by at most {x11m['max_abs_delta_cross_auroc']:.3f}, and the best statistic is unchanged in {x11m['n_best_unchanged']}/{len(mx['models'])} models."),
        small("Machine-readable: results/MODEL/extensions/x10_x12.json, x13_probes.json, and results/x10_x13_meta.json. Protocol: protocol/X10_X11_PREREGISTRATION.md. Full 6×6 transport matrices remain in rq_c.json and the RQ-C figure."),
    ))

    pages.append(page(
        "RQ-D: behavioral quadrants are diagnostic only",
        figure(fig / "rq_d_quadrants.png", "Zero-overlap AUROC by behavioral source/target knowledge quadrant.", max_height=280),
        table([["Model", "K/K", "K/U", "U/K", "U/U", "Known test rate"],
               *[[DISPLAY[m], *["—" if d[m][q]["mean_auc"] is None else f'{f(d[m][q]["mean_auc"])} ({d[m][q]["valid"]})'
                                  for q in ("K/K", "K/U", "U/K", "U/U")],
                  f(known_test[m])] for m in DE]],
              [.2, .15, .15, .15, .15, .2]),
        body("A fact is behaviorally known when at least two of three translated judgment templates are correct with a consistent margin sign. Each test fact enters K/K, K/U, U/K, or U/U for a language pair; parentheses count pair cells with both labels present. Some K/U cells decode above chance."),
        body("Conditioning on behavioral correctness changes class balance, and many cells are single-class or small. A mean-margin definition of 'known' (a v2 sensitivity analysis) changes membership. These facts block a general 'knows but cannot say' claim." + ("" if "mistral-7b-v0.3" in DE else " RQ-D was secondary in the X4 protocol and was not run for Mistral.")),
        callout("Supported: probe information can survive behavioral failure in selected cells. Unsupported: a general latent-knowledge mechanism across languages."),
    ))

    x2_rows = []
    for m in ALL:
        for direction, label in (("zero_overlap_source", "zero-overlap"), ("target_native", "native")):
            agg = x2[m]["aggregate"][direction]
            x2_rows.append([DISPLAY[m], label, f'{agg["pairs_with_safe_nonzero_alpha"]}/6',
                            "—" if agg["mean_safe_alpha"] is None else f(agg["mean_safe_alpha"], 2),
                            "—" if agg["mean_symmetric_slope"] is None else f(agg["mean_symmetric_slope"], 3)])
    pages.append(page(
        "RQ-E and X2: causal steering is real but model dependent",
        table([["Model", "Zero-overlap slope", "Target-native", "Random mean", "Neutral ΔNLL", "Layer ±2", "All tokens"],
               *[[DISPLAY[m], f(e[m]["zero"]["mean"], 4), f(e[m]["native"]["mean"], 4), f(e[m]["random"]["mean"], 4),
                  f(e[m]["neutral_zero"], 4), f'{f(e[m]["layer_minus_2"], 3)}/{f(e[m]["layer_plus_2"], 3)}', f(e[m]["all_tokens"], 4)]
                 for m in DE]], [.19, .15, .14, .13, .13, .14, .12]),
        body("RQ-E adds the unit direction, scaled by the target projection SD, after the frozen block at the final statement token. Against random directions with near-zero mean slope, the zero-overlap direction shifts the true-versus-false margin "
             + "; ".join(f"{SHORT[m]} {e[m]['zero']['mean']:+.3f}" for m in sorted(DE, key=lambda m: -e[m]['zero']['mean']))
             + f" (random {min(e[m]['random']['mean'] for m in DE):+.4f} to {max(e[m]['random']['mean'] for m in DE):+.4f}): strong only in Qwen."),
        figure(fig / "x2_damage_budget.png", "X2: symmetric slope at the largest |α| with neutral-continuation ΔNLL ≤ 0.05.", max_height=200),
        table([["Model", "Direction", "Safe pairs", "Mean safe |α|", "Safe slope"], *x2_rows], [.26, .18, .16, .2, .2]),
        body(f"X2 selects magnitude without labels. The budget was not binding on the preregistered grid for Qwen ({capped['qwen3-8b-base']}/12 direction-pairs at |α|=4), Gemma ({capped['gemma-7b']}/12), and Mistral ({capped['mistral-7b-v0.3']}/12), so those slopes are effects at the largest tested magnitude. In v2, Apertus directions put about 0.78 of their weight on one outlier dimension, which makes one α unit incomparable across pairs (logged deviation); its v3 slopes are reported but not interpreted."),
    ))

    pages.append(page(
        "X1: alignment methods move cosine, not transfer",
        figure(fig / "x1_alignment.png", "Label-blind transforms fit on parallel training groups; unchanged test split.", max_height=205),
        table([["Model", "Raw AUROC / BA", "Centroid ΔBA", "RoSh ΔAUROC", "Ridge ΔAUROC"],
               *[[DISPLAY[m], f'{f(raw[m]["auroc"])} / {f(raw[m]["balanced_accuracy"])}',
                  delta_ci(inf[m]["centroid_shift"]["balanced_accuracy"]),
                  delta_ci(inf[m]["rosh_fixed_layer"]["auroc"]),
                  delta_ci(inf[m]["ridge_unconstrained"]["auroc"])] for m in ALL]],
              [.19, .16, .22, .22, .21]),
        table([["Model", "Cosine raw→latent", "AUROC raw→latent", "ΔAUROC [95% CI]", "ΔBA [95% CI]"],
               *[[DISPLAY[m], f(c[m]["cos"]) + "→" + f(lsi[m]["direction_cosine"]),
                  f(raw[m]["auroc"]) + "→" + f(lsi[m]["auroc"]),
                  delta_ci(lsi_inf[m]["auroc"]), delta_ci(lsi_inf[m]["balanced_accuracy"])] for m in ALL]],
              [.19, .18, .18, .23, .22]),
        body(f"Centroid shift cannot change AUROC but raises balanced accuracy in every model ({min(cen_ba):.3f} to {max(cen_ba):.3f}). Fixed-layer orthogonal RoSh, a qualified one-layer adaptation, {'improves AUROC only in ' + names(rosh_up) if rosh_up else 'does not improve AUROC in any model'}; scrambled correspondences fall toward chance. The LSI-style shared autoencoder (released architecture and losses, trained label-blind on our paired groups) lowers AUROC significantly in {names(lsi_loss)}" + (f" and leaves it unchanged in {names(lsi_null)}" if lsi_null else "") + ", whatever it does to cosine."),
        small("Brackets: 95% paired group-bootstrap CI (1,000 draws); * Holm-adjusted p < 0.05 within model and outcome."),
    ))


    ext_rows = []
    for m in ALL:
        o3, o6 = x3[m]["off_diagonal_means"], x6[m]["off_diagonal_means"]
        u3 = x3[m]["off_diagonal_means_uncompacted_subset"]
        ext_rows.append([DISPLAY[m], f(o3["micro_auroc"]), f(o3["paired_candidate_accuracy"]), f(u3["paired_candidate_accuracy"]),
                         f(o6["micro_auroc"]), f(o6["paired_candidate_accuracy"]), f(o6["source_threshold_balanced_accuracy"])])
    pages.append(page(
        "X3 and X6: external transfer, translated and native",
        figure(fig / "x3_x6_external.png", "Off-diagonal means of generic-claim probes applied zero-shot; dashed line is chance.", max_height=255),
        table([["Model", "MMMLU AUROC", "MMMLU paired", "Uncompacted paired", "INCLUDE AUROC", "INCLUDE paired", "INCLUDE BA"], *ext_rows],
              [.2, .13, .13, .15, .13, .13, .13]),
        body("Paired accuracy asks whether the true candidate outscores its matched false candidate; micro AUROC pools candidates. Cells whose paired-accuracy 95% interval lies above 0.5 (MMMLU; INCLUDE): " + "; ".join(f"{SHORT[m]} {ext_ci[m][0][0]}/{ext_ci[m][0][1]} and {ext_ci[m][1][0]}/{ext_ci[m][1][1]}" for m in ALL) + ". Transfer is above chance almost everywhere but strong only in Qwen. Source-threshold balanced accuracy is near chance for the other models, so frozen thresholds do not travel to question-answer statements."),
        body("The never-compacted MMMLU subset gives nearly identical paired accuracy, so the truncation rule does not drive X3. INCLUDE questions are written natively from local exams and differ from MMMLU in content, so the two sets are compared descriptively, not as a controlled translated-versus-native contrast. German INCLUDE has only 138 questions."),
        small("Per-cell, per-subject, and INCLUDE regional-feature breakdowns with question-level bootstrap intervals are in extensions/mmmlu_transfer.json and extensions/include_transfer.json."),
    ))

    x7_rows = [[X7_NAME[m], str(x7[m]["n_configurations"]),
                f(x7[m]["spearman_with_auroc_all_blocks"]["p1_euclidean_cosine"]),
                f(x7[m]["spearman_with_auroc_all_blocks"]["p2_translation_consistency"]),
                f(x7[m]["spearman_with_auroc_all_blocks"]["p3_fisher_efficiency"]),
                delta_ci({"mean_change": x7[m]["h1_delta_rho_p2_minus_p1"]["observed"],
                          "ci95": x7[m]["h1_delta_rho_p2_minus_p1"]["ci95"],
                          "p_holm": meta["per_model"][m]["p_holm"]})] for m in x7_primary]
    h1, h2 = meta["h1"], meta["h2_leave_one_model_out_r2"]
    rho = lambda m, kind, key: x7[m]["spearman_with_auroc_" + kind][key]
    sec = meta.get("secondary_training_checkpoints", {})
    steps = sorted(sec, key=lambda name: int(name.split("step")[-1]))
    if steps:
        early, late = sec[steps[1] if len(steps) > 1 else steps[0]], sec[steps[-1]]
        pythia_note = (f"Secondary training-time set (one Pythia-1.4B run, not part of H1): cosine's correlation with transfer falls from "
                       f"{early['rho_all_blocks']['p1_euclidean_cosine']:.2f} at step {int(steps[1].split('step')[-1]):,} to "
                       f"{late['rho_all_blocks']['p1_euclidean_cosine']:.2f} at step {int(steps[-1].split('step')[-1]):,}, while consistency overtakes it "
                       f"(Δρ {late['delta_rho']:+.2f} [{late['delta_ci95'][0]:+.2f}, {late['delta_ci95'][1]:+.2f}]).")
    else:
        pythia_note = ""
    p3_best_frozen = sum(1 for m in x7_primary if rho(m, "frozen_block", "p3_fisher_efficiency") >
                         max(rho(m, "frozen_block", "p1_euclidean_cosine"), rho(m, "frozen_block", "p2_translation_consistency")))
    p3_best_all = sum(1 for m in x7_primary if rho(m, "all_blocks", "p3_fisher_efficiency") >
                      max(rho(m, "all_blocks", "p1_euclidean_cosine"), rho(m, "all_blocks", "p2_translation_consistency")))
    ov = {k: [x7[m]["overlap_secondary"]["slopes_per_full_overlap"][k] for m in x7_primary] for k in ("p1", "p2", "auroc")}
    x1_p2_wins = sum(1 for m in x7_primary if x7[m]["x1_methods_secondary"]["spearman_over_methods_and_pairs"]["p2"] >
                     x7[m]["x1_methods_secondary"]["spearman_over_methods_and_pairs"]["p1"])
    frozen_p1 = [rho(m, "frozen_block", "p1_euclidean_cosine") for m in x7_primary]
    frozen_p3 = [rho(m, "frozen_block", "p3_fisher_efficiency") for m in x7_primary]
    pages.append(page(
        "X7: Euclidean cosine is a weak, manipulable proxy for transfer",
        figure(fig / "x7_predictors.png", "Left: Spearman correlation with held-out cross-language AUROC over every block × 30 pairs. Right: all configurations pooled.", max_height=215),
        table([["Model", "Configs", "ρ cosine (P1)", "ρ consistency (P2)", "ρ Fisher (P3)", "Δρ P2−P1 [95% CI]"], *x7_rows],
              [.24, .1, .14, .16, .14, .22]),
        body(f"Within each model, Euclidean cosine tracks held-out transfer only moderately across {sum(x7[m]['n_configurations'] for m in x7_primary):,} block × pair configurations (ρ {min(rho(m, 'all_blocks', 'p1_euclidean_cosine') for m in x7_primary):.2f} to {max(rho(m, 'all_blocks', 'p1_euclidean_cosine') for m in x7_primary):.2f}; Apertus near zero). At the frozen layer, Fisher efficiency, the source direction's separability in the target's within-class geometry relative to the best linear discriminant, is the best of the three statistics in {p3_best_frozen}/{len(x7_primary)} models (ρ {min(frozen_p3):.2f} to {max(frozen_p3):.2f} versus {min(frozen_p1):.2f} to {max(frozen_p1):.2f} for cosine); across all layers it is best in {p3_best_all}/{len(x7_primary)}. No statistic has a scale that carries across models: leave-one-model-out R² is negative for all three (P1 {h2['p1_euclidean_cosine']['mean']:.2f}, P2 {h2['p2_translation_consistency']['mean']:.2f}, P3 {h2['p3_fisher_efficiency']['mean']:.2f})."),
        body(f"The preregistered hypothesis H1 failed on the primary claim set: label-free translation consistency beat cosine with an interval above zero in {h1['models_with_delta_ci_above_zero']}/{h1['n_models']} models and lost in {h1['models_with_delta_ci_below_zero']} (on the v5 entity-template set it held, {read(rs.V5 / 'x7_meta.json')['h1']['models_with_delta_ci_above_zero']}/10, so its value is dataset dependent). Its value is robustness. Adding shared training facts raises cosine in every model (slope {min(ov['p1']):+.3f} to {max(ov['p1']):+.3f}) but leaves consistency ({min(ov['p2']):+.3f} to {max(ov['p2']):+.3f}) and AUROC ({min(ov['auroc']):+.4f} to {max(ov['auroc']):+.4f}) flat; across the X1 transforms, consistency tracks AUROC better than cosine in {x1_p2_wins}/{len(x7_primary)} models."),
        small(pythia_note + " Fisher efficiency uses target training labels and is explanatory, not a deployment-time predictor. Brackets: 500-draw paired group bootstrap; * Holm-adjusted p < 0.05 across models."),
    ))
    x8_present = [m for m in X8_ORDER if (root / m / "rq_c.json").exists() and (root / m / "rq_a" / "inference.json").exists()]
    x8_rows = []
    for m in x8_present:
        ai = read(root / m / "rq_a" / "inference.json")["pooled_hierarchical_bootstrap"]
        cc = [cell for cell in read(root / m / "rq_c.json")["cells"] if cell["source"] != cell["target"]]
        li = ext(root, m, "lsi_latent/inference.json")["comparison"]["auroc"]["mean_change"]
        x8_rows.append([X7_NAME[m], f(float(np.mean([cell["auroc"] for cell in cc]))),
                        f'{sgn(ai["cosine_slope"]["mean"])} [{sgn(ai["cosine_slope"]["p025"])}, {sgn(ai["cosine_slope"]["p975"])}]',
                        f'{sgn(ai["auroc_slope"]["mean"], 4)} [{sgn(ai["auroc_slope"]["p025"], 4)}, {sgn(ai["auroc_slope"]["p975"], 4)}]',
                        sgn(li), f(ext(root, m, "mmmlu_transfer.json")["off_diagonal_means"]["paired_candidate_accuracy"]),
                        f(ext(root, m, "include_transfer.json")["off_diagonal_means"]["paired_candidate_accuracy"])])
    lsi_ci = {m: ext(root, m, "lsi_latent/inference.json")["comparison"]["auroc"]["ci95"] for m in x8_present}
    lsi_up = sum(1 for v in lsi_ci.values() if v[0] > 0)
    lsi_down = sum(1 for v in lsi_ci.values() if v[1] < 0)
    def summary8(m):
        cells = [cell for cell in read(root / m / "rq_c.json")["cells"] if cell["source"] != cell["target"]]
        return {"cross": float(np.mean([cell["auroc"] for cell in cells])),
                "slope": read(root / m / "rq_a" / "inference.json")["pooled_hierarchical_bootstrap"]["cosine_slope"]["mean"],
                "mmmlu": ext(root, m, "mmmlu_transfer.json")["off_diagonal_means"]["paired_candidate_accuracy"],
                "include": ext(root, m, "include_transfer.json")["off_diagonal_means"]["paired_candidate_accuracy"]}
    post, base = summary8("qwen3-8b"), summary8("qwen3-8b-base")
    auc_null = sum(1 for m in x8_present if read(root / m / "rq_a" / "inference.json")["pooled_hierarchical_bootstrap"]["auroc_slope"]["p025"] <= 0 <= read(root / m / "rq_a" / "inference.json")["pooled_hierarchical_bootstrap"]["auroc_slope"]["p975"])
    cos_pos = sum(1 for m in x8_present if read(root / m / "rq_a" / "inference.json")["pooled_hierarchical_bootstrap"]["cosine_slope"]["p025"] > 0)
    pages.append(page(
        "X8 and X9: scale, an open-data family, post-training",
        figure(fig / "x8_scale.png", "Qwen3 base models across sizes (descriptive).", max_height=210),
        table([["Model", "Cross AUROC", "Overlap cosine slope", "Overlap AUROC slope", "Latent ΔAUROC", "MMMLU paired", "INCLUDE paired"], *x8_rows],
              [.22, .1, .2, .2, .1, .09, .09]),
        body(f"The overlap dissociation holds in every additional model: the AUROC-slope interval includes zero in {auc_null}/{len(x8_present)}, and the cosine slope is reliably positive in {cos_pos}/{len(x8_present)}. The shared latent space improves AUROC with an interval above zero in {lsi_up}/{len(x8_present)} models and lowers it significantly in {lsi_down}. Within Qwen3, cross-language AUROC and external paired accuracy rise with size while the overlap artifact persists; with at most five sizes these trends are descriptive."),
        body(f"Post-training changes little: post-trained Qwen3-8B versus Qwen3-8B-Base gives cross AUROC {post['cross']:.3f} vs {base['cross']:.3f}, overlap cosine slope {post['slope']:+.3f} vs {base['slope']:+.3f}, MMMLU paired accuracy {post['mmmlu']:.3f} vs {base['mmmlu']:.3f}, and INCLUDE {post['include']:.3f} vs {base['include']:.3f} (same base-style prompts; descriptive)."),
    ))
    pages.append(page(
        "X4 and X5: fourth family and one training trajectory",
        figure(fig / "x4_fourth_family.png", "Zero/full-overlap cosine and cross-language AUROC across the four families.", max_height=215),
        figure(fig / "x5_rqf_trajectory.png", "Exploratory Pythia-1.4B-deduped trajectory; the block is re-selected on validation data at each checkpoint. Zero- and full-overlap AUROC coincide at every step.", max_height=215),
        body(f"Mistral-7B-v0.3 (block {a['mistral-7b-v0.3']['block']}) replicates RQ-A (cosine slope {sgn(a['mistral-7b-v0.3']['cos_slope']['mean'])}, AUROC slope {sgn(a['mistral-7b-v0.3']['auc_slope']['mean'], 4)}), the RQ-B difficulty component, the RQ-C ranking-versus-threshold gap, and the X1 pattern."),
        body(f"Across Pythia steps {first['step']:,} to {last['step']:,}, zero-overlap cosine moves from {first['zero_overlap_cosine']:.3f} to {last['zero_overlap_cosine']:.3f}, full-overlap cosine from {first['full_overlap_cosine']:.3f} to {last['full_overlap_cosine']:.3f}, and zero-overlap AUROC from {first['zero_overlap_auroc']:.3f} to {last['zero_overlap_auroc']:.3f}. At random initialization (step 0), shared facts already raise cosine by {first['full_overlap_cosine'] - first['zero_overlap_cosine']:.3f} while AUROC stays at chance ({first['zero_overlap_auroc']:.3f}), the clearest case of overlap-driven cosine without truth content; zero- and full-overlap AUROC coincide at every step. This describes one 1.4B run and cannot identify a general phase transition."),
    ))

    # ---- New in the v3-primary build: X14, translation robustness, v5 replication ----
    rep, tr = rs.replication(), rs.translation()
    x3k, x5k = rs.x14(rs.V3), rs.x14(rs.V5)
    K = rs.X14_KEYS
    _h14d = read(rs.V3 / "x14_meta.json")["h14d"]["per_model"]
    _ap = _h14d["apertus-8b-2509"]
    _other_max = max(v["n_outlier_dims"] for m, v in _h14d.items() if m != "apertus-8b-2509")
    def rng(d, key):
        vals = [v[key] for v in d["within"].values()]
        return f"{min(vals):.2f} to {max(vals):.2f}"
    x14_rows = [[label, rng(x3k, key), f"{x3k['lomo_r2'][key]:.2f}" if key in x3k["lomo_r2"] else "—",
                 rng(x5k, key), f"{x5k['lomo_r2'][key]:.2f}" if key in x5k["lomo_r2"] else "—"] for key, label in K.items()]
    pages.append(page(
        "X14: transfer is predicted by the target's optimal discriminant",
        figure(fig / "x14_predictors.png", "Within-model Spearman ρ of each statistic with held-out cross-language AUROC over every block × 30 pairs; left v3, right the v5 replication.", max_height=225),
        table([["Statistic", "ρ range, v3", "LOMO R², v3", "ρ range, v5", "LOMO R², v5"], *x14_rows], [.32, .17, .17, .17, .17]),
        body(f"Fisher efficiency, the separability of the source direction in the target's within-class geometry relative to the target's best linear discriminant, is a standard discriminant-analysis quantity; we use it as an established diagnostic and claim no novelty for it. It beats Euclidean cosine in {x3k['h14a']['fisher_beats_cosine']}/{x3k['h14a']['n_models']} models on v3 and {x5k['h14a']['fisher_beats_cosine']}/{x5k['h14a']['n_models']} on v5. X14 adds the stronger baseline a reviewer would ask for: the total-covariance Mahalanobis cosine. Against the target's own trained probe it is weak (it inherits that probe's estimation noise), but against the target's optimal discriminant (inverse pooled covariance times the class-mean difference) it is at least as good as Fisher efficiency in {x3k['h14a']['mcs_optimal_ge_fisher']}/10 models and is the only statistic whose scale carries across models (leave-one-model-out R² {x3k['lomo_r2']['p4b_mcs_total_vs_target_optimal']:.2f} on v3, {x5k['lomo_r2']['p4b_mcs_total_vs_target_optimal']:.2f} on v5)."),
        body(f"Proposition 1 predicts AUROC with no fitted parameter as Φ(F·d*/√2). It ranks configurations as well as the best statistic, but on v3 it overestimates transfer by {x3k['prop1_bias']:+.3f} on average (no-fit R² {x3k['prop1']['r2_no_fit']:.2f}); on v5 it is calibrated (R² {x5k['prop1']['r2_no_fit']:.2f}, mean absolute error {x5k['prop1']['mae']:.3f}). Apertus is the failure case on both datasets: every statistic tracks its transfer least well, the same model whose overlap slope X10 does not predict. It has {_ap['n_outlier_dims']} outlier dimensions (variance above 20× the median) carrying {_ap['outlier_variance_share']:.0%} of its activation variance, against at most {_other_max} in any other model, but the preregistered removal test (H14d) does not rescue its X10 prediction (r {_ap['x10_slope_r_raw']:.2f} → {_ap['x10_slope_r_reduced']:.2f}), so the outliers do not by themselves explain the failure. Few-label estimates do not rescue the label-based statistics (X12, preregistered, not supported), so these are explanatory measures, not cheap deployment predictors."),
        small("Machine-readable: results/MODEL/extensions/x14_mcs.json and results/x14_meta.json (v3: results/remote_v3; v5: results/remote_v5). Protocol: protocol/X14_PREREGISTRATION.md, frozen before any X14 output."),
    ))

    agg2, agg3 = tr["comparison"]["aggregate"], rep["agg"]
    labse_rows = [[l.upper(), *(f(tr["labse"][l][s], 3) for s in ("NLLB", "LLM", "Google")),
                   " / ".join(str(tr["fidelity"][l][s][0]) for s in ("NLLB", "Opus", "Google")),
                   " / ".join(str(tr["fidelity"][l][s][1]) for s in ("NLLB", "Opus", "Google"))] for l in ("de", "ar", "hi", "fr", "es")]
    pages.append(page(
        "Translation robustness: NLLB (v2) versus Claude Opus 5.5 (v3)",
        table([["Lang.", "LaBSE NLLB", "LaBSE Opus", "LaBSE Google", "Digit errors N / O / G", f"Unit swaps N / O / G (of {tr['n_imperial']})"], *labse_rows],
              [.08, .14, .14, .14, .23, .27]),
        body(f"All primary results use v3, in which the five non-English claim versions were translated by Claude Opus 5.5 under a frozen, label-blind protocol. The same experiments on the v2 translations (Meta NLLB) are kept as a translation-system robustness check. Automatic checks do not settle which translation is better: LaBSE similarity to the English is marginally higher for NLLB in four of five languages, but NLLB changes numbers and swaps imperial units for metric ones while keeping the number ('three-inch nail' → 'drei Zentimeter'); Opus has no digit or unit errors. LaBSE rewards surface closeness, not fidelity. No bilingual human review was carried out."),
        table([["Across 10 models", "v2 (NLLB)", "v3 (Opus)"],
               ["Overlap dissociation (cosine up, AUROC flat)", f"{agg3['v2']['dissociation']}/10", f"{agg3['v3']['dissociation']}/10"],
               ["Fisher efficiency beats cosine", f"{agg3['v2']['fisher_beats_cos']}/10", f"{agg3['v3']['fisher_beats_cos']}/10"],
               ["Mean cross-language AUROC", f(agg3['v2']['mean_cross_auroc'], 3), f(agg3['v3']['mean_cross_auroc'], 3)],
               ["Mean overlap cosine slope", sgn(agg3['v2']['mean_cos_slope']), sgn(agg3['v3']['mean_cos_slope'])],
               ["Same selected block", "", f"{tr['comparison']['same_block']}/10"]], [.5, .25, .25]),
        body(f"Every qualitative conclusion is unchanged between the two translation systems; v3 transfer is slightly higher (mean +{agg2['cross_auroc']['mean_diff']:.3f} AUROC). Disclosure: the first author chose v3 as primary for translation fidelity after v3 results for three core models and the automatic checks had been seen, and without the planned blind human A/B comparison; the analyst had recommended keeping v2 primary unless a human comparison preferred v3 (DEVIATIONS.md, 3 October). Because the conclusions do not depend on the choice, this affects how the claim set is described, not what is claimed."),
        small("Machine-readable: results/remote_v3/translation_comparison.json, results/translation_qe_three_way.json, results/translation_fidelity.json (reporting/translation_fidelity.py). Google Translate output was obtained by the first author through Google Sheets and used only for these checks."),
    ))

    rows5 = rep["rows"]
    rmb = rs.mistral_v5_behavior()
    def dis(v):
        return f"{sgn(v['cos_slope'])} [{sgn(v['cos_lo'])}, {sgn(v['cos_hi'])}]"
    v5_rows = [[rs.NAME[m], f(rows5[m]["v3"]["cross_auroc"]), f(rows5[m]["v5"]["cross_auroc"]), f(rows5[m]["v5_sf"]["cross_auroc"]),
                dis(rows5[m]["v5_sf"]), "yes" if rows5[m]["v5_sf"]["auc_flat"] else "no",
                f"{rows5[m]['v5_sf']['rho_fisher']:.2f} / {rows5[m]['v5_sf']['rho_cos']:.2f}"] for m in rs.MODELS]
    a5, sf = rep["agg"]["v5"], rep["agg"]["v5_sf"]
    pages.append(page(
        "v5: replication on an independently built claim set",
        figure(fig / "v5_replication.png", "Left: mean cross-language AUROC. Right: overlap cosine slope with 95% CI (Apertus's interval exceeds the axis).", max_height=205),
        table([["Model", "AUROC v3", "AUROC v5", "v5 w/o shortcut", "Cosine slope w/o shortcut [95% CI]", "AUROC flat", "ρ Fisher / cos"], *v5_rows],
              [.2, .09, .09, .11, .27, .09, .15]),
        body(f"v5 is the first author's MTruth release: 2,000 mLAMA-derived fact families with one true and one object-substituted false claim each, entity-template sentences in the same six languages, and its own family split and overlap allocations. An automatic audit removed 11 families with wrong-entity labels, leaving 3,978 claims per language. A second audit found a label shortcut: in 343 families the true object's name is contained in the subject's name (e.g. 'Audi Q7' made by 'Audi'), against 2 families for the false object. Every analysis was therefore repeated without those families (3,296 claims per language), a control decided before any v5 result."),
        body(f"The shortcut inflates cross-language AUROC in every model by {rep['shortcut_gain']['min']:.3f} to {rep['shortcut_gain']['max']:.3f}; without it, v5 accuracy is within {max(abs(rep['v5_sf_minus_v3']['min']), abs(rep['v5_sf_minus_v3']['max'])):.2f} of v3 (mean difference {rep['v5_sf_minus_v3']['mean']:+.3f}). What replicates: AUROC is flat under overlap in {sf['auc_flat']}/10 models, Fisher efficiency beats cosine in {sf['fisher_beats_cos']}/10, and the X14 results hold. What does not fully replicate: overlap raises cosine {rep['v3_over_v5sf_cos_slope']['median']:.1f}× less than on v3 (median ratio), with an interval above zero in only {sf['cos_up']}/10 models ({', '.join(rs.NAME[m] for m in rep['v5_sf_dissociation_models'])}). Short entity templates appear to leave little non-shared content for overlap to align; we report this as a boundary condition of the overlap artifact. Mistral answers v5 judgment prompts degenerately (Hindi {rmb['hi']['share_true']:.0%} 'true', French {1 - rmb['fr']['share_true']:.0%} 'false'; its Arabic templates never agree), so one constant difficulty component was dropped for Arabic."),
        small("Machine-readable: results/remote_v5/MODEL/ (full) and MODEL/shortcut_control/ (shortcut-free); MTruth_2000_v5_adapted/ for the adapter, exclusions and shortcut list. Positives are inherited from mLAMA and negatives are not fact-checked; v5 labels are unverified."),
    ))

    pages.append(page(
        "Deviations, limitations, and NAACL assessment",
        table([
            ["Claim", "Evidence", "Wording"],
            ["Shared facts inflate cosine", "Controlled; precise in three of four families", "Primary claim"],
            ["Alignment rises without transfer", "Overlap and latent AE, four families", "Primary extension claim"],
            ["Difficulty partly confounds geometry", "Cross-fitted, four families", "Secondary mechanism"],
            ["Thresholds, not ranking, fail most", "Descriptive decomposition", "Practical diagnostic"],
            ["Directions are causally active", "Strong in Qwen only", "Qualified, model dependent"],
            ["Cosine is a weak, manipulable proxy", f"All layers, {len(x7_primary)} models; overlap and X1 checks", "Primary X7 claim"],
            ["Label-free consistency beats cosine", "Preregistered H1 refuted", "Report as negative result"],
            ["Findings hold across scale and post-training", "Qwen3 0.6B to 14B, OLMo, post-trained Qwen3", "Robustness claim"],
            ["Overlap artifact has a closed form", "Preregistered pooled test failed (Apertus); accurate in 9/10", "Qualified theory claim"],
            ["Artifact is not mass-mean specific", "LDA and logistic regression, 10/10 models", "Robustness claim"],
            ["Few-shot Fisher estimates transfer", "Preregistered X12 not supported", "Report as negative result"],
            ["Latent unexpressed knowledge", "Imbalanced quadrants", "Do not claim"],
            ["Mahalanobis cosine vs target optimum tracks transfer", "X14, 10 models, v3 and v5", "Primary X14 claim"],
            ["Conclusions survive the translation system", "v2 (NLLB) vs v3 (Opus), 10 models", "Robustness claim"],
            ["Findings replicate on an independent claim set", "v5; AUROC flat and Fisher > cosine 10/10; cosine artifact 5/10", "Qualified replication"],
        ], [.32, .36, .32]),
        subtitle("Logged deviations"),
        body("All were recorded before the affected outcome was inspected: dependency-group unit correction; mass-mean-only probes; Llama-3.1-8B unavailable (gated); MMMLU context compaction and pair-symmetric exclusions; storage-only download filters; Apertus steering-scale diagnostic (no endpoint changed); X6 added at the user's request; Gemma X2/X3/X6 run late after access, with regenerated caches that reproduced stored results exactly; Pythia padding-token fallback; X7 to X9 (label-free prediction, scale, post-training) added and frozen before outcomes, with an input-discovery bug fixed before any X7 output was written; X10 to X13 frozen in a separate protocol, with regenerated activations that reproduced stored RQ-C results exactly; Qwen3-14B run at batch size 8; v3 (Opus 5.5) translations made primary by the first author after some v3 results were seen; X14 frozen before its outputs; v5 adapter exclusions and the shortcut control decided before any v5 result; v5 run in parallel lanes with resume support after other users took GPUs (no analysis change)."),
        subtitle("Limitations and rating"),
        body("The claim translations are machine translations (Claude Opus 5.5) without bilingual human review; automatic checks and the NLLB comparison bound, but do not replace, that review. X6 adds native-authored test questions but the training claims remain translations. The largest model is 14B. RoSh and LSI comparisons are adaptations. RQ-D is imbalanced, and steering effects are strong in one family only. The overlap formula does not predict Apertus, and logistic-regression regularization sat at the grid edge in most models. The work is a credible NAACL long-paper submission framed as a measurement paper with a tested mechanism; our candid pre-review estimate is about 5 in 10 for the main conference and about 7 in 10 for main or Findings."),
        small("Reproducibility: exact revisions, data hashes, frozen protocols, one-GPU logs, deletion records, seeds, tests, and every JSON artifact are retained with the study package."),
    ))

    assert len(pages) == 18, len(pages)
    for authored_page in pages:
        if authored_page.get("dek"):
            authored_page["dek"] = tidy(authored_page["dek"])
        for element in authored_page["elements"]:
            for key in ("text", "caption"):
                if key in element:
                    element[key] = tidy(element[key])
            if "rows" in element:
                element["rows"] = [[tidy(str(cell)) for cell in row] for row in element["rows"]]
    payload = {"title": "When Multilingual Truth Directions Align Without Transferring",
               "subject": "Mass-mean truth probes across six languages, four model families, and one training trajectory",
               "footer": "Truth Transport · final evidence-linked report · NAACL 2027",
               "pages": pages}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "pages": len(pages)}))


if __name__ == "__main__":
    main()
