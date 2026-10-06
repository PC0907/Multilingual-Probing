#!/usr/bin/env python3
"""Assemble the evidence-linked fifteen-page report content manifest."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


MODELS = ["gemma-7b", "qwen3-8b-base", "apertus-8b-2509"]
DISPLAY = {"gemma-7b": "Gemma-7B", "qwen3-8b-base": "Qwen3-8B-Base", "apertus-8b-2509": "Apertus-8B"}
LANGUAGES = ["English", "German", "Arabic", "Hindi", "French", "Spanish"]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def body(text: str) -> dict:
    return {"type": "body", "text": text}


def small(text: str) -> dict:
    return {"type": "small", "text": text}


def subtitle(text: str) -> dict:
    return {"type": "subtitle", "text": text}


def reference(text: str) -> dict:
    return {"type": "reference", "text": text}


def callout(text: str) -> dict:
    return {"type": "callout", "text": text}


def table(rows: list[list[str]], widths: list[float]) -> dict:
    return {"type": "table", "rows": rows, "widths": widths}


def figure(path: Path, caption: str, max_height: int = 340, width_fraction: float = 1.0) -> dict:
    return {"type": "figure", "path": str(path.resolve()), "caption": caption,
            "max_height": max_height, "width_fraction": width_fraction}


def page(title: str, *elements, dek: str | None = None, eyebrow: str | None = None) -> dict:
    payload = {"title": title, "elements": list(elements)}
    if dek:
        payload["dek"] = dek
    if eyebrow:
        payload["eyebrow"] = eyebrow
    return payload


def f(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def mean_cross(cells: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in cells if row["source"] != row["target"]]))


def rq_a_stats(result_root: Path, model: str) -> dict:
    inference = read(result_root / model / "rq_a" / "inference.json")
    pooled = inference["pooled_hierarchical_bootstrap"]
    return {
        "block": inference["selected_block_number"],
        "cos_slope": pooled["cosine_slope"],
        "auc_slope": pooled["auroc_slope"],
        "zero_cos": pooled["zero_cosine"],
        "full_cos": pooled["full_cosine"],
        "zero_auc": pooled["zero_auroc"],
        "full_auc": pooled["full_auroc"],
        "cos_sig": sum(row["p_holm"] < .05 for row in inference["cosine_full_vs_zero_tests_holm_15"]),
        "auc_sig": sum(row["p_holm"] < .05 for row in inference["transfer_full_vs_zero_tests_holm_30"]),
    }


def rq_b_stats(result_root: Path, model: str) -> dict:
    payload = read(result_root / model / "rq_b.json")
    pairs = [row for row in payload["truth_pair_results"] if row["source"] != row["target"]]
    difficulty_auc = [row["auroc"] for row in payload["difficulty_transfer"] if row["source"] != row["target"]]
    return {
        "cos_original": float(np.mean([row["truth_cosine_original"] for row in pairs])),
        "cos_residual": float(np.mean([row["truth_cosine_residual"] for row in pairs])),
        "auc_original": float(np.mean([row["original_transfer"]["auroc"] for row in pairs])),
        "auc_residual": float(np.mean([row["residual_transfer"]["auroc"] for row in pairs])),
        "ba_original": float(np.mean([row["original_transfer"]["balanced_accuracy"] for row in pairs])),
        "ba_residual": float(np.mean([row["residual_transfer"]["balanced_accuracy"] for row in pairs])),
        "difficulty_cos": float(np.mean([row["cosine"] for row in payload["difficulty_direction_cosines"]])),
        "difficulty_auc": float(np.mean(difficulty_auc)),
        "zero_residual_cos": payload["overlap_summary"][0]["mean_residual_cosine"],
        "zero_residual_auc": payload["overlap_summary"][0]["mean_residual_transfer_auroc"],
    }


def rq_c_stats(result_root: Path, model: str) -> dict:
    cells = read(result_root / model / "rq_c.json")["cells"]
    cross = [row for row in cells if row["source"] != row["target"]]
    prior30 = [row["prior_shift"]["0.3"] for row in cross]
    prior70 = [row["prior_shift"]["0.7"] for row in cross]
    return {
        "cos": float(np.mean([row["direction_cosine"] for row in cross])),
        "auc": float(np.mean([row["auroc"] for row in cross])),
        "ba": float(np.mean([row["balanced_accuracy"] for row in cross])),
        "dprime": float(np.mean([row["standardized_separation"] for row in cross])),
        "recenter_gain": float(np.mean([row["unlabeled_adaptation"]["balanced_accuracy_change"] for row in cross])),
        "prior30_gain": float(np.mean([row["adapted_balanced_accuracy"]["mean"] - row["zero_shot_balanced_accuracy"]["mean"] for row in prior30])),
        "prior70_gain": float(np.mean([row["adapted_balanced_accuracy"]["mean"] - row["zero_shot_balanced_accuracy"]["mean"] for row in prior70])),
    }


def rq_d_stats(result_root: Path, model: str) -> dict:
    payload = read(result_root / model / "rq_d.json")
    result = {}
    for name in ["K/K", "K/U", "U/K", "U/U"]:
        rows = [q for pair in payload["pairs"] for q in pair["quadrants"] if q["quadrant"] == name]
        valid = [q["zero_overlap_transfer"]["auroc"] for q in rows if q["zero_overlap_transfer"]["auroc"] is not None]
        result[name] = {
            "mean_auc": float(np.mean(valid)) if valid else None,
            "valid": len(valid),
            "pair_count": sum(q["n_groups"] for q in rows),
            "true": sum(q["n_true"] for q in rows),
            "false": sum(q["n_false"] for q in rows),
        }
    return result


def rq_d_margin_stats(result_root: Path, model: str) -> dict:
    payload = read(result_root / model / "rq_d_mean_margin.json")
    result = {"known_rate_test": payload["known_rate_test"]}
    for name in ["K/K", "K/U", "U/K", "U/U"]:
        rows = [q for pair in payload["pairs"] for q in pair["quadrants"] if q["quadrant"] == name]
        valid = [q["auroc"] for q in rows if q["auroc"] is not None]
        result[name] = {"mean_auc": float(np.mean(valid)) if valid else None, "valid": len(valid)}
    return result


def rq_e_stats(result_root: Path, model: str) -> dict:
    payload = read(result_root / model / "steering_summary.json")
    named = payload["named_direction_pair_distribution"]["margin"]["all"]
    random = payload["random_direction_pair_draw_distribution"]["margin"]["all"]
    neutral = read(result_root / model / "steering_neutral_flores" / "summary.json")
    neutral_zero = [row["mean_delta_nll"] for row in neutral if row["direction"] == "zero_overlap_source"]
    neutral_native = [row["mean_delta_nll"] for row in neutral if row["direction"] == "target_native"]
    sensitivities = {}
    for name, filename in {
        "layer_minus_2": "steering_sensitivity_layer_minus_2_summary.json",
        "layer_plus_2": "steering_sensitivity_layer_plus_2_summary.json",
        "all_tokens": "steering_sensitivity_all_statement_tokens_summary.json",
    }.items():
        sensitivity = read(result_root / model / filename)
        sensitivities[name] = sensitivity["named_direction_pair_distribution"]["margin"]["all"]["zero_overlap_source"]["mean"]
    return {
        "zero": named["zero_overlap_source"],
        "full": named["full_overlap_source"],
        "native": named["target_native"],
        "difficulty": named["difficulty"],
        "language": named["language_identity"],
        "random": random,
        "neutral_zero": float(np.mean(neutral_zero)),
        "neutral_native": float(np.mean(neutral_native)),
        **sensitivities,
    }


def tidy(text: str) -> str:
    parts = re.split(r"(<[^>]*>)", text)
    for index, part in enumerate(parts):
        if part.startswith("<"):
            continue
        part = part.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")
        parts[index] = part
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--figure-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    required = [
        args.results_root / model / filename
        for model in MODELS
        for filename in ["rq_b.json", "rq_c.json", "rq_d.json", "steering_summary.json", "all_layers.json", "translation_sensitivity.json"]
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing final artifacts:\n" + "\n".join(missing))

    a = {model: rq_a_stats(args.results_root, model) for model in MODELS}
    b = {model: rq_b_stats(args.results_root, model) for model in MODELS}
    c = {model: rq_c_stats(args.results_root, model) for model in MODELS}
    d = {model: rq_d_stats(args.results_root, model) for model in MODELS}
    dm = {model: rq_d_margin_stats(args.results_root, model) for model in MODELS}
    e = {model: rq_e_stats(args.results_root, model) for model in MODELS}

    pages = []
    pages.append(page(
        "Matched facts, misleading angles",
        callout("Main result: shared training-fact identity raises cross-language mass-mean cosine, but it does not improve held-out transfer. Direction alignment and predictive transfer are therefore different measurements."),
        body("We tested one balanced set of 2,000 English claims and parallel German, Arabic, Hindi, French, and Spanish translations in Gemma-7B, Qwen3-8B-Base, and Apertus-8B. Every primary probe is a mass-mean direction. The design fixes each fit at 400 dependency groups and changes only the fraction of matched fact identities: 0%, 25%, 50%, 75%, or 100%. Fifty allocations per condition separate content overlap from language alignment."),
        table([
            ["Model", "Frozen block", "Cosine slope", "Transfer AUROC slope", "Zero-overlap steering"],
            *[[DISPLAY[m], str(a[m]["block"]), f(a[m]["cos_slope"]["mean"]), f(a[m]["auc_slope"]["mean"], 4), f(e[m]["zero"]["mean"], 4)] for m in MODELS],
        ], [.23, .14, .19, .22, .22]),
        subtitle("What the completed study supports"),
        body("Gemma and Qwen show precise positive overlap slopes for cosine; Apertus is noisy in the raw RQ-A fit but shows a monotone residualized overlap curve. Across all three models, transfer AUROC stays flat as overlap changes. Cross-fitted removal of behavioral difficulty reduces both geometry and transfer, yet a nontrivial residual remains. Source-to-target score ranking transfers better than a frozen source threshold, and unlabeled target recentering improves balanced accuracy without changing AUROC."),
        body("Causal steering is strongly model dependent. Qwen's zero-overlap direction produces a large, clean judgment-margin shift; Gemma's effect is small; Apertus supplies the third-family test. Behavioral knowledge quadrants are descriptive because conditioning on model correctness changes label balance."),
        subtitle("Practical implication"),
        body("A multilingual probing paper should no longer present a high cosine from parallel training data as evidence that two languages use the same latent mechanism. At minimum, it should report a disjoint-fact estimate, target AUROC, and source-threshold transport. Our experiment turns that recommendation into a controlled measurement protocol and shows that the correction changes the scientific conclusion."),
        small("Status at report generation: RQ-A through RQ-E complete for three models. The official Llama-3.1-8B checkpoint was inaccessible under its manual gate. RQ-F emergence across intermediate checkpoints remains an optional follow-up."),
        dek="A controlled multilingual truth-probing study across six languages and three open 7B-8B model families",
        eyebrow="FINAL 15-PAGE TECHNICAL REPORT / 1 OCTOBER 2026",
    ))

    pages.append(page(
        "Research questions and novelty boundary",
        table([
            ["RQ", "Question", "Status and answer"],
            ["A", "How much alignment is caused by shared factual examples?", "Answered: overlap raises cosine in Gemma/Qwen, while transfer is flat; Apertus raw slope is uncertain."],
            ["B", "Does truth alignment survive removal of item difficulty?", "Answered: yes, but both cosine and AUROC fall; the amount is model dependent."],
            ["C", "Why can aligned directions transfer differently?", "Answered: ranking, score offset, scale, and threshold transport are separable."],
            ["D", "Can a direction decode truth when behavior fails in a target language?", "Mixed: some K/U cells are decodable, but selection and imbalance prevent a strong knowledge claim."],
            ["E", "Is the zero-overlap direction causally used?", "Answered: strong for Qwen, weak for Gemma, evaluated for Apertus with random and neutral-text controls."],
            ["F", "When does cross-lingual truth alignment emerge?", "Optional and not run; checkpoint storage and reproducibility evidence were prioritized."],
        ], [.08, .42, .50]),
        subtitle("Closest prior work"),
        body("Li et al. already compare multilingual probing accuracy, layer trends, and cross-language vector similarity in Gemma and Qwen. Marks and Tegmark establish mass-mean probing and causal truth interventions. RoSh, released three days before this audit, studies 1,500 parallel claims in eight languages and applies per-language rotations plus shifts to answer-token states. LSI is another direct cross-lingual latent-intervention baseline. These papers rule out a broad claim of first multilingual truth probing or first multilingual intervention."),
        callout("Defensible novelty: an exact fixed-n fact-identity overlap dose response, combined with train-only cross-fitted difficulty removal and causal zero-overlap mass-mean steering."),
        body("The paper should be framed as a measurement and decomposition study. RQ-C and RQ-D supply mechanisms and diagnostics; RQ-E validates whether the allocation-independent component can influence behavior. The central claim is about what cosine measures under matched-content training, not about a universal language-free truth axis."),
        subtitle("What is still open"),
        body("The current literature has already established multilingual truth probing, cross-language similarity, difficulty geometry, query-language effects, and causal latent interventions. What remains open is whether those conclusions change when direction estimators share no fact identities and when behavioral difficulty is estimated out of fold. That narrow gap is the paper's strongest defensible contribution."),
        reference('Sources: <link href="https://aclanthology.org/2025.xllm-1.7/">Li et al. (XLLM 2025)</link>; <link href="https://arxiv.org/abs/2310.06824">Marks and Tegmark (COLM 2024)</link>; <link href="https://arxiv.org/abs/2609.34678">Ahmad et al., RoSh (2026)</link>; <link href="https://arxiv.org/abs/2608.28860">Ghorbanpour et al., LSI (2026)</link>.'),
    ))

    pages.append(page(
        "Dataset v2: what 1,972 dependency groups mean",
        body("The source CSV contains 2,000 statement rows, exactly 1,000 true and 1,000 false. Translation and paraphrase collisions mean these rows are not always statistically independent. We normalize claims across all six language files and connect rows that express the same proposition or an explicitly linked variant. This produces 1,972 dependency groups. Splits, allocations, and bootstraps operate on groups so near-duplicate claims cannot leak across partitions or receive excess weight."),
        table([
            ["Property", "Value", "Consequence"],
            ["Languages", ", ".join(LANGUAGES), "One aligned surface per claim ID and language"],
            ["Rows / groups", "2,000 / 1,972", "Group-resampled uncertainty; group-disjoint splits"],
            ["Labels", "1,000 true / 1,000 false", "Balanced source data and balanced 400-group fits"],
            ["Split", "1,200 / 400 / 400 rows", "Train / validation / held-out test"],
            ["Pure train groups", "1,170", "Two mixed-label groups excluded from probe fitting"],
            ["Documented repairs", "47 sentences, 21 IDs", "Definite v1 meaning changes repaired and logged"],
            ["Human review", "Pending", "Candidate parallel dataset; no validated-benchmark claim"],
        ], [.25, .28, .47]),
        subtitle("Translation controls"),
        body("The 47 repairs correct concrete semantic failures such as changed units, animal names, and predicates. Corrected items remain marked as translation-risk cases, preventing the clean-data sensitivity from benefiting silently from post hoc repairs. A 1,000-row bilingual review pack exists, but no completed independent German-Arabic-Hindi-French-Spanish annotation is available. The report therefore uses 'parallel translated claims' throughout."),
        body("The dependency groups also absorb duplicate or logically linked rows. For example, a positive statement and its explicit negation must travel together even though their labels differ. Two such mixed-label groups occur in training. They are valid split and bootstrap units but cannot be averaged into one labeled vector, so they are excluded from confirmatory probe fits. Same-label paraphrases are averaged before fitting, giving each independent proposition one vote."),
        subtitle("Allocation matching"),
        body("The supplied data have no documented topic field. Primary allocations match truth label, pair-normalized length quartile, negation, number presence, and automatic translation-risk status. Depending on the language pair, 5-12 sparse groups are excluded to make exact zero-overlap allocations feasible. We do not treat inferred clusters as observed topics."),
        small("Version: 2.0-candidate. Manifest and per-language SHA-256 hashes are retained with the dataset; partition leakage check: zero dependency groups cross splits."),
    ))

    pages.append(page(
        "Experimental design and estimands",
        body("For language l, the mass-mean direction is the mean hidden state for true claims minus the mean hidden state for false claims. The representation is the decoder-block residual stream at the final statement token inside one frozen, translated true/false judgment prompt. Each model uses one block selected only by the mean of six within-language validation AUROCs: Gemma block 18 of 28, Qwen block 21 of 36, and Apertus block 15 of 32."),
        table([
            ["Component", "Frozen design"],
            ["Probe", "Mass mean only; no logistic hidden-state probe"],
            ["Fit size", "400 pure-label dependency groups: 200 true, 200 false"],
            ["Overlap", "0/100/200/300/400 shared groups; nested within 50 allocations"],
            ["Evaluation", "Fixed 400-row held-out test, disjoint from all fitted groups"],
            ["Uncertainty", "Group bootstrap plus allocation resampling; paired sign-flip endpoints with Holm correction"],
            ["Difficulty removal", "Five train-only folds; out-of-fold residualization; final train direction transforms validation/test"],
            ["Score transport", "AUROC, source-threshold balanced accuracy, d-prime, offset, scale, target diagnostic threshold"],
            ["Steering", "Post-block residual addition at final statement token; alpha in {-4,-2,-1,0,1,2,4}"],
        ], [.28, .72]),
        subtitle("Leakage barriers"),
        body("Test states never choose a layer, threshold, alpha, residualization direction, or allocation. Target-optimal thresholds are diagnostic and use validation labels only. Unlabeled recentering uses target validation moments with labels hidden from the transformation. Prior-shift checks separate calibration and evaluation samples. Translations are resampled together by dependency group."),
        subtitle("Core quantities"),
        body("For a source probe, scores are centered so its own class midpoint is zero. Cosine compares two direction vectors after fitting. AUROC is invariant to a positive affine transformation and therefore measures ranking only. Standardized separation divides the target true-false mean gap by pooled target-score SD. Global offset is the target midpoint relative to the source threshold. Recentered balanced accuracy estimates how much threshold error can be repaired from unlabeled target moments."),
        subtitle("Interpretation rule"),
        callout("Cosine measures angular agreement between fitted estimators. AUROC measures target ranking. Balanced accuracy measures whether the source threshold transports. None alone establishes a universal representation."),
    ))

    pages.append(page(
        "RQ-A: shared facts inflate cosine, not transfer",
        figure(args.figure_dir / "rq_a_overlap.png", "Figure 1. Each line averages all ordered language pairs and 50 fixed-size allocations. Training overlap changes; the held-out test is fixed.", 325),
        table([
            ["Model", "Cosine slope [95%]", "Zero -> full cosine", "AUROC slope [95%]", "Holm sig."],
            *[[DISPLAY[m],
               f'{f(a[m]["cos_slope"]["mean"],4)} [{f(a[m]["cos_slope"]["p025"],4)}, {f(a[m]["cos_slope"]["p975"],4)}]',
               f'{f(a[m]["zero_cos"]["mean"])} -> {f(a[m]["full_cos"]["mean"])}',
               f'{f(a[m]["auc_slope"]["mean"],4)} [{f(a[m]["auc_slope"]["p025"],4)}, {f(a[m]["auc_slope"]["p975"],4)}]',
               f'{a[m]["cos_sig"]}/15 cosine; {a[m]["auc_sig"]}/30 AUROC'] for m in MODELS],
        ], [.16, .26, .19, .25, .14]),
        body("Gemma's pooled cosine slope is 0.0542 and Qwen's is 0.0383 per full-overlap change; both intervals exclude zero. Apertus' raw slope is positive but imprecise. No model has a Holm-significant transfer endpoint among the 30 ordered pairs. The clean translation-risk exclusion repeats the Gemma and Qwen pattern, so documented risky translations are not driving the headline result."),
        callout("Answer to RQ-A: multilingual direction similarity partly reflects shared estimation content. Higher cosine under matched facts does not imply better zero-shot transfer on new facts."),
        small("Reliability-adjusted cosine is secondary because its disattenuation denominator can be nonpositive or unstable. Apertus has 50 undefined adjusted cells across the 150 pair-by-overlap summaries; raw cosine remains primary."),
    ))

    pages.append(page(
        "RQ-B: difficulty explains part, not all, of alignment",
        figure(args.figure_dir / "rq_b_difficulty.png", "Figure 2. Off-diagonal means before and after train-only, cross-fitted removal of an equal-language behavioral-difficulty direction.", 310),
        table([
            ["Model", "Cosine original -> residual", "AUROC original -> residual", "Zero-overlap residual", "Difficulty transfer"],
            *[[DISPLAY[m], f'{f(b[m]["cos_original"])} -> {f(b[m]["cos_residual"])}',
               f'{f(b[m]["auc_original"])} -> {f(b[m]["auc_residual"])}',
               f'cos {f(b[m]["zero_residual_cos"])}; AUC {f(b[m]["zero_residual_auc"])}',
               f'cos {f(b[m]["difficulty_cos"])}; AUC {f(b[m]["difficulty_auc"])}'] for m in MODELS],
        ], [.16, .22, .22, .22, .18]),
        body("Difficulty combines correctness, aligned answer margin, prediction consistency, negative entropy, and negative statement surprisal, standardized within model and language. Easy-minus-hard contrasts are label balanced. Gemma loses 0.125 cosine and 0.051 AUROC after removal. Qwen loses less geometry and about 0.018 AUROC. Apertus loses 0.038 cosine and 0.061 AUROC. A residual zero-overlap component remains in every model."),
        body("Three prespecified sensitivities remove a per-language single direction, a pooled five-component subspace, or a per-language five-component subspace. They do not recover one invariant answer: Qwen's per-language five-dimensional removal is much stronger than its pooled removal, while Apertus' pooled behavioral direction is close to orthogonal across languages. This is evidence that 'difficulty' itself has language- and model-specific geometry."),
        callout("Answer to RQ-B: truth alignment survives difficulty removal, but difficulty is a material confound. The surviving component is strongest in Qwen and weakest in Apertus."),
    ))

    pages.append(page(
        "RQ-C: ranking transfers better than thresholds",
        figure(args.figure_dir / "rq_c_transport_summary.png", "Figure 3. Off-diagonal means over all 30 source-target directions. Recenter gain is balanced-accuracy change from label-blind target validation moments.", 285),
        table([
            ["Model", "Cosine", "AUROC", "Source-threshold BA", "d-prime", "Recenter gain"],
            *[[DISPLAY[m], f(c[m]["cos"]), f(c[m]["auc"]), f(c[m]["ba"]), f(c[m]["dprime"]), f(c[m]["recenter_gain"],4)] for m in MODELS],
        ], [.20, .14, .14, .20, .14, .18]),
        body("Qwen combines the highest cross-language cosine (0.854) with the best mean AUROC (0.791), but its mean frozen-threshold balanced accuracy is lower (0.697). Gemma has lower cosine yet similar standardized separation. Apertus has large raw score offsets and weaker transfer; its pairwise direction cosine ranges from 0.043 to 0.993 while AUROC ranges from 0.546 to 0.804. Across models, unlabeled affine recentering changes threshold decisions while leaving AUROC invariant, as it should."),
        body("The imposed 30% and 70% true-prior checks use separate calibration and evaluation samples. Recenter gains vary with the target prior, confirming that threshold transport is partly a location problem. The source scalar logistic fit appears only as a probability calibrator; it is not a second hidden-state probe."),
        callout("Answer to RQ-C: cosine, target ranking, standardized separation, score offset, and threshold performance are distinct. Report the complete transport profile instead of using cosine as a proxy for transfer."),
        small("Pages 8-10 contain all 36 source-target cells for each model. Rows are probe-training languages; columns are evaluation languages. Diagonals are native references."),
    ))

    for page_number, model in zip([8, 9, 10], MODELS):
        pages.append(page(
            f"RQ-C matrix {page_number - 7}: {DISPLAY[model]}",
            figure(args.figure_dir / f"rq_c_matrix_{model}.png", "Figure 4. Complete 6 x 6 matrices. Offset is normalized by the target-score pooled SD. Recenter gain uses label-blind target-validation moments.", 455),
            body(f"{DISPLAY[model]} uses frozen block {a[model]['block']}. Off-diagonal mean cosine is {f(c[model]['cos'])}, AUROC {f(c[model]['auc'])}, source-threshold balanced accuracy {f(c[model]['ba'])}, and standardized separation {f(c[model]['dprime'])}. Mean recentering gain is {f(c[model]['recenter_gain'],4)}. The visual separation between AUROC and balanced accuracy identifies target-language shifts that preserve ordering but move the decision boundary."),
            small("All displayed cells use the full 1,170 pure-group training pool for their source-language direction and the fixed 400-row target test. The overlap experiment on page 5 uses equal 400-group fits and therefore answers a different estimand."),
        ))

    pages.append(page(
        "RQ-D: behavioral knowledge quadrants are informative but fragile",
        figure(args.figure_dir / "rq_d_quadrants.png", "Figure 5. Mean zero-overlap AUROC within behaviorally defined quadrants. Parentheses show the number of ordered-pair cells containing both labels.", 305),
        table([
            ["Model", "K/K", "K/U", "U/K", "U/U"],
            *[[DISPLAY[m]] + [f'{f(d[m][q]["mean_auc"])} ({d[m][q]["valid"]})' if d[m][q]["mean_auc"] is not None else "undefined" for q in ["K/K", "K/U", "U/K", "U/U"]] for m in MODELS],
        ], [.22, .195, .195, .195, .195]),
        body("A fact-language case is behaviorally known when at least two of three frozen prompt templates are correct and all three predictions have the same sign. K/U means the source language passes this criterion and the target language does not. Gemma's behavior is nearly label deterministic, leaving only six valid K/U cells. Qwen produces all 30 two-class K/U cells but their mean AUROC is 0.612; Apertus has 17 valid K/U cells with mean 0.784."),
        body(f"The striking Qwen U/K and K/K values, along with below-chance U/U performance, show how conditioning on correctness can restructure the label and score distributions. Changing the behavioral criterion to per-token-normalized answer margins changes K/U to {f(dm['gemma-7b']['K/U']['mean_auc'])} ({dm['gemma-7b']['K/U']['valid']} valid cells) for Gemma, {f(dm['qwen3-8b-base']['K/U']['mean_auc'])} ({dm['qwen3-8b-base']['K/U']['valid']}) for Qwen, and {f(dm['apertus-8b-2509']['K/U']['mean_auc'])} ({dm['apertus-8b-2509']['K/U']['valid']}) for Apertus. The conclusion is sensitive to answer-token normalization."),
        callout("Answer to RQ-D: some target-language failures retain decodable truth information, especially in Apertus K/U cells. The evidence does not justify saying the model 'knows but cannot express' without broader elicitation and balanced quadrant construction."),
    ))

    pages.append(page(
        "RQ-E: causal steering is real, but highly model dependent",
        figure(args.figure_dir / "rq_e_steering.png", "Figure 6. Left: judgment-margin slope per alpha for named directions, plus random-direction 95% ranges. Right: mean neutral FLORES delta NLL at alpha +/-4 on a symmetric-log scale.", 315),
        table([
            ["Model", "Zero-overlap slope", "Random 95%", "Layer -2 / +2", "All tokens", "Neutral delta NLL"],
            *[[DISPLAY[m], f(e[m]["zero"]["mean"],4),
               f'[{f(e[m]["random"]["p025"],4)}, {f(e[m]["random"]["p975"],4)}]',
               f'{f(e[m]["layer_minus_2"],4)} / {f(e[m]["layer_plus_2"],4)}',
               f(e[m]["all_tokens"],4), f(e[m]["neutral_zero"],3)] for m in MODELS],
        ], [.16, .18, .20, .18, .13, .15]),
        body("The intervention adds a unit-normalized, target-scale-matched direction after the frozen decoder block at the final statement token during prefill. Qwen's zero-overlap slope is 0.529, far outside its random-direction range, and remains strong at the layer-minus-two and all-statement-token sensitivities. Gemma's primary slope is only 0.0165 and overlaps zero across its six language-pair aggregates, although layer-minus-two is larger. Apertus has a small reversed-sign mean slope (-0.0136), near the edge of its random-direction distribution [-0.0126, 0.0187], with substantial pair heterogeneity."),
        body("Token scope magnifies the architectural split. All-statement-token zero-overlap steering is +0.6958 for Qwen, +0.0263 for Gemma, and -0.1075 for Apertus. A positive decoding direction can therefore oppose the downstream behavioral readout in Apertus. This is consistent with a model-specific readout Jacobian, not one universal intervention sign."),
        body("Neutral-text costs matter: a direction can move the answer margin simply by disrupting language modeling. Mean zero-overlap delta NLL at alpha +/-4 is 0.026 for Gemma and 0.042 for Qwen, smaller than their target-native costs. Apertus is catastrophic at the same nominal scale: mean delta NLL is 20.016, equivalent to roughly 8.7 orders of magnitude in perplexity. Its reversed causal shift is therefore not a useful intervention at alpha 4. Direction specificity is further checked against difficulty, language-identity, and 20 random directions per pair."),
        callout("Answer to RQ-E: the zero-overlap mass-mean component is causally coupled to Qwen's judgment behavior. Gemma does not show a precise primary-layer effect, and Apertus shifts mostly in the opposite direction. Causal use and sign are not architecture universal; the best intervention layer may precede the best observational probe layer."),
    ))

    pages.append(page(
        "Layer, null, and translation-risk controls",
        figure(args.figure_dir / "layer_sensitivity.png", "Figure 7. Native AUROC, cross-language AUROC, and direction cosine over every decoder block. Dashed lines mark the validation-selected block.", 285),
        table([
            ["Control", "What was tested", "Result"],
            ["Label permutation", "Within-language, stratum-preserving random labels", "Observed directions separate from the null"],
            ["Covariance-matched null", "Random-label probes under activation covariance", "Random cross-language geometry is much smaller"],
            ["Fact-identity permutation", "Break cross-language pairing while preserving strata", "Records unavoidable fixed points and removes correspondence"],
            ["All layers", "Native AUROC, transfer AUROC, cosine at every block", "Selected middle blocks outperform final blocks"],
            ["Translation-risk exclusion", "Reallocate after removing any flagged group", "Gemma/Qwen overlap pattern persists"],
            ["Prompt/token sensitivity", "Frozen translated prompts and full answer likelihood", "Per-token margin retained as RQ-D sensitivity"],
        ], [.22, .39, .39]),
        body("Final-block representations are not adequate substitutes. Gemma falls from about 0.756 native AUROC at block 18 to 0.610 at block 28; its cross cosine falls from 0.707 to 0.162. Qwen falls from about 0.785 native AUROC at block 21 to 0.741 at block 36, with cosine falling from 0.854 to 0.311. Apertus falls from 0.689 native AUROC at block 15 to 0.595 at block 32, while cosine falls from 0.530 to 0.392. The selected-layer rule therefore prevents a large, predictable measurement error."),
        body("Automatic translation-risk removal is deliberately conservative and does not prove semantic equivalence. It shows only that the main overlap pattern is not concentrated in known or automatically flagged items. Independent bilingual review remains the largest data-quality gap."),
    ))

    pages.append(page(
        "Reproducibility, deviations, and limits",
        table([
            ["Item", "Evidence retained"],
            ["Models", "Pinned Qwen 49e3418f and Apertus 3162c996 revisions; Gemma snapshot ff6768d9"],
            ["Execution", "One model on GPU 0 at a time; checkpoints deleted only after artifact verification"],
            ["Data", "Versioned v2 manifest, 6 language hashes, group map, split, corrections, review pack"],
            ["Representations", "All-layer activation caches, prompt hashes, tokenizer/config hashes, final-token indices"],
            ["Analysis", "50 allocations x 5 overlap levels x 15 language pairs; group-aware inference and nulls"],
            ["Behavior", "Three templates, full-sequence answer likelihood, summed and token-normalized margins"],
            ["Causal tests", "Seven alphas, 20 random directions per pair, adjacent layers, all-token and neutral-text controls"],
            ["Tests", "Seven pipeline unit tests passed before model execution; JSON outputs replay without checkpoints"],
        ], [.24, .76]),
        subtitle("Registered deviations"),
        body("Llama-3.1-8B was planned but the official checkpoint is manually gated and the experiment host lacked access. Apertus replaces the missing third primary family; Gemma remains the direct comparison to Li et al. No topic matching is claimed because the CSV has no documented topic labels. RQ-F was optional and was deferred to preserve caches and reproducibility evidence within storage constraints."),
        subtitle("Limits on claims"),
        body("The dataset is machine translated and researcher screened, not independently validated. The claims are generic and may not span contentious, temporal, culturally localized, or compositional factuality. Three 7B-8B architectures do not define model-universal behavior. Behavioral known/unknown labels depend on three prompt templates. Steering changes a constrained true/false judgment margin and does not establish general factual generation improvement."),
        subtitle("Execution and failure record"),
        body("The complete pipeline ran serially on GPU 0. A checkpoint directory was deleted only after cache completeness, tokenizer/config hashes, concise result artifacts, and reload-independent score replay were verified. The initial model-download helper failed before downloading Apertus because it queried free space before creating the parent directory; the parent-creation order was fixed and the run resumed without changing scientific settings. Seven tests pass after the fix."),
        callout("The results are strong enough for a focused paper, but the translation review and direct RoSh/LSI baseline are the two highest-priority additions before submission."),
    ))

    pages.append(page(
        "Recommended NAACL paper and next decisive experiments",
        callout("Recommended title: Matched Facts, Misleading Angles: Decomposing Multilingual Truth-Direction Alignment"),
        body("Lead with one claim: cross-language direction cosine is inflated by matched training content and should not be interpreted as transfer. Establish that claim with the fixed-n overlap intervention, then show how much survives train-only difficulty removal. Use score transport and behavioral quadrants to explain failure modes, and use zero-overlap steering as causal validation rather than as the paper's sole novelty."),
        table([
            ["Priority", "Experiment", "Decision value"],
            ["1", "Complete independent bilingual review of a stratified 1,000-row pack", "Converts a candidate parallel dataset into defensible multilingual evidence"],
            ["2", "Reproduce RoSh and LSI on the same v2 split and three models", "Establishes whether controlled zero-overlap MM is competitive with stronger interventions"],
            ["3", "Obtain gated Llama-3.1-8B access or add an equally established open base model", "Raises architectural breadth and avoids a planned-model gap"],
            ["4", "Run native-authored fact sets with documented human review", "Separates translationese from language-specific knowledge"],
            ["5", "Run Apertus intermediate checkpoints for RQ-F", "Tests when overlap-sensitive geometry and causal use emerge"],
            ["6", "Evaluate free-form generation and calibrated abstention", "Connects binary judgment steering to practical factuality"],
        ], [.10, .48, .42]),
        subtitle("Acceptance assessment"),
        body("Current readiness: about 7/10 for a NAACL main-conference submission. The controlled RQ-A/RQ-B design is novel and the three-model heterogeneity is scientifically valuable. The score-transport and causal sections provide depth. Acceptance risk remains material because RoSh and LSI are recent, translation review is incomplete, and Llama is missing. Human validation plus direct intervention baselines could move the work into a stronger 8/10 range; no numerical rating can predict reviewer outcomes."),
        subtitle("Selected references"),
        reference('Marks and Tegmark. <link href="https://arxiv.org/abs/2310.06824">The Geometry of Truth</link>. COLM 2024.  |  Burger et al. <link href="https://arxiv.org/abs/2407.12831">Truth is Universal</link>. NeurIPS 2024.'),
        reference('Li et al. <link href="https://aclanthology.org/2025.xllm-1.7/">Exploring Multilingual Probing in Large Language Models</link>. XLLM 2025.  |  Civelli et al. <link href="https://aclanthology.org/2026.acl-short.66/">A Shared Geometry of Difficulty in Multilingual Language Models</link>. ACL 2026.'),
        reference('Ahmad et al. <link href="https://arxiv.org/abs/2609.34678">Fair Fact-Checking with RoSh</link>. 2026.  |  Ghorbanpour et al. <link href="https://arxiv.org/abs/2608.28860">Latent-Space Intervention for Cross-Lingual Factual Consistency</link>. EMNLP 2026.'),
        reference('Poulis et al. <link href="https://arxiv.org/abs/2604.03754">Testing the Limits of Truth Directions in LLMs</link>. 2026.  |  <link href="https://arxiv.org/abs/2502.17955">Language Models Factuality Depends on the Language of Inquiry</link>. 2025.'),
    ))

    assert len(pages) == 15
    for authored_page in pages:
        if authored_page.get("dek"):
            authored_page["dek"] = tidy(authored_page["dek"])
        for element in authored_page["elements"]:
            if "text" in element:
                element["text"] = tidy(element["text"])
            if "caption" in element:
                element["caption"] = tidy(element["caption"])
            if "rows" in element:
                element["rows"] = [[tidy(str(cell)) for cell in row] for row in element["rows"]]

    content = {
        "title": "Matched facts, misleading angles: multilingual truth-direction report",
        "subject": "Controlled fact overlap, difficulty decomposition, score transport, behavioral quadrants, and causal steering",
        "footer": "MULTILINGUAL TRUTH TRANSPORT / MASS-MEAN PROBING / 1 OCT 2026",
        "pages": pages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "pages": len(pages)}))


if __name__ == "__main__":
    main()
