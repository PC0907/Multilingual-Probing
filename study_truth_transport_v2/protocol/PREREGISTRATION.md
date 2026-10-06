# Preregistration: multilingual truth-direction measurement on dataset_v2

Status: prospective analysis contract frozen before confirmatory RQ-A results. Date: 1 October 2026. This is a local time-stamped analysis plan, not a third-party registry entry. Unknown model layers and checkpoint revisions remain blank until cache preflight; filling those fields does not permit changing hypotheses or selection criteria.

## Scope and dataset

Dataset: `generic_claims_2000_v2`, manifest SHA-256 recorded in `data/rq_a_allocation_manifest.json`.

Languages: English, German, Arabic, Hindi, French and Spanish.

Independent unit: dependency group. Same-label paraphrase members are averaged within language. Two mixed-label training groups are excluded from probe fitting and listed in `data/derived_group_metadata.json`.

Primary models: Qwen3-8B-Base, Llama-3.1-8B base and Apertus-8B base. Secondary direct-comparison model: Gemma-7B. Optional models are excluded from headline tests unless added in a dated amendment before their results are inspected.

## Frozen representation

- Residual-stream output after each decoder block.
- Final statement token inside the frozen language-specific judgment prompt.
- No mean pooling in the primary analysis.
- Inputs exceeding the frozen context limit abort rather than truncate.
- Decoder blocks numbered 1 through L; embedding output excluded.

Prompt texts and answer strings: **pending tokenizer and speaker-review freeze**. This field must be completed before fresh activation or behavioral extraction.

## Primary-layer selection

For each model, fit a mass-mean probe at every decoder block separately in each language using all eligible pure-label training groups. Evaluate within-language AUROC on validation groups. Select the one block maximizing the arithmetic mean of the six validation AUROCs. Break exact numerical ties in favor of the earlier block.

No test activations, RQ-A allocations, behavioral difficulty results or intervention results may influence layer selection.

| Model | Exact revision | Layers considered | Selected block | Mean validation AUROC | Selection timestamp UTC | Evidence file hash |
|---|---|---:|---:|---:|---|---|
| Qwen3-8B-Base | pending | pending | pending | pending | pending | pending |
| Llama-3.1-8B base | pending | pending | pending | pending | pending | pending |
| Apertus-8B base | pending | pending | pending | pending | pending | pending |
| Gemma-7B | pending | pending | pending | pending | pending | pending |

After this table is completed, selected blocks cannot change. All-block results are appendix sensitivity analyses.

## RQ-A allocation

For each of 15 unordered language pairs, use the frozen plan under `data/rq_a_allocations/`. Each fit contains 400 dependency groups and 200 groups per label. Overlap contains exactly 0, 100, 200, 300 or 400 shared group identities. Fifty paired repetitions are used.

The allocator exactly matches label, pair-normalized length quartile, negation in either surface, number presence and translation-risk status. Topic is not matched because the source data has no documented topic annotation.

Transfer is evaluated in both directions for each unordered pair, producing 30 ordered cells per condition and model.

## Headline hypotheses

H1 — overlap inflation: raw cross-language direction cosine has a positive linear trend with shared-group fraction.

H2 — independent-content alignment: at 0% overlap, raw cosine and the reliability-adjusted estimate exceed the corresponding stratified label-permutation controls.

H3 — geometry versus utility: the standardized overlap slope for cosine differs from the overlap slope for target-test AUROC. The direction is not assumed in advance; the estimate and interval determine interpretation.

H4 — difficulty contribution: removing the cross-fitted difficulty direction changes zero-overlap cosine and transfer. A residual above-null signal supports truth-specific information beyond the measured difficulty composite; disappearance supports a difficulty explanation.

H5 — score transport: source-threshold balanced accuracy improves after validation-estimated unlabeled recentering/rescaling, while AUROC remains numerically unchanged. Degradation under 30/70 and 70/30 priors quantifies prior sensitivity.

H6 — behavioral compartmentalization: source-to-target AUROC in the behaviorally source-known/target-unknown quadrant exceeds 0.5 and its stratified permutation control. A null result is retained as evidence against the proposed compartmentalized-signal interpretation.

H7 — causal validation: steering along a zero-overlap source direction changes the target-language true-versus-false judgment margin monotonically with alpha and more strongly than norm- and target-scale-matched random directions. Claims are limited to the frozen prompt and intervention site.

## Primary estimands

- raw direction cosine;
- reliability-adjusted cosine, reported unclipped with reliability denominator;
- target-test AUROC;
- balanced accuracy at the unchanged source midpoint threshold;
- global offset, target pooled score scale and standardized separation;
- validation-only target-optimal threshold displacement;
- recentering change in balanced accuracy;
- K/U quadrant AUROC;
- steering dose-response slope in judgment logit margin.

## Uncertainty

- 1,000 dependency-group bootstrap draws for held-out predictive metrics.
- 50 paired allocations per overlap condition.
- Hierarchical combination of allocation and fact-group uncertainty for pooled RQ-A transfer estimates.
- Training-group resampling or allocation distribution for cosine uncertainty; test-only bootstrap is not used for direction cosine.
- Holm correction across 15 symmetric pair contrasts or 30 directional transfer contrasts, as applicable.
- Cells with fewer than 30 groups are not interpreted separately; RQ-D pools them under the frozen rule.

## Controls

- within-language disjoint matched-n reliability;
- stratified within-language label permutation;
- fact-identity permutation with fixed-point count recorded;
- 1,000 uniform random directions;
- 1,000 regularized covariance-matched directions;
- random, difficulty and language-identity directions in RQ-E;
- flagged-translation exclusion sensitivity for RQ-A, RQ-C and RQ-D.

## Go/no-go after Phase 1

Proceed to the full core study if Qwen or Gemma shows either:

1. a non-negligible overlap slope in cosine with an interval excluding zero; or
2. zero-overlap cosine above permutation controls and materially above the random-direction floor after accounting for reliability.

If neither criterion holds, stop new-model acquisition and reframe around score transport or the validation of matched-content practice. The Phase-1 result remains reportable.

## Deviations

Every deviation is recorded before viewing the affected outcome, with date, reason, affected hypotheses and whether it is confirmatory or exploratory. No result-driven replacement of models, layers, prompts, strata or alpha values is permitted.
