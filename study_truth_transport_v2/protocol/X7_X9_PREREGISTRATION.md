# X7 to X9: label-free transfer prediction, scale, and post-training

Status: frozen on 2 October 2026 before any X7, X8, or X9 activation, score, or
outcome existed. These are post-core extensions. The core and X1 to X6 results
were known when this document was written, so nothing here is a confirmatory
test of the original hypotheses.

## Motivation

The completed study shows that Euclidean cosine between language-specific truth
directions can rise (shared training facts, shared latent spaces) without any
gain in held-out transfer. If cosine is the wrong statistic, what should be
reported instead? X7 tests a label-free alternative. X8 and X9 test whether the
core findings and X7 hold across scale, an additional open-data family, and
post-training.

## X7: translation consistency as a label-free predictor of transfer

Unit of analysis: a configuration (model, decoder block, ordered language pair
s→t). Every decoder block of every model is used; nothing is selected.

For each configuration, mass-mean probes are fitted per language on the frozen
training partition (pure dependency groups) at that block.

- Outcome: AUROC of the source probe applied to target-language test groups.
- P1, Euclidean cosine (the field's standard statistic; needs target labels):
  cosine between the source and target mass-mean directions.
- P2, translation consistency (needs no target labels): Pearson correlation
  between the source probe's scores on source-language and target-language
  versions of the same validation dependency groups. Labels are never read.
  Spearman correlation is a prespecified sensitivity.
- P3, Fisher efficiency (secondary, explanatory; needs target training labels):
  `w'Δμ_t / sqrt(w'Σ_t w · Δμ_t'Σ_t^{-1}Δμ_t)`, where `Δμ_t` and the pooled
  within-class covariance `Σ_t` come from target training data, with analytic
  Ledoit-Wolf shrinkage toward a scaled identity. It equals the source
  direction's target separability as a fraction of the best linear
  discriminant's.

Primary hypotheses:

- H1 (within model): across all (block, ordered pair) configurations of one
  model, Spearman ρ(P2, AUROC) exceeds ρ(P1, AUROC). Inference: 500 paired
  bootstrap draws resampling validation groups and label-stratified test groups,
  recomputing P2 and AUROC (training probes held fixed, so P1 is fixed); 95%
  percentile CI for Δρ = ρ(P2) − ρ(P1) per model; Holm correction across models.
  Summary: number of models with Δρ CI above zero, and a two-sided sign test.
- H2 (across models): leave-one-model-out. A one-predictor linear regression of
  AUROC on the predictor is fitted on all other models and scored by R² on the
  held-out model. Report mean held-out R² for P1 and P2.

Secondary analyses (descriptive, reported in full):

- the same comparisons at the frozen layer only (30 pairs per model);
- P3 alongside P1 and P2;
- RQ-A allocations at the frozen layer: overlap slope of P1 versus P2 for the
  allocation probes (P2 does not involve the target probe, so a content-overlap
  artifact should move P1 but not P2);
- X1 transforms at the frozen layer (RoSh, scrambled RoSh, ridge, PCA-1): P2 is
  computed with each method's effective target score vector;
- Pythia-1.4B-deduped checkpoints, as a training-time set kept out of H1/H2.

Decision rule: H1 is supported only if Δρ's CI is above zero in a majority of
models. If P2 does not beat P1, that is reported as the result.

## X8: scale and an open-data family

Models (exact revisions frozen in `config/model_revisions.json`):
Qwen3-0.6B-Base, Qwen3-1.7B-Base, Qwen3-4B-Base (with the existing
Qwen3-8B-Base), OLMo-2-1124-7B, and Qwen3-14B-Base only if at least 40 GiB of
disk is free when its download starts (otherwise reported as infeasible).
Each runs the frozen core (layer selection, RQ-A, RQ-B with behavioral scoring,
RQ-C, controls, all-layer analysis), X1 alignment baselines and LSI latent, X3
MMMLU and X6 INCLUDE external transfer (frozen compaction rule), and X7. X2
steering is not run for these models. Scale trends over the Qwen3 series are
descriptive (at most five sizes).

## X9: post-training

Qwen3-8B (post-trained, used with the same base-style prompts, no chat
template) receives the full X8 battery and is compared with Qwen3-8B-Base on
RQ-A slopes, RQ-C transport, X1/LSI, X3/X6, and X7. The comparison is
descriptive with dependency-group intervals from each model's own analyses.

## X7 coverage of existing models

The four core families are re-downloaded at their exact revisions. Their
all-layer caches are regenerated with the frozen extractor at the original
batch size, must reproduce the stored RQ-C AUROC and cosine to within 1e-6, and
are then used only for X7. Pythia checkpoints are re-downloaded for the
secondary training-time set.

## Constraints carried over

Mass-mean probes only; dependency-group splits; test labels never select
anything; one GPU at a time; exact revisions; deviations logged before the
affected outcome is inspected; checkpoints and reproducible caches deleted
after verified concise outputs.
