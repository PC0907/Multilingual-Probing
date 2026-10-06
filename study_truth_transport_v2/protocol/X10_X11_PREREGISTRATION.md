# X10 and X11: an estimator-noise account of overlap-inflated cosine, and automatic translation-quality sensitivity

Status: frozen on 3 October 2026 before any X10 or X11 activation, score, or
outcome existed. The RQ-A cosine and AUROC curves that X10 will be compared
against were already known; X10 tests whether a model fitted only on separate
training statistics predicts them quantitatively. Post-core extensions; not
confirmatory tests of the original hypotheses.

## X10: estimator-noise model of RQ-A

Setting. For a language pair (s, t) the RQ-A allocations draw, from a pool of
paired training groups, n = 400 groups per language (n_y = 200 per class
y ∈ {0, 1}), with exactly k = f·n groups shared (k_y = f·n_y per class,
following the label-stratified allocation) and the remaining groups disjoint.

Model. For each language l, pool group i with label y_i has frozen-layer
activation x_i^l = m_{y_i}^l + r_i^l, where m_y^l is the pool class mean.
Let Δ^l = m_1^l − m_0^l, N_y the pool class size, V_y^l = mean_i |r_i^l|² and
C_y = mean_i r_i^s · r_i^t over class-y pool groups (the cross-language
covariance of a fact's residual). Sampling without replacement gives

    E[w^s·w^t] = Δ^s·Δ^t + Σ_y [k_y C_y − (n_y² − k_y) C_y / (N_y − 1)] / n_y²
    E|w^l|²    = |Δ^l|² + Σ_y V_y^l (N_y − n_y) / (n_y (N_y − 1))

and the predicted cosine is cos(f) = E[w^s·w^t] / sqrt(E|w^s|² E|w^t|²)
(ratio-of-expectations approximation). Its slope in f is
Σ_y C_y (1 + 1/(N_y − 1)) / n_y divided by the same denominator. The source
direction's distribution does not depend on f, so the model predicts a zero
AUROC slope.

Inputs. Only pool-level statistics (Δ, V, C) computed from training-partition
activations at each model's frozen layer; no allocation results, test data, or
RQ-A outcomes are used. The pool is the set of groups appearing in any plan of
that pair's allocation file.

Outcomes, over all 15 unordered pairs × 10 primary models (150 units):
- Pearson correlation between predicted and observed cosine slope (observed:
  linear fit of RQ-A mean raw cosine over the five overlap levels);
- calibration: median of observed/predicted slope, and the slope of a
  regression of observed on predicted through the origin;
- the same comparison for predicted versus observed cosine at f = 0 and f = 1.

Decision rule (frozen): X10 is "supported" if the slope correlation r ≥ 0.7
and the median observed/predicted slope ratio lies in [0.8, 1.25]; "partially
supported" if r ≥ 0.5; otherwise "not supported".

## X11: automatic translation-quality sensitivity

Each claim's English statement is compared with each translation using LaBSE
sentence embeddings (`sentence-transformers/LaBSE`, pinned revision; CLS pooler
output, L2-normalized, cosine similarity). In each non-English language, the
dependency groups whose lowest member similarity falls in the bottom 10% are
flagged. Flags use no labels and no model outputs.

Sensitivity, per model, at the frozen layer, excluding flagged groups from
training and test in the languages where they are flagged:
- mean cross-language AUROC and source-threshold balanced accuracy;
- frozen-layer Spearman correlation of cosine, translation consistency, and
  Fisher efficiency with transfer (the X7 statistics), and which is best.

Decision rule (frozen): conclusions are "robust" if, in every model, the mean
cross-language AUROC changes by less than 0.02 and the best frozen-layer X7
statistic is unchanged. This is an automatic proxy, not human review, and is
reported as such.

## Execution

Each of the ten primary models is re-downloaded at its exact revision and its
all-layer cache is regenerated with the frozen extractor at the original batch
size (`runners/regenerate_and_verify_cache.sh`), which must reproduce every
stored RQ-C AUROC and cosine to within 1e-6 before use (this path reproduced
stored results exactly for every model in X7). X10 and X11 read only the frozen
layer. Checkpoints and caches are deleted after the concise outputs are
written; one GPU at a time; only the user's own directories are written.

## X12: few-shot target labels for predicting transfer (added before any X10-X12 outcome)

Question: when only k labeled target-language examples are available, does
the covariance-aware statistic predict transfer better than simply measuring
the source probe's AUROC on those k examples?

Configurations: every decoder block × 30 ordered pairs, per primary model (as
in X7). For each configuration and each k ∈ {8, 16, 32, 64}, draw R = 20
label-balanced samples of k target validation groups (fixed seeds; the same
draws for every configuration of a model and target language). Predictors
computed from each draw:
- DIRECT_k: AUROC of the source probe on the k labeled target groups;
- FISHER_k: w'Δμ_k / sqrt(w'Σ_u w · Δμ_k'Σ_u^{-1}Δμ_k), with Δμ_k the class-mean
  difference of the k labeled groups and Σ_u the Ledoit-Wolf-shrunk total
  covariance of the unlabeled target training activations (no training labels);
- COS_k: cosine between the source direction and Δμ_k.
Outcome: Spearman correlation with the held-out full-test AUROC across a
model's configurations, averaged over the R draws (draw-level Spearman values
are reported too).

Hypothesis H3 (primary for X12): at k = 16, mean ρ(FISHER_k) > mean ρ(DIRECT_k)
within model. Inference: per model, the difference is summarized over the 20
draws (mean and 2.5/97.5 percentiles across draws); across models, a two-sided
sign test. H3 is "supported" if the difference is positive in at least 8 of
the 10 primary models. Other k values are reported as a curve.

## Theory note (analytic, no data)

The paper will state and prove that, for class-conditional Gaussians with a
shared covariance Σ_t, the AUROC of any linear score w is Φ(F(w)·d*/√2), where
F is the Fisher efficiency and d* the optimal discriminability, and that
Euclidean cosine with the target mass-mean direction equals F for every w only
when Σ_t is isotropic. This is a derivation, not an empirical claim.

## X13: probe-family robustness (added before any X10-X13 outcome; user lifted the mass-mean-only constraint)

Probe families, all fit on training groups at each model's frozen layer:
- MM: mass mean (the study's estimator);
- LDA: shrinkage linear discriminant, Σ_LW^{-1}Δμ with the pooled within-class
  covariance and analytic Ledoit-Wolf shrinkage; midpoint threshold;
- LR: L2 logistic regression on raw activations, C ∈ {0.001, 0.01, 0.1, 1}
  chosen once per model by mean within-language validation AUROC; threshold 0
  on the logit.

Analyses per model and family:
1. Overlap dose response on the first 10 allocations per overlap level
   (repetitions 0-9) of every unordered pair, both directions: probe-probe
   cosine and transfer AUROC; slopes per unit overlap averaged over the 30
   ordered pairs with a t-interval over pairs.
2. Full-training cross-language transfer over 30 ordered pairs: mean AUROC and
   source-threshold balanced accuracy.
3. Frozen-layer X7 statistics for each family's source direction: Spearman of
   cosine (with the target probe of the same family) and of Fisher efficiency
   with transfer.

Decision rule (frozen): the overlap dissociation "replicates" for a family if,
in at least 8 of 10 models, the mean cosine slope has an interval above zero
and the mean AUROC slope has absolute value below 0.01. Families are compared
descriptively otherwise; no family is selected using test data.
