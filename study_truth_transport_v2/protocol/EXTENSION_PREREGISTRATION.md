# Post-core extension protocol

Status: frozen before inspecting any extension outcome. Date: 1 October 2026.
The core RQ-A--RQ-E results were already known when this document was written,
so every analysis below is a prospective replication or extension rather than a
new confirmatory test of the original hypotheses.

## X1: same-split alignment baselines

Question: does the zero-overlap mass-mean result remain informative when it is
compared directly with stronger cross-lingual alignment methods?

For every ordered language pair and completed model, fit transformations on the
original dependency-group-disjoint training partition and score the unchanged
test partition. Transform fitting uses parallel group identities but never truth
labels. The source mass-mean direction is the only supervised component.

Methods:

1. raw source mass mean;
2. centroid shift from target to source;
3. a single-layer Rotation--Shift (RoSh) map using the standard full orthogonal
   Procrustes SVD at the model's frozen primary layer;
4. a scrambled-correspondence RoSh control using a fixed derangement;
5. a fixed-ridge unconstrained target-to-source map, evaluated through its
   induced source-probe score vector;
6. raw-space LSI ablations: remove the first principal component of paired
   target-minus-source differences, and move the target centroid 0.6 of the way
   toward the source centroid.
7. an LSI-style shared autoencoder trained without labels on all parallel
   training-group pairs at the frozen layer, followed by mass-mean probing in
   its 256-dimensional latent space. The architecture and reconstruction,
   alignment, variance, covariance, and speckle-noise settings follow the
   authors' released training scripts. This is a same-dataset adaptation, not
   their TED-trained end-to-end QA checkpoint.

Primary outcomes are target AUROC, source-threshold balanced accuracy, and
paired changes from the raw source probe. The RoSh and LSI names are qualified
as fixed-layer or raw-space variants whenever they do not reproduce the full
multi-layer or autoencoder pipeline of the original papers.

## X2: damage-budgeted causal steering

Question: do the architecture differences in RQ-E remain after intervention
magnitude is selected by a model-independent off-target damage budget?

For the six prespecified language directions, evaluate the zero-overlap and
target-native directions at alpha magnitudes 0.125, 0.25, 0.5, 1, 2, and 4.
Neutral-text damage is measured with a token-scope-matched intervention. The
largest symmetric magnitude with mean delta NLL no greater than 0.05 is selected
without looking at truth labels. Report the judgment-margin change at that
magnitude, the local symmetric slope, and the random-direction envelope. If no
nonzero magnitude passes, report that no safe operating point was found.

## X3: professionally translated external evaluation

Question: does a truth direction trained on the supplied generic claims transfer
to independently sourced, professionally translated content?

Use OpenAI MMMLU for Arabic, German, Spanish, French, and Hindi, paired with the
original English MMLU test questions. Select ten questions per available subject
using a fixed hash ordering. Each question yields one correct candidate record
and one deterministically selected incorrect candidate record. The paired
candidate records remain in the same resampling group. No MMMLU example is used
to fit or select a truth direction, layer, threshold, or transformation.

Evaluate every trained source-language mass-mean direction in each MMMLU target
language. Primary outcomes are AUROC and paired candidate accuracy. Report macro
means across subjects and dependency-group bootstrap intervals. This is an
out-of-domain transfer test; it is not described as native-authored content.

## X4: established-family replication

If a public, ungated Mistral-7B base checkpoint is downloadable, run the frozen
activation extraction, validation-only layer selection, RQ-A, RQ-B, and RQ-C on
the same dataset and splits. Behavioral scoring is required for RQ-B. RQ-D and
causal steering are secondary because the decisive purpose is replication of the
overlap and score-transport claims in a widely used fourth family. The checkpoint
is processed on one GPU and deleted after artifact verification.

## X5: developmental trajectory

If storage and public checkpoint access permit after X1--X4, use six checkpoints
from one Pythia deduplicated training trajectory. At each checkpoint, select a
layer using validation data and estimate zero-overlap and full-overlap cosine and
transfer with the frozen allocations. This answers RQ-F only for that training
trajectory; it does not identify a universal training phase transition.

## Decision rules and multiplicity

- Group identities, not rows, are the uncertainty unit.
- Test labels never select a layer, transform, alpha, or checkpoint.
- Baseline comparisons use paired group bootstraps and Holm correction within
  each model and outcome family.
- External-transfer claims require both macro-subject and micro estimates.
- A method that improves alignment but harms AUROC, balanced accuracy, or neutral
  NLL is reported as a tradeoff rather than an improvement.
- Unavailable gated checkpoints, missing official artifacts, and infeasible
  compute are documented as scope limitations rather than silently substituted.
