# Cross-Lingual Truth Representations

Do the geometric structures encoding propositional truth inside a language model
correspond across languages?

This repository contains code for measuring per-language truth directions in
frozen LLMs, comparing their geometry, testing cross-lingual probe transfer, and
separating both from translation artefacts.

Target venue: NAACL 2027 (ARR October cycle).

---

## The question

A linear probe trained on a model's hidden activations can predict whether a
statement is true, at 85–95% accuracy, including for statements the model itself
asserts falsely. The vector normal to that probe's decision boundary is the
*truth direction*. Almost all evidence for it is English-only.

Two possibilities, with opposite consequences:

- **One shared direction.** Language is clothing on input and output; an
  English-trained safety monitor works everywhere at no extra cost.
- **Separate directions per language.** An English-validated monitor fails for
  most of the world's users, and fails *silently* — still emitting confident
  scores.

The literature pulls both ways. Wendler et al. find a latent-English pivot in
middle layers (predicts sharing); Schut et al. argue the concept space is
English-centric rather than universal; CLAS reports cross-lingual steering
transfer correlating with representational *divergence* rather than alignment.

## What is and isn't new here

"Do truth probes transfer across languages?" is partly answered — CrossHallu
(Arabic–English), Shared Doubt (confidence estimation), MultiHaluDet
(French/Bangla/Amharic). All measure AUROC under transfer. All use
machine-translated data.

Three things remain unclaimed:

1. **Geometry rather than detection performance.** Transfer AUROC tells you a
   probe worked; it does not tell you whether two languages found the same
   direction. We measure directions, principal angles between truth subspaces,
   and layer profiles.
2. **The dissociation.** The interesting hypothesis is that geometric alignment
   does *not* predict transfer. That makes RQ1 and RQ2 mutually necessary rather
   than sequential.
3. **Native sourcing with a content-disjoint control.** If a probe transfers on
   translated data, shared truth representation and shared translation residue
   are indistinguishable. Native, content-disjoint data separates them.

## Research questions

| | Question | Method | Status |
|---|---|---|---|
| RQ1 | Do truth directions align across languages? | Per-language mass-mean probes at every layer; pairwise cosine and principal angles, normalised to a within-language ceiling | Core |
| RQ2 | What predicts when a probe transfers? | Train in A, test in B, all ordered pairs; regress on tokenizer fertility, typological distance, script, pretraining-share proxy | Core, descriptive |
| RQ3 | Is transfer mediated by the latent-English pivot? | Verify the pivot exists in the models under study, then ablate it and measure transfer degradation | Conditional on verification |
| RQ4 | Does a direction from one language steer another? | Cross-lingual ITI / contrastive activation addition, with mandatory persona-prompting baselines | First to cut |

RQ1 + RQ3, with RQ2 descriptive, is a complete paper. RQ4 is not a ten-week
project.

## Repository layout

```
configs/
  languages.yaml          language set, scripts, families, covariates
  models.yaml             model ids, layer counts, hook points
src/
  data/
    wikidata.py           SPARQL queries for locality-filtered entities
    templates.py          per-language statement templates + negation frames
    build_statements.py   entity + template -> labelled true/false pairs
    multiloko.py          adapter: MultiLoKo QA -> declarative statements
  activations/
    extract.py            forward passes, all-layer activation caching
    pooling.py            token-position selection (last, mean, entity-final)
  probes/
    mass_mean.py          mu_true - mu_false, per layer
    logistic.py           supervised linear baseline
    ccs.py                Burns et al. unsupervised contrast-pair probe
  geometry/
    similarity.py         cosine, normalised to ceiling
    subspace.py           principal angles via SVD of orthonormal bases
    baselines.py          random-direction floor, split-half ceiling
  transfer/
    crosslingual.py       train-in-A test-in-B grid
    covariates.py         fertility, lang2vec distance, script, resource proxy
  pivot/
    verify.py             replicate Wendler et al. on target models
    ablate.py             subspace removal, causal test for RQ3
scripts/
  00_build_english.py     English statement sets, all types
  01_ceiling.py           split-half English cosine — the number everything is normalised to
  02_extract.py           cache activations for a language set
  03_geometry.py          RQ1 layer profiles
  04_transfer.py          RQ2 grid
  05_disjoint_control.py  content-disjoint alignment check
data/
  raw/  interim/  processed/
results/
  figures/  tables/  logs/
```

## Measurement cautions (read before interpreting any number)

**Cosine similarity is meaningless without baselines.** In a few-thousand
dimensional space, random directions have cosine near zero, so any positive
value looks impressive. Two anchors are required:

- **Floor:** random-direction baseline.
- **Ceiling:** two probes trained on disjoint *English* splits. They will not
  reach 1.0. Whatever they reach is the realistic maximum, and all cross-lingual
  alignment is reported normalised to it.

**Use principal angles, not single-vector cosine, where Burger et al. apply.**
The truth subspace is 2D — one axis for true/false, one separating affirmative
from negated statements. Comparing planes requires angles, not a dot product.
This means negated pairs are mandatory in the data, not optional.

**Twenty languages with five collinear regressors is not a causal regression.**
RQ2 is framed as descriptive correlation with leave-one-family-out
cross-validation and explicit confidence intervals. Pretraining share is
unpublished for most open models and is used as a documented proxy.

**Verify the pivot before assuming it.** Wendler et al. characterised the
latent-English pivot on Llama-2. Newer tokenizers and training mixes may not
reproduce it. A negative result here is a reportable preliminary finding, not a
project failure.

## Data design

Probe training needs **declarative statements with binary labels**, several
thousand per language. No multilingual resource at this format and scale exists;
existing cross-lingual probing work uses machine-translated TruthfulQA.

Our approach:

- **Wikidata-templated native generation** for scale. Entities filtered by
  locality (administrative region, country of origin), filled into
  language-specific templates. Negatives generated by entity substitution, so
  labels are correct by construction.
- **Several statement types per language** (geography, comparisons,
  institutional facts, translations), because a probe trained on one topic can
  learn a topic classifier rather than a truth direction.
- **Negated counterparts for every affirmative**, required by the 2D subspace
  measurement.
- **Matched surface form** between true and false: differ only in the swapped
  entity, not in length or syntax.
- **Human annotation reserved** for facts templates cannot reach — literature,
  history, cultural practice. Every hand-written item needs a second annotator
  confirming the label; label noise propagates directly into the truth direction
  and looks like low cross-lingual alignment for uninteresting reasons.

### Why not translate an English set

Two distinct objections, and a probing study suffers both:

1. **Content bias.** Translation imports English facts, so the model is only
   tested on topics it is already comparatively good at. MultiLoKo reports local
   versus English-translated data shifting scores by more than twenty points.
2. **Surface residue.** Translated text carries translationese — mirrored word
   order, calqued phrasing. If English and Hindi statements are translations of
   each other, they share facts *and* form, and a transferring probe cannot be
   attributed to either.

Translated data is therefore kept as a **deliberate comparison arm**, not
discarded. MultiLoKo's parallel native / human-translated / machine-translated
partitions hold facts constant across three provenances, which isolates the
translation artefact more cleanly than native-vs-translated with different facts.

### Content-disjoint control

Derive the direction from English set A; evaluate against target-language set B
covering entirely different facts. If alignment collapses under this control,
the finding was an artefact.

Match statement *types* and rough difficulty across languages even while facts
differ, or content-disjoint quietly becomes difficulty-disjoint and any
difference is uninterpretable.

**Run this in week two.** It can invalidate everything downstream, and that is
much cheaper to learn early.

## Compute

Probing requires forward passes only — no base-model training. One pass caches
all layers simultaneously, so layer sweeps cost disk, not compute. The bottleneck
is annotated data, which makes the scoping decision a data-collection decision.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Week one: the number everything else is normalised against.
python scripts/00_build_english.py --n-per-type 1000
python scripts/01_ceiling.py --model <model-id> --splits 2
```

`01_ceiling.py` trains two mass-mean probes on disjoint English splits and
reports their cosine similarity per layer. Until that curve exists, no
cross-lingual number can be interpreted.

## Open decisions

- **Language set.** Tamil is absent from MultiLoKo, which constrains how much of
  the native slice that resource can carry. A set dissociating script from
  family — e.g. Hindi/Urdu (near-identical language, different scripts) plus a
  Dravidian language plus a high-resource European control — buys more than
  breadth.
- **Dataset size.** To be decided from the English scaling curve
  (400 / 800 / 1200 …): find where in-language probe accuracy saturates, then
  match that per language.
- **ARR cycle.** October (NAACL 2027, ~10 weeks) versus December (ACL 2027,
  ~4 months). The native slice is the deciding factor.

## Related work to read adversarially

Before finalising framing, write one paragraph each on exactly what we do that
these do not:

- CrossHallu (arXiv:2607.04029) — cross-lingual hallucination signal transfer
- Shared Doubt (arXiv:2605.31220) — zero-shot cross-lingual confidence probing
- CLAS (arXiv:2601.16390) — transfer through divergence, not alignment

Do not use any variant of "the geometry of truth" in the title; Bao et al.
(Findings ACL 2025) have it, and it is the model-dimension analogue of this work.
