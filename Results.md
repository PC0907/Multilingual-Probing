# Initial probing results: Llama 3.1 8B, English and German

Status as of 7 September 2026. First complete pass: extraction, in-language
probe accuracy over layer depth, and surface-form controls. Teammates are
running the remaining models against the same protocol.

## Setup

`meta-llama/Llama-3.1-8B` (base, not instruct). Forward passes only, no
fine-tuning. Activations pooled at the final token, cached at every layer
(0 to 32, where layer 0 is the embedding output), bfloat16 inference and float32
storage. Analysis reports every fourth layer; the cache holds all of them so the
grid can be refined without further GPU time.

Probe is the mass-mean direction of Marks and Tegmark: theta is the difference of
class means, classification projects onto theta and thresholds at the midpoint of
the training projections. No learned scale, no regularisation. A direction that
classifies is one that exists in the representation rather than one fitted into
existence.

Five-fold cross-validation with whole groups assigned to folds, where a group is
a set of statements sharing an opening phrase. This matters because hand-authored
datasets contain near-minimal pairs, and a random split puts one variant in train
and its partner in test.

## Datasets

| set | source | n | true/false | groups |
| --- | --- | --- | --- | --- |
| `en_cities` | geometry-of-truth `cities.csv` | 1496 | 748 / 748 | 723 |
| `en` | Truth_is_Universal `common_claim_true_false.csv`, balanced subsample | 1996 | 998 / 998 | 1898 |
| `de` | own construction, four topics | 1996 | 997 / 999 | 1309 |

The German set was written for this project: 500 statements each in geography,
sports, entertainment and history/news, balanced within every topic. The English
subsample is size-matched to it.

## Results

Peak grouped accuracy, with the lexical baseline below it. The lexical baseline
is a TF-IDF logistic regression on the raw strings under the same grouped folds:
accuracy reachable with no model and no activations at all.

| set | peak layer | peak accuracy | lexical floor | margin |
| --- | --- | --- | --- | --- |
| `en_cities` | 16 (50% depth) | 0.976 | 0.475 | +0.50 |
| `en` | 12 (38% depth) | 0.757 | 0.596 | +0.16 |
| `de` | 12 (38% depth) | 0.656 | 0.449 | +0.21 |

German by topic, at the peak layer:

| topic | accuracy |
| --- | --- |
| history/news | 0.690 |
| geography | 0.670 |
| sports | 0.627 |
| entertainment | 0.506 |

## Controls

Shuffled labels, permuted within the training fold only, sit at 0.500 for German
and 0.504 for CommonClaim across all layers, confirming the splits do not leak.

The leakage gap, random-split accuracy minus grouped-split accuracy, is -0.016
for German, +0.000 for CommonClaim and -0.001 for cities. Grouping does not
inflate any result. For German this is because the near-minimal pairs place the
same entities on both sides of the label, so surface form carries no consistent
signal, which the below-chance lexical baseline confirms independently.

Centred and uncentred fits agree exactly, as expected: the shared residual-stream
offset cancels in the difference of class means, so centring shifts projections
and threshold by the same constant. It becomes consequential later, at the
geometry stage, where absolute orientation matters.

## What this suggests

**The pipeline is validated.** `cities` reaches 0.976 at the exact midpoint of
the network with the expected profile: chance at the embeddings, 0.900 by layer
8, plateau across the second half. This is consistent with published results for
this dataset on Llama-family models, and it was run with the same code that
produced the German numbers.

**Construction method dominates the measurement.** The gap between `cities` at
0.976 and CommonClaim at 0.757 is 22 points, within one language, one model and
one pooling choice, differing only in how the statements were built.
Template-generated geography against crowd-sourced mixed claims costs more than
the entire English-German difference. Any cross-lingual comparison run on
differently-constructed data is therefore measuring construction before it
measures language.

**Reported against its floor, the German probe does more work than the English
one.** CommonClaim's raw 0.757 exceeds German's 0.656, but CommonClaim starts
from a 0.596 surface-form floor: its false statements are disproportionately
debunked folk beliefs, and the most false-indicating tokens are "not", "actually"
and "contrary to popular belief". The probe adds 16 points there against 21 for
German. Raw accuracy inverts the comparison.

**Probe accuracy within one language is not uniform across content.** German
entertainment sits at chance at every layer while geography and history reach
0.67 to 0.69, on matched construction and equal sample sizes. The entertainment
statements concern German directors, comic prizes and television channels, which
is the category MultiLoKo identifies as locally sourced and unlikely to be
salient in an English-dominated pretraining corpus. Labels were checked by hand
and are correct, so this is not a data-quality artefact.

## Limitations

Single model, single seed, single pooling choice. Everything below is open.

Pooling is final-token throughout, because none of the three datasets carries
entity spans. This is the choice the extraction code itself flags as a confound
for cross-lingual work, since the final token is not the same linguistic position
in verb-final languages. `cities.csv` carries city and country columns from which
spans are recoverable, so the effect of pooling can be measured there.

The German set has almost no negated statements, so polarity and truth cannot be
separated and the two-dimensional truth subspace cannot be estimated. RQ1's
principal-angle method is unavailable for German as the data currently stands.
Rebuilding on the full truth-by-polarity design would fix this and would also
supply entity spans.

The shuffled-label control on `cities` fluctuates more than the others, reaching
0.625 at layer 24 and 0.411 at layer 32 against a mean of 0.522. Plausibly noise
at 723 groups over five folds, but not yet confirmed across seeds.

The entertainment result has two readings that have not been separated: the model
may not encode these facts, or it may encode them somewhere the final-token probe
cannot read. A behavioural check, asking the model the same facts directly and
measuring accuracy per topic, distinguishes them and has not been run.

## Next

Behavioural check on German by topic. Repeat all sweeps across seeds to settle
the shuffled-control variance. Decide whether to rebuild the German set on the
truth-by-polarity design with entity spans, which gates RQ1. Then the
topic-transfer matrix within German, which is the content-disjoint control
brought forward to a single language, where content and language are not
confounded.
