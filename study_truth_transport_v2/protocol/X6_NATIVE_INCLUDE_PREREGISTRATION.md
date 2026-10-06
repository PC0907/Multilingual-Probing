# X6: native-authored external evaluation (INCLUDE)

Status: frozen on 2 October 2026 before any INCLUDE activation, score, or
performance value existed. Added at the user's request to address the
missing native-authored evaluation. It is a post-core extension, reported
separately from the confirmatory core and from X1 to X5.

## Question

Does a mass-mean truth direction trained on the translated generic-claim set
transfer zero-shot to multiple-choice exam questions written natively in each
target language from local sources?

## Data

- `CohereLabs/include-base-44`, dataset revision
  `d2e1f6015f67a43c02a9a68db98e2298e2d6a660`, Apache-2.0, test split only.
- Target languages: Arabic, German, Hindi, French, Spanish. INCLUDE has no
  English subset; the native-English comparison is the original English MMLU
  target already in X3.
- Every test row is used, except two label-blind exclusions applied in file
  order: (1) rows whose four options are not pairwise distinct after
  whitespace stripping, because the true and false candidates could then be
  identical text; (2) later repeats of an exact question text within a
  language. Realized counts after exclusion (computed from the data, before any
  model run): ar 540, de 138, hi 543, fr 419, es 549.
- Each question gives one correct candidate and one incorrect candidate
  chosen among the three wrong options by the X3 SHA-256 rule
  (`include-wrong-v1|20261003|<language>|<row>`). Statements use the X3
  templates and activation prefixes unchanged. The two candidates form one
  resampling group.
- Prompts that exceed the frozen context limit are compacted with the X3 rule
  in `protocol/DEVIATIONS.md` (question-head truncation shared by both
  candidates; the question is excluded in that language if infeasible).

## Analysis

- Models: every model whose exact checkpoint is available. A model whose
  all-layer training cache was deleted has it regenerated with the frozen
  extractor. The regenerated cache must reproduce the stored RQ-C AUROC matrix
  to within 1e-6 before it is used.
- For each of the six source-language directions (generic-claim training
  partition only, frozen layer) and each INCLUDE target language: micro
  AUROC, paired candidate accuracy, source-threshold balanced accuracy, and a
  1,000-draw question-group bootstrap for AUROC and paired accuracy.
- Secondary breakdown by INCLUDE `regional_feature` (agnostic, culture,
  region implicit, region explicit), reported descriptively.
- No INCLUDE item selects any layer, direction, threshold, transform, or
  truncation. No outcome-dependent re-filtering.
- Native (X6) versus translated (X3) results use different questions and are
  compared descriptively only, not as a controlled contrast.
