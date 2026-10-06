# Design audit and amendments

Date: 1 October 2026. This audit was completed before running RQ-A on dataset_v2 activations.

## Decision

The proposed study has a strong central question, but several statements in the initial specification do not hold for the supplied data and would create leakage or invalid uncertainty estimates if implemented literally. The amendments below are part of the frozen analysis contract.

## Dataset units and sample size

The dataset has 2,000 claim rows, not 2,000 independent facts. Multilingual collision grouping yields 1,972 dependency groups. The new split contains 1,200 training rows but 1,172 training groups. Two training groups contain a true statement and its explicitly negated false counterpart:

- “The closest star to Earth is the Sun” / “... is not the Sun”;
- “The universe is infinite” / “The universe is not infinite.”

These are valid dependency groups for split and bootstrap purposes but have no single group label. They are excluded from group-mean probe fitting and RQ-A allocation. The remaining training pool has 1,170 pure-label groups: 580 false and 590 true.

Same-label paraphrases inside a group are averaged before fitting or evaluation, giving every independent group weight one. Averaging a mixed-label group would destroy its truth contrast, so mixed groups are excluded from confirmatory probe fits. Sensitivity analyses may retain their individual rows while clustering uncertainty by group, but that result must be labeled separately.

## Missing topic metadata

The supplied CSV contains only `statement` and `label`. No documented topic field exists. Inferring topics with an LLM or unsupervised clustering and then treating them as observed metadata would introduce a model-dependent post hoc variable. Primary allocation therefore does not claim topic matching.

RQ-A instead matches exactly on:

- truth label;
- pair-normalized statement-length quartile;
- negation in either language surface;
- number presence in the English source proposition;
- automatic translation-risk status in either language.

Each pair-specific plan records sparse strata excluded for zero-overlap feasibility. Only 5–12 of 1,170 groups are excluded, depending on the language pair. A sensitivity may use embedding clusters as a descriptive lexical-composition control, but these cannot be described as human topics.

## Translation validation

Dataset_v2 contains 47 researcher-screened translation repairs across 21 claim IDs. It remains machine translated and is not independently human validated. The 1,000-row bilingual review pack is an annotation instrument, not evidence that review has occurred. Until completed annotations exist, all claims must say “parallel translated claims,” not “validated multilingual benchmark.”

Automatic translation flags are conservative screening variables. Negation-marker mismatch is often a false positive caused by lexicalization, and absence of a flag is not proof of equivalence. Corrected rows remain marked as translation-risk cases so the headline sensitivity does not benefit from post hoc removal of known errors.

## Representation and prompt consistency

The v1 activation caches were extracted from bare statements. The new specification requires the final statement token inside a translated judgment prompt. Bare-statement and prompted activations are different estimands. They must not be mixed in one headline analysis.

Phase 1 may replay v1 caches only as an explicitly exploratory continuity check. Confirmatory dataset_v2 runs require fresh caches under one frozen prompt format per language. The exact rendered prompt, tokenizer IDs, truncation counts and final-token index must be saved with every cache.

Prompt translation has not been independently speaker reviewed. Until it is, prompt-equivalence is a limitation and prompt sensitivity is required. The prompt must not contain the gold label, record ID, topic, partition or translation flag.

## Layer selection

Selecting a different layer for every language and then comparing vectors from different blocks confounds language with depth. The primary layer is one block per model, selected by the mean of six within-language validation AUROCs. The layer is frozen before RQ-A–RQ-E. Decoder-block outputs are numbered 1 through L; embedding output is excluded. All-layer curves are sensitivity results and cannot replace the frozen-layer result.

The validation split participates only in layer selection and diagnostics such as target-optimal threshold. Test activations and behavioral outputs cannot influence layer choice.

## RQ-A allocation

The overlap unit is dependency-group identity. For each language pair and repetition, sample A is fixed across the five overlap levels. Both fits contain 400 groups, 200 per label, and identical stratum counts. Sample B contains exactly 0, 100, 200, 300 or 400 groups from sample A. Nested shared prefixes and fixed exclusive orderings pair conditions and reduce Monte Carlo noise.

Overlap is varied in training only. Test groups remain fixed and disjoint from every training group. Direction cosine is symmetric and analyzed over 15 unordered pairs. Transfer metrics are directional and analyzed over 30 ordered pairs using the same underlying allocations.

## Reliability adjustment

The requested `cos / sqrt(r_s r_t)` is a disattenuation analogue, not a conventional cosine. It can exceed one and can become unstable when reliability is small. It will be reported unclipped with its denominator and undefined cases. Plots may show a reference band but must not silently truncate estimates to [−1, 1]. Raw cosine remains primary.

Reliability is estimated from two disjoint 400-group samples with the same pair-specific composition constraints. Because reliability estimates reuse a finite training pool, their Monte Carlo spread is not a population confidence interval.

## Random and permutation controls

Uniform random unit directions are a geometric floor but ignore anisotropic activation covariance. A covariance-matched floor requires a prespecified regularized covariance estimator because hidden dimension exceeds sample size. The primary null is label permutation within language and allocation stratum, preserving sample sizes and covariance. A secondary covariance-matched Gaussian null uses a diagonal-plus-low-rank shrinkage estimate fitted on training activations only.

Fact-identity permutation must break language correspondence without changing labels or stratum composition. It cannot reuse the same permutation for true and false strata if that creates accidental fixed points; fixed-point counts are recorded.

## Bootstrap and repeated allocations

Translations are never bootstrapped independently. Test uncertainty resamples dependency groups. Allocation variance and test-group uncertainty are distinct:

1. compute each metric for every frozen allocation;
2. within an allocation, bootstrap test groups;
3. aggregate by sampling allocations and test bootstrap replicates hierarchically.

Calling the central 95% range across 50 allocations a confidence interval would be incorrect. It is labeled an allocation-sensitivity interval unless combined with the group bootstrap.

For cosine, uncertainty comes mainly from fitted training directions. A test-only bootstrap does not quantify it. Direction-cosine intervals therefore resample training groups within the allocation constraints or use the repeated-allocation distribution. Test bootstrap is used for transfer metrics.

## Trend analysis and multiple comparisons

There are only 15 independent unordered language pairs, and pairs share languages. A random-intercept model for pair alone understates dependence. Primary inference uses a paired hierarchical bootstrap across allocations and language pairs. Mixed-effects slopes are descriptive, with crossed source- and target-language effects where the optimizer is identifiable.

Holm adjustment is applied to prespecified pair-level endpoint contrasts: 15 tests for symmetric cosine and 30 tests for directional transfer metrics. The five overlap points are modeled as a trend rather than treated as five independent hypothesis families.

## RQ-B difficulty

“Difficulty” is partly defined by correctness, which uses the supplied truth label, and partly by confidence, consistency, entropy and surprisal. It is not label free. Each component's direction is oriented explicitly so higher composite values mean easier/more confidently known.

Components are standardized within model and language before averaging. Statement surprisal is token-normalized and reported separately because tokenization differs across languages. Correctness is discrete and should not be treated as Gaussian evidence.

The original cross-fitting statement is insufficient for test residualization. The amended procedure is:

- create five folds inside training groups only;
- for each fold, estimate the difficulty direction on the other four folds and residualize the held-out training fold;
- fit the truth direction from the union of out-of-fold residualized training activations;
- estimate one final difficulty direction on all training groups and use it only to transform validation and test activations;
- never estimate a removal direction from validation or test activations.

For a pooled multilingual difficulty direction, center within language and weight languages equally. Otherwise the direction may encode language identity. Per-language removal is the primary sensitivity.

Easy and hard terciles must be selected within language and label, or label imbalance will contaminate the difficulty vector. Middle-tercile examples are not used to estimate the binary easy–hard direction but remain available for evaluation.

## Behavioral true/false scoring

Translated “true” and “false” strings can contain different numbers of tokens. Comparing only the first answer token is invalid. The behavioral margin is the difference between the full sequence log likelihoods of the two frozen answer strings under the same prompt. Both summed and per-token-normalized margins are recorded; the primary variant must be frozen after tokenizer inspection and before accuracy is examined.

Prompt-template consistency requires identical semantic scope, answer format and label order. Template translation review remains a prerequisite for strong cross-language behavioral claims.

## RQ-C transport and recentering

Target-optimal thresholds use target validation labels only and are diagnostic. They are never used for zero-shot test claims.

Unlabeled target recentering and rescaling must estimate moments on a separate unlabeled calibration partition, preferably validation inputs with labels hidden from the transformation. Estimating moments directly on the evaluated test subset is transductive and can exploit its class prior.

The 30/70 and 70/30 prior-shift checks require separate calibration and evaluation samples with the same imposed prior. Reusing one subsample for both understates variance. AUROC should remain invariant to affine recentering up to numerical ties; any reported AUROC change is a diagnostic for an implementation error or non-affine procedure.

Expected calibration error is bin dependent and unstable at n=400. Report adaptive-bin ECE with the binning rule, plus Brier score and log loss. The source logistic mapping is a probability calibrator, not an additional hidden-state probe.

## RQ-D knowledge quadrants

The “known” label is behavioral and prompt dependent. It does not establish stored knowledge. Use “behaviorally known under the three frozen templates.” The source and target quadrant assignments use test labels for analysis, which is acceptable for evaluation but cannot influence fitting.

Conditioning on correctness can induce selection bias and change class balance. Every quadrant reports label counts as well as group counts. AUROC is undefined for a single-class quadrant. Pooling occurs only under the preregistered <30-group rule and uses a hierarchical group bootstrap.

A K/U AUROC above 0.5 shows decodable truth information in target-language hidden states despite failure on the frozen behavioral prompts. It does not prove that the model “knows but cannot express” the fact under every elicitation method.

## RQ-E steering

Probe training and intervention must use the same prompted context and same residual-stream location. Steering a bare-statement direction inside a judgment prompt would combine representation shift with intervention.

Hooks are placed after the frozen decoder block at the final statement token during the prefill pass. “All statement tokens” is a separate sensitivity. Direction and target projection scale are computed from training or validation groups only. Test outcomes cannot determine alpha, layer or normalization.

Multiple alphas on the same examples create repeated measures. Dose-response slopes use fact-group clustered uncertainty. Random directions need at least 20 draws as specified, but inference treats random-direction draw and fact group as separate variance sources.

The language-identity direction can have much larger norm and overlap with the truth direction. All controls are unit normalized, target-scale matched, and their cosine with the truth direction is reported. An orthogonalized language-direction sensitivity separates direction identity from raw overlap.

For neutral-text perplexity, intervention is applied under a separately frozen token-position policy. Applying only at the final token changes only the next-token loss; it is not comparable to an all-token intervention without explicit labeling.

## Models and storage

Model names, revisions, licenses and tokenizer hashes must be frozen before download. “Llama-3.1-8B base” refers to the official pretrained checkpoint, not an instruct model. Reported language support is a model-card statement and does not define whether a language is absent from pretraining.

Deleting checkpoints is permitted only after cache completeness, activation hashes, tokenizer/config hashes and a reload-independent score replay are verified. Activations and logs are research evidence and are never deleted merely to free checkpoint space. One model is loaded on one selected GPU at a time. Optional models and RQ-F run only after the core RQ-A–RQ-E completion matrix is satisfied.

## Scope recommendation

The main paper should prioritize Qwen3-8B-Base, Llama-3.1-8B, Apertus-8B and the Gemma-7B comparison. Qwen3.5, OLMo and intermediate checkpoints are appendix work only. If translation review remains incomplete, claims involving fine-grained language differences must be qualified, and flagged-item sensitivity is mandatory.
