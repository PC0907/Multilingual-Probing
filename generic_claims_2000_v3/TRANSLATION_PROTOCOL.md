# v3 translations: protocol (frozen 3 October 2026, before any v3 translation existed)

Reason: the first author reported problems with the v2 NLLB-200-distilled-600M translations.

Translator: Claude Opus 5.5 (model id claude-opus-5-5), the assistant running this study session, translating directly in-session. This is LLM machine translation, not human translation, and is reported as such.

Label blindness: the translator reads only `translation_work/source_en_labelfree.json` / `chunk_*.txt` (id + English sentence). Truth labels, groups, partitions, and v2 translations are not shown during translation.

Rules:
1. Translate meaning faithfully, including false claims. Never correct, soften, hedge, or "fix" a false statement; keep every number, unit, name, date, quantifier and negation exactly.
2. Natural, standard written register (de, fr, es: standard; ar: Modern Standard Arabic; hi: standard Devanagari Hindi, common loanwords allowed where natural).
3. One declarative sentence per claim where the English is one sentence; no added explanation, quotes, or notes.
4. Keep proper names in the target language's standard form; keep digits as Western Arabic numerals in all languages.

Frozen downstream design: v2 dependency groups, 1,200/400/400 split, RQ-A allocation files, prompt templates (config/prompts.json), MMMLU and INCLUDE external data, and every analysis protocol are reused unchanged. Only the five non-English claim texts change. v2 results are retained as a translation-system robustness comparison.

Checks after translation (before any model run): 2,000 rows per language, ids identical to English, no empty strings, script check (Arabic/Devanagari/Latin), digit-sequence agreement with English, and cross-group duplicate detection (identical translations in different dependency groups are reported; groups are not re-merged).

## Post-translation log (3 October 2026)

- All 2,000 claims were translated label-blind in 20 chunks of 100. During translation, the translator fixed its own errors before any check: a gharial/alligator term (hi), an Arabic agreement error, two German word choices, and three unit conversions (pint → half-litre; miles → kilometres in de/fr/es), which were restored to the source units in line with rule 1.
- Automated checks: 2,000 rows, ids identical to English, 0 empty strings, 0 script errors, 0 digit mismatches. Every remaining metric unit also appears in the English source.
- Cross-group duplicate check: 9 cases where claims from different v2 dependency groups received identical translations (de 4, fr 2, ar 1, hi 1, es 1). They come from 7 English near-paraphrase pairs that the v2 grouping did not link. Groups were not re-merged (frozen design). Disclosure: this check printed the truth labels and partitions of these 7 pairs, after translation was complete. One pair showed a translation fidelity error: claim_0810 "Koala puppies are called joeys" had been rendered as "koala young/babies" in ar and es, which collapsed it with claim_1068 "Baby koalas are called joeys". claim_0810 was re-translated in all five languages to render "puppies" literally. No other translation was changed after labels were seen.

## LaBSE comparison with v2 (3 October 2026, computed on the server before any model result)
Same LaBSE revision (836121a), per dependency group (minimum over members), 1,972 groups:

| lang | mean v2 (NLLB) | mean v3 (LLM) | median paired diff | share v3 > v2 | 10th pct v2 | 10th pct v3 |
|---|---|---|---|---|---|---|
| de | 0.866 | 0.859 | -0.000 | 0.39 | 0.810 | 0.799 |
| ar | 0.855 | 0.848 | -0.005 | 0.44 | 0.768 | 0.753 |
| hi | 0.849 | 0.842 | -0.003 | 0.40 | 0.789 | 0.775 |
| fr | 0.888 | 0.877 | -0.002 | 0.30 | 0.838 | 0.816 |
| es | 0.897 | 0.887 | -0.001 | 0.30 | 0.851 | 0.830 |

v3 is slightly lower on LaBSE in every language, so LaBSE gives no evidence that v3 is better. The largest per-claim drops are cases where the NLLB output is literal but wrong or ungrammatical (e.g. fr "Les pingouins sont connus pour leurs roches romantiques" for "Penguins have been known to romance rocks"; de "Ein Hippopotamus gebären unter Wasser") and v3 is correct but less word-aligned. That pattern is consistent with LaBSE rewarding surface and cognate overlap rather than fidelity, but it is an observation on a handful of cases, not a measurement. Any claim that v3 is better must rest on human judgement (the author review), not on LaBSE.

Correction (3 October 2026): in the table above, "share v3 > v2" counted groups whose scores differ only by floating-point noise (identical or near-identical translations) as higher or lower. Recomputed on CPU with the same LaBSE revision (CPU means match the GPU values to about 1e-7), with |diff| ≤ 1e-4 counted as a tie. Share of groups where v3 is higher / tied / lower than v2: de 0.36/0.13/0.50, ar 0.44/0.01/0.55, hi 0.40/0.04/0.56, fr 0.29/0.17/0.54, es 0.27/0.20/0.53. Means and percentiles are unchanged. Script: study_truth_transport_v2/reporting/compare_translation_qe.py (the Google Translate comparison uses the same script).

## Three-way comparison with Google Translate (3 October 2026)
Google translations: the first author ran =GOOGLETRANSLATE in Google Sheets on the label-free English file and exported it (generic_claims_2000_gt/, export hash in SOURCE.json). All three systems were scored with the same LaBSE revision on the same CPU (study_truth_transport_v2/results/translation_qe_three_way.json).

| lang | LaBSE mean NLLB / LLM / Google | 10th pct NLLB / LLM / Google | Google higher/tied/lower vs NLLB |
|---|---|---|---|
| de | 0.866 / 0.859 / 0.855 | 0.810 / 0.799 / 0.790 | 0.34/0.13/0.53 |
| ar | 0.855 / 0.848 / 0.856 | 0.768 / 0.753 / 0.771 | 0.48/0.02/0.50 |
| hi | 0.849 / 0.842 / 0.833 | 0.789 / 0.775 / 0.757 | 0.33/0.05/0.62 |
| fr | 0.888 / 0.877 / 0.874 | 0.838 / 0.816 / 0.811 | 0.28/0.17/0.55 |
| es | 0.897 / 0.887 / 0.884 | 0.851 / 0.830 / 0.824 | 0.24/0.22/0.54 |

Number and unit fidelity (claims whose digits differ from the English; 42 English claims use imperial units):
- digit mismatches, NLLB / LLM / Google: de 5/0/5, ar 7/0/0, hi 2/0/0, fr 5/0/11, es 5/0/1;
- imperial-unit claims rendered with metric units, NLLB / LLM / Google: de 8/0/11, ar 1/0/1, hi 1/0/1, fr 11/0/18, es 3/0/13. NLLB typically swaps the unit but keeps the number, which changes the claim (e.g. "two and a half foot bubble" → "zweieinhalb Meter"; "three-inch nail" → "drei Zentimeter"; "five feet of snow" → "ein Meter"). Google converts the number approximately (e.g. "75 cm", "1,5 Meter"), which keeps the rough magnitude but not the stated figure.
- identical strings between systems (de/ar/hi/fr/es): NLLB-LLM 0.13/0.01/0.04/0.17/0.19; NLLB-Google 0.13/0.01/0.04/0.17/0.22; LLM-Google 0.34/0.10/0.08/0.34/0.36.
Reading: LaBSE ranks NLLB highest in four of five languages, even though NLLB has the most unit and meaning errors in these checks. Treat LaBSE as a measure of surface closeness to the English, not of fidelity. These are automatic checks; human judgement (the blind A/B task) remains the deciding evidence.
