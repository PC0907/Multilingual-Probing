# X14: Mahalanobis-cosine baselines, calibrated prediction, and an outlier-dimension failure analysis

Status: frozen on 3 October 2026, before any X14 output existed, on the v3 (primary) claim set. The X7 statistics (cosine, Fisher efficiency) were already known for the v2 set and for the four v3 core families; X14 adds new statistics and tests and does not change X7.

Motivation. Fisher efficiency F(w) = w'Δ / sqrt(w'Σ_pool w · Δ'Σ_pool⁻¹Δ) equals the within-class Mahalanobis cosine between w and the target's optimal discriminant Σ_pool⁻¹Δ (Ying, Hase & Kriegeskorte 2026). They recommend the total-covariance version (MCS_tot) and report that it predicts out-of-distribution AUROC linearly, with a near-universal slope. The paper currently claims that no statistic is calibrated across models; that claim must be tested against MCS_tot first.

Configurations. Exactly the X7 configurations: every decoder block × 30 ordered pairs, source mass-mean probe from the training partition, outcome = AUROC on target test groups.

Statistics per configuration (target quantities from the target training partition only):
- P1 Euclidean cosine (as in X7);
- P3 Fisher efficiency F (as in X7; Ledoit-Wolf-shrunk Σ_pool);
- P4a MCS_tot(w_s, w_t): Mahalanobis cosine under the target's total sample covariance, against the target mass-mean probe;
- P4b MCS_tot(w_s, Σ_pool⁻¹Δ_t): the same, against the target's optimal (shrinkage-LDA) discriminant, which is Ying et al.'s reference;
- P5 parameter-free Proposition-1 prediction: AUROC_hat = Φ(F · d*_t / √2), where d*_t = sqrt(Δ'Σ_pool⁻¹Δ) is estimated on target training data.

Outcomes and decision rules:
- H14a (within-model ranking): Spearman ρ with AUROC per model for P1, P3, P4a, P4b, all blocks and frozen block. Reported descriptively. The claim "Fisher efficiency tracks transfer better than cosine" stays only if ρ(P3) > ρ(P1) in at least 8 of 10 models. If ρ(P4b) ≥ ρ(P3) in at least 8 of 10 models, the paper presents MCS_tot as the recommended diagnostic.
- H14b (cross-model calibration): leave-one-model-out R² of a linear fit AUROC ~ statistic (fit on 9 models, predict the held-out one, pooled SSE over models), for P1, P3, P4a, P4b. A statistic counts as calibrated across models if its LOMO R² ≥ 0.5. Also reported: the pooled OLS slope for P4b (Ying et al. report about 0.56).
- H14c (Proposition 1 as a calibrated, parameter-free predictor): pooled R² of AUROC_hat against AUROC with no fitting (1 − SSE/SST) and mean absolute error. Calibrated if R² ≥ 0.5 and MAE ≤ 0.05.
- If any of P4a, P4b or P5 is calibrated, the paper's sentence "no statistic is calibrated across models" is withdrawn and replaced by the result.

Outlier-dimension failure analysis (H14d). Label-free definition at each model's frozen block: a hidden dimension is an outlier if its variance, pooled over the training partitions of all six languages, exceeds 20 times the median dimension variance. On features with the outlier dimensions removed (all models; zero dimensions removed means identical results), recompute (i) Spearman ρ of cosine and of Fisher efficiency with AUROC at the frozen block, and (ii) the X10 estimator-noise prediction: predicted versus observed overlap slope of cosine, observed from 10 RQ-A allocations per level re-fitted on the reduced features, for raw and reduced features alike. Prediction: in Apertus, removing the outlier dimensions raises the within-model X10 slope r above 0.5 and raises ρ(cosine). Supported if both hold in Apertus. Other models are controls, expected to change little.

Implementation: runners/analyze_x14_mcs.py (per model, needs that model's activation cache) and runners/analyze_x14_meta.py (cross-model). Models whose cache has already been deleted get it regenerated with runners/regenerate_and_verify_cache.sh, which requires exact reproduction of stored RQ-C before any X14 output is written.
