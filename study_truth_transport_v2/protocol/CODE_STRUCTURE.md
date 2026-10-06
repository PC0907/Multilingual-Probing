# Code structure and data flow

## Modules

- `src/data.py`: aligned dataset loading, dependency-group indexing and group-mean activation aggregation.
- `src/allocation.py`: fixed-size, fixed-composition overlap allocations and validation.
- `src/probes.py`: mass-mean fitting, source midpoint threshold and scoring.
- `src/metrics.py`: AUROC, balanced accuracy, cosine and score-transport decomposition.
- `src/bootstrap.py`: stratified dependency-group bootstrap.
- `src/residualization.py`: one-direction and multi-direction linear removal.
- `src/steering.py`: direction normalization and target-scale-matched steering deltas. Model-specific hook runners will call this pure function.
- `src/prepare_rq_a.py`: freezes derived metadata and all pair-specific RQ-A allocations.

Planned runners keep model I/O separate from statistical analysis:

- `runners/extract_activations.py`: one-model/one-GPU all-layer extraction with resumable atomic caches;
- `runners/behavioral_judgments.py`: three-template answer-sequence likelihoods and statement surprisal;
- `runners/select_layer.py`: validation-only frozen primary layer;
- `runners/analyze_rq_a.py`: geometry, transfer, reliability and null controls;
- `runners/analyze_rq_b.py`: cross-fitted difficulty directions and residualization;
- `runners/analyze_rq_c.py`: score transport, calibration and prior-shift recentering;
- `runners/analyze_rq_d.py`: behavioral knowledge quadrants;
- `runners/run_rq_e.py`: model-specific intervention hooks and off-target checks.

Every runner writes a configuration signature, input hashes, model/tokenizer revision, completion state and output hashes. Existing output with a different signature is rejected rather than overwritten.

## Data flow

`dataset_v2 → rendered prompts → all-layer caches → validation-only layer freeze → frozen allocation plans → RQ-A → behavioral results → RQ-B/RQ-D → RQ-C → RQ-E`

Test data never feeds prompt design, layer selection, direction fitting, difficulty-direction fitting, recentering moments or steering normalization.
