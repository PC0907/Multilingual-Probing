# Multilingual truth transport (v3 claims)

Code and data for a study of how linear "truth directions" learned in one language transfer to another, using
mass-mean probes on 2,000 parallel true/false claims in English, German, Arabic, Hindi, French and Spanish, across
10 open language models and 6 Pythia training checkpoints.

The study's questions:
- **RQ-A:** does sharing training facts between two languages inflate the cosine between their truth directions,
  without improving held-out transfer?
- **RQ-B:** how much of the direction is item difficulty?
- **RQ-C:** does ranking transfer, or does threshold transport fail?
- **RQ-D:** behavioural knowledge quadrants.
- **RQ-E:** causal steering.
- **X1 to X14:** extensions, including alignment baselines, external tests on MMMLU and INCLUDE, and which geometry
  statistic tracks transfer.

The hypotheses and frozen decision rules are in `study_truth_transport_v2/protocol/`.

---

## Quick start

```bash
# 0. environment (Python 3.9, one CUDA GPU with 40 GB)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. unpack the frozen overlap allocations
gunzip -k study_truth_transport_v2/data/rq_a_allocations/*.json.gz

# 2. build the model prompts from the v3 claims
python study_truth_transport_v2/runners/prepare_prompts.py --dataset-root generic_claims_2000_v3 \
  --prompt-config study_truth_transport_v2/config/prompts.json --output study_truth_transport_v2/data/prompts_v3

# 3. check everything is in place (no GPU needed)
python -m pytest -q study_truth_transport_v2/tests

# 4. run all models on GPU 0 (needs a Hugging Face token file; see "Model access")
bash study_truth_transport_v2/runners/run_v3_queue.sh "$PWD" "$(which python)" 0 ~/.secrets/hf_token
```

Results are written to `results/<model_id>/` as JSON.

---

## Repository layout

```
generic_claims_2000_v3/          the claims: en/de/ar/hi/fr/es.json, dependency groups, split, translation protocol
study_truth_transport_v2/        the study package (the name is historical)
  config/                        model revisions, prompt templates, study settings
  src/                           core library: allocation, probes, bootstrap, metrics, residualization, steering
  runners/                       experiment and analysis entry points; run_v3_queue.sh drives everything
  reporting/                     figures, PDF report, paper numbers and results page, built from result JSON
  protocol/                      preregistrations and the full deviations log
  schemas/                       activation-cache and behavioural-result schemas
  tests/                         pytest suite
  data/derived_group_metadata.json        frozen group metadata
  data/rq_a_allocations/*.json.gz          frozen overlap allocations (unpack in step 1)
  data/rq_a_allocation_manifest.json       SHA-256 hashes of the allocations
results_summary_v3/              cross-model summaries of the reported run (X7, X10 to X13, X14, Pythia, translation checks)
scripts/check_before_push.sh     privacy scan to run before every push
```

The non-English claims were machine-translated by an LLM (Claude Opus 5.5) under the frozen, label-blind protocol in
`generic_claims_2000_v3/TRANSLATION_PROTOCOL.md`. They have not been reviewed by bilingual annotators.

---

## Step by step

### Step 0: Requirements
- Python 3.9 (the reported runs used 3.9.7) and the exact package versions in `requirements.txt`.
- One NVIDIA GPU with 40 GB. Smaller GPUs work for models up to about 4B parameters.
- About 50 GB of free disk at the peak, for the largest model (Qwen3-14B: checkpoint plus activation cache).
- No GPU is needed for steps 1 to 3.

### Step 1: Unpack the overlap allocations
```bash
gunzip -k study_truth_transport_v2/data/rq_a_allocations/*.json.gz
```
These files fix which training facts each language pair shares at each overlap level (0, 25, 50, 75, 100%; 50
random allocations per level). They were frozen before any experiment and cannot be regenerated from the v3 claims
alone, so they are shipped. To verify them, compare
`shasum -a 256 study_truth_transport_v2/data/rq_a_allocations/*.json` with
`study_truth_transport_v2/data/rq_a_allocation_manifest.json`.

### Step 2: Build the prompts
```bash
python study_truth_transport_v2/runners/prepare_prompts.py --dataset-root generic_claims_2000_v3 \
  --prompt-config study_truth_transport_v2/config/prompts.json --output study_truth_transport_v2/data/prompts_v3
```
This writes `data/prompts_v3/<lang>.activation.json` (probe inputs) and `<lang>.behavior.json` (yes/no judgment
prompts). The step is deterministic and reproduces the study's prompt files byte for byte.

### Step 3: Run the tests
```bash
python -m pytest -q study_truth_transport_v2/tests
```
Expected: all tests pass. Tests that validate the external data (step 5) or need model tokenizers are skipped until
those exist.

### Step 4: Model access
The driver downloads each model from Hugging Face at a pinned revision (see `config/model_revisions.json` and
`runners/run_v3_queue.sh`). Some models, such as Gemma, require accepting a licence on Hugging Face first.

1. Create a read token at https://huggingface.co/settings/tokens.
2. Save it in a private file outside this repository, e.g. `~/.secrets/hf_token`, and run `chmod 600 ~/.secrets/hf_token`.
3. Pass that file's path as the driver's last argument. The token is read from that file and never written into the
   repository.

### Step 5: External test data (optional; not distributed)
The external evaluations use third-party datasets under their own licences. Download them yourself, then build the
inputs; see each script's `--help` for its arguments.

| Data | Source | Script | Output expected by the runners |
|---|---|---|---|
| MMLU (English) and MMMLU | `cais/mmlu`, `openai/MMMLU` | `runners/prepare_mmmlu_external.py` | `data/mmmlu_external_v1/` |
| INCLUDE | `CohereLabs/include-base-44` | `runners/prepare_include_external.py` | `data/include_external_v1/` |
| Context fit per model (512 tokens) | the two above | `runners/compact_mmmlu_context.py` | `data/{mmmlu,include}_external_v1_fit512/<model_id>/` |
| FLORES-200 neutral text | https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz | `runners/prepare_neutral_texts.py` | `data/flores_neutral_v1/` |

Which analyses need it:
- The core analyses (RQ-A to RQ-C, X1, X7, X10 to X14) need none of this.
- Steering (RQ-E, X2) needs FLORES.
- The external tests X3 and X6 need MMMLU and INCLUDE.

### Step 6: Run the experiments
```bash
bash study_truth_transport_v2/runners/run_v3_queue.sh RUN_ROOT PYTHON GPU_INDEX HF_TOKEN_FILE
# example
bash study_truth_transport_v2/runners/run_v3_queue.sh "$PWD" "$(which python)" 0 ~/.secrets/hf_token
```
- `RUN_ROOT` is the folder that contains `study_truth_transport_v2/`, normally the repository root. Models, activation
  caches and results are written under it, in `models/`, `caches/` and `results/`; all three are git-ignored.
- For each model the driver:
  1. downloads it;
  2. extracts activations for all six languages;
  3. selects the layer on validation data;
  4. runs every analysis;
  5. verifies the outputs and writes `results/<model_id>/V3_COMPLETE`;
  6. deletes that model's checkpoint and cache.
- Runtime is roughly 2 to 4 hours per 7 to 8B model, much less for small models.
- The driver can be restarted: completed models are skipped, and within a model, finished steps are skipped.
- It runs on one GPU. Before each GPU step it checks that the GPU is idle and stops rather than share it. Use another
  `GPU_INDEX` if that one is busy.

To run a single extra analysis on an existing model, call the module directly; every runner documents its arguments:
```bash
python -m study_truth_transport_v2.runners.analyze_rq_c --help
```

### Step 7: Cross-model summaries
```bash
M="qwen3-0.6b-base qwen3-1.7b-base qwen3-4b-base qwen3-8b-base qwen3-14b-base qwen3-8b olmo-2-1124-7b gemma-7b apertus-8b-2509 mistral-7b-v0.3"
python -m study_truth_transport_v2.runners.analyze_x7_meta --results-root results --primary-models $M --output results/x7_meta.json
python -m study_truth_transport_v2.runners.analyze_x10_x13_meta --results-root results --models $M --output results/x10_x13_meta.json
python -m study_truth_transport_v2.runners.analyze_x14_meta --results-root results --models $M --output results/x14_meta.json
```
Compare your output with `results_summary_v3/`, which holds the same summaries from the reported run.

### Step 8: Figures and report (optional)
```bash
MPLBACKEND=Agg python study_truth_transport_v2/reporting/make_figures.py --results-root results --output-dir figures
MPLBACKEND=Agg python study_truth_transport_v2/reporting/make_extension_figures.py --results-root results --output-dir figures
RESULTS_ROOT="$PWD/results" PDF_PYTHON="$(which python)" bash study_truth_transport_v2/reporting/build_final.sh
```
The X14 and replication figures and the report's comparison pages also read the translation-robustness and
replication runs, which are not part of this repository. Without them those parts are skipped or fail.




## Licence
Add a licence file before making the repository public. The external datasets keep their own licences.
