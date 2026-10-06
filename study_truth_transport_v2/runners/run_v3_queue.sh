#!/usr/bin/env bash
# One-GPU rerun of every model-dependent analysis on the v3 (LLM-translated) claim set
# (generic_claims_2000_v3/TRANSLATION_PROTOCOL.md). Analysis code, dependency groups, split,
# allocations, prompt templates and external test data are identical to v2; only the five
# non-English claim texts change. Writes only inside RUN_ROOT; each model's checkpoint and
# caches are deleted as soon as its outputs are verified.
set -euo pipefail
if [[ $# -ne 4 ]]; then echo "usage: $0 RUN_ROOT PYTHON GPU_INDEX HF_TOKEN_PATH" >&2; exit 2; fi
run_root=$1; python_bin=$2; gpu=$3; token=$4
# LaBSE revision used for X11 in every dataset version (results_summary_v3/translation_qe_labse.revision.json).
LABSE_REVISION=${LABSE_REVISION:-836121a0533e5664b21c7aacc5d22951f2b8b25b}
export PROMPT_SET=prompts_v3
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
study="$run_root/study_truth_transport_v2"; r="$study/runners"
data="$study/data/$PROMPT_SET"
cd "$run_root"
mkdir -p "$run_root/results" "$run_root/logs"

# X11 input: LaBSE source-translation similarity on the v3 texts.
qe="$run_root/results/translation_qe_labse.json"
if [[ ! -f "$qe" ]]; then
  labse_rev=$LABSE_REVISION
  labse="$run_root/models/labse-${labse_rev:0:7}"
  [[ -d "$labse" ]] || "$python_bin" -m study_truth_transport_v2.runners.download_model \
      --repository sentence-transformers/LaBSE --revision "$labse_rev" --output "$labse" --minimum-free-gib 5 \
      --ignore-pattern '*.h5' --ignore-pattern '*.onnx' --ignore-pattern 'onnx/*' --ignore-pattern 'openvino/*'
  env CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m study_truth_transport_v2.runners.compute_translation_qe \
    --prompt-root "$data" --model "$labse" --output "$qe"
  printf '{"repository":"sentence-transformers/LaBSE","revision":"%s"}\n' "$labse_rev" > "$run_root/results/translation_qe_labse.revision.json"
  rm -rf "$labse"
fi

full_model() {
  # repo revision id batch tier min_free_gib [ignore patterns]; tier=core runs RQ-A..E, steering and X2.
  local repo=$1 rev=$2 id=$3 batch=$4 tier=$5 min_free=$6; shift 6
  local ignore=(); for p in "$@"; do ignore+=(--ignore-pattern "$p"); done
  local res="$run_root/results/$id" ext="$run_root/results/$id/extensions"
  local path="$run_root/models/${id}-${rev:0:7}" cache="$run_root/caches/$id"
  if [[ -f "$res/V3_COMPLETE" ]]; then echo "skip complete $id"; return 0; fi
  mkdir -p "$ext"
  echo "start $id $(date -u +%FT%TZ)"
  [[ -d "$path" ]] || HF_TOKEN_PATH="$token" "$python_bin" -m study_truth_transport_v2.runners.download_model \
      --repository "$repo" --revision "$rev" --output "$path" --minimum-free-gib "$min_free" "${ignore[@]}"
  if [[ ! -f "$res/CORE_COMPLETE" ]]; then
    if [[ $tier == core ]]; then
      "$r/run_complete_model.sh" "$run_root" "$python_bin" "$path" "$id" "$gpu" "$batch" "$batch"
    else
      "$r/run_core_replication.sh" "$run_root" "$python_bin" "$path" "$id" "$gpu" "$batch" "$batch"
    fi
    touch "$res/CORE_COMPLETE"
  fi
  # Core families: X1, LSI, MMMLU caches and X2 damage-budgeted steering need the checkpoint.
  if [[ $tier == core ]]; then
    "$r/run_extension_model.sh" "$run_root" "$python_bin" "$path" "$id" "$gpu" "$batch"
  else
    set +e; "$r/run_extension_model.sh" "$run_root" "$python_bin" "$run_root/models/absent-$id" "$id" "$gpu" "$batch"; st=$?; set -e
    [[ $st -eq 0 || $st -eq 3 ]] || exit "$st"
  fi
  # External test caches at this model's (possibly re-selected) frozen block.
  local block; block=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_block_number"])' "$res/layer_selection.json")
  for spec in "mmmlu:mmmlu_external_v1:en de ar hi fr es" "include:include_external_v1:ar de hi fr es"; do
    local short=${spec%%:*} rest=${spec#*:}; local name=${rest%%:*} langs=${rest#*:}
    for language in $langs; do
      local target="$ext/${short}_cache_fit512/$language"
      [[ -f "$target/COMPLETE" ]] && continue
      [[ -e "$target" ]] && { echo "refusing partial output: $target" >&2; exit 1; }
      flock -n /tmp/truth_transport_gpu.lock env CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" \
        -m study_truth_transport_v2.runners.extract_selected_layer \
        --input "$study/data/${name}_fit512/$id/$language.json" --model "$path" \
        --output "$target" --block-number "$block" --batch-size "$batch"
    done
  done
  rm -rf "$path"
  for spec in "mmmlu:mmmlu_external_v1:mmmlu_transfer.json:en de ar hi fr es:X3" "include:include_external_v1:include_transfer.json:ar de hi fr es:X6"; do
    IFS=: read -r short name output langs label <<<"$spec"
    [[ -f "$ext/$output" ]] && continue
    local extra=()
    if [[ $label == X6 ]]; then
      extra=(--breakdown-field regional_feature
             --analysis-label "post-core X6 zero-shot native-authored INCLUDE candidate-truth transfer"
             --external-data-label "CohereLabs/include-base-44 test (native-language exams); no fitting")
    fi
    # shellcheck disable=SC2086
    "$python_bin" -m study_truth_transport_v2.runners.analyze_mmmlu_transfer --model-id "$id" \
      --train-prompt-root "$data" --train-cache-root "$cache" --layer-selection "$res/layer_selection.json" \
      --external-prompt-root "$study/data/${name}_fit512/$id" --external-cache-root "$ext/${short}_cache_fit512" \
      --external-languages $langs --output "$ext/$output" --bootstrap-draws 1000 "${extra[@]}"
  done
  "$r/run_x7_analysis.sh" "$run_root" "$python_bin" "$id" "$gpu" --with-x1-methods
  local common=(--model-id "$id" --prompt-root "$data" --cache-root "$cache"
                --layer-selection "$res/layer_selection.json" --allocation-root "$study/data/${ALLOC_SET:-rq_a_allocations}")
  [[ -f "$ext/x10_x12.json" ]] || "$python_bin" -m study_truth_transport_v2.runners.analyze_x10_x12 "${common[@]}" \
      --rqa-summary "$res/rq_a/summary.json" --qe "$qe" --output "$ext/x10_x12.json"
  [[ -f "$ext/x13_probes.json" ]] || env CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -m study_truth_transport_v2.runners.analyze_x13_probes \
      "${common[@]}" --output "$ext/x13_probes.json"
  "$python_bin" - "$res" "$tier" <<'PY'
import json, sys
from pathlib import Path
root, tier = Path(sys.argv[1]), sys.argv[2]
need = {"rq_c.json": 36, "rq_b.json": None, "rq_a/inference.json": 150, "all_layers.json": None,
        "extensions/alignment_baselines.json": 30, "extensions/lsi_latent/inference.json": None,
        "extensions/mmmlu_transfer.json": 36, "extensions/include_transfer.json": 30,
        "extensions/transfer_predictors.json": None, "extensions/x10_x12.json": None, "extensions/x13_probes.json": None}
if tier == "core":
    need.update({"rq_d.json": None, "steering_summary.json": None, "extensions/damage_budgeted_steering.json": None})
for name, count in need.items():
    value = json.loads((root / name).read_text())
    if count is not None:
        assert len(value["cells"]) == count, name
print("verified v3 artifacts", root)
PY
  rm -rf "$cache" "$run_root/behavior/$id"
  touch "$res/V3_COMPLETE"
  echo "V3 complete $id $(date -u +%FT%TZ)"
}

# Core families first (the paper's primary evidence), then scale/post-training/open-data models.
full_model Qwen/Qwen3-8B-Base 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 qwen3-8b-base 16 core 25
full_model google/gemma-7b ff6768d9368919a1f025a54f9f5aa0ee591730bb gemma-7b 8 core 25 '*.gguf'
full_model mistralai/Mistral-7B-v0.3 caa1feb0e54d415e2df31207e5f4e273e33509b1 mistral-7b-v0.3 16 core 25 consolidated.safetensors
full_model swiss-ai/Apertus-8B-2509 3162c99675aa588097cecd4a24b9aa1f712af477 apertus-8b-2509 16 core 25
full_model Qwen/Qwen3-0.6B-Base da87bfb608c14b7cf20ba1ce41287e8de496c0cd qwen3-0.6b-base 16 scale 8
full_model Qwen/Qwen3-1.7B-Base ea980cb0a6c2ae4b936e82123acc929f1cec04c1 qwen3-1.7b-base 16 scale 10
full_model Qwen/Qwen3-4B-Base 906bfd4b4dc7f14ee4320094d8b41684abff8539 qwen3-4b-base 16 scale 15
full_model Qwen/Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218 qwen3-8b 16 scale 25
full_model allenai/OLMo-2-1124-7B 7df9a82518afdecae4e8c026b27adccc8c1f0032 olmo-2-1124-7b 16 scale 25
full_model Qwen/Qwen3-14B-Base 0b0bd3732e2c374d483664439ea334928b65f304 qwen3-14b-base 8 scale 40

# Secondary training-time set (RQ-F/X5 and X7): Pythia-1.4B-deduped checkpoints, same revisions as v2.
for spec in 0:ee49b763c5873072fda3e56443c77108f73d9913 1000:c55aeee0fcbae6ad588405867a416b850cd0d548 \
            4000:a643715ae75dc97ef7b1a7694f8995ceb96eb1cf 16000:6b146da0aa9f577be631521754dad50d2a6e9727 \
            64000:ace43ca627f244d37a4d9246bb6230fbacb43517 143000:6d1288ca1da05b700367a229ed2000de0eab8c4d; do
  step=${spec%%:*}; rev=${spec#*:}; id="pythia-1.4b-step${step}"
  res="$run_root/results/$id"; path="$run_root/models/$id"
  [[ -f "$res/V3_COMPLETE" ]] && continue
  mkdir -p "$res"
  printf '{"repository":"EleutherAI/pythia-1.4b-deduped","tag":"step%s","resolved_revision":"%s"}\n' "$step" "$rev" > "$res/checkpoint.json"
  [[ -d "$path" ]] || "$python_bin" -m study_truth_transport_v2.runners.download_model --repository EleutherAI/pythia-1.4b-deduped \
      --revision "$rev" --output "$path" --minimum-free-gib 12 --ignore-pattern optimizer.pt --ignore-pattern pytorch_model.bin
  "$r/run_single_model_extraction.sh" "$run_root" "$python_bin" "$path" "$id" "$gpu" 32
  rm -rf "$path"
  for step_mod in select_layer:"--output $res/layer_selection.json" analyze_rq_a:"--layer-selection $res/layer_selection.json --allocation-root $study/data/${ALLOC_SET:-rq_a_allocations} --output-dir $res/rq_a" \
                  analyze_rq_c:"--layer-selection $res/layer_selection.json --output $res/rq_c.json" analyze_all_layers:"--layer-selection $res/layer_selection.json --output $res/all_layers.json"; do
    mod=${step_mod%%:*}; args=${step_mod#*:}
    # shellcheck disable=SC2086
    "$python_bin" -m study_truth_transport_v2.runners.$mod --model-id "$id" --prompt-root "$data" --cache-root "$run_root/caches/$id" $args
  done
  "$r/run_x7_analysis.sh" "$run_root" "$python_bin" "$id" "$gpu"
  rm -rf "$run_root/caches/$id"
  touch "$res/RQF_COMPLETE" "$res/V3_COMPLETE"
  echo "V3 complete $id $(date -u +%FT%TZ)"
done
"$python_bin" -m study_truth_transport_v2.runners.analyze_rqf_trajectory \
  --results-root "$run_root/results" --output "$run_root/results/rq_f_pythia_trajectory.json"
echo "V3 queue complete $(date -u +%FT%TZ)"
