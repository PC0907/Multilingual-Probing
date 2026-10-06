#!/usr/bin/env bash
# X7 analysis on a verified cache (protocol/X7_X9_PREREGISTRATION.md), followed by X14
# (protocol/X14_PREREGISTRATION.md) while the same cache is available.
set -euo pipefail
if [[ $# -lt 4 ]]; then echo "usage: $0 RUN_ROOT PYTHON MODEL_ID GPU_INDEX [--with-x1-methods]" >&2; exit 2; fi
run_root=$1; python_bin=$2; model_id=$3; gpu_index=$4; shift 4
study_root="$run_root/study_truth_transport_v2"
result_root="$run_root/results/$model_id"
output="$result_root/extensions/transfer_predictors.json"
mkdir -p "$result_root/extensions"
if [[ ! -f "$output" ]]; then
  # GPU only if gpu_ok.sh passes (no other user's process on it); otherwise CPU.
  x7_dev="$gpu_index"; "$run_root/study_truth_transport_v2/runners/gpu_ok.sh" "$gpu_index" x7 2>/dev/null || x7_dev=""
  env CUDA_VISIBLE_DEVICES="$x7_dev" "$python_bin" -m study_truth_transport_v2.runners.analyze_transfer_predictors \
    --model-id "$model_id" --prompt-root "$study_root/data/${PROMPT_SET:-prompts_v1}" \
    --cache-root "$run_root/caches/$model_id" --layer-selection "$result_root/layer_selection.json" \
    --allocation-root "$study_root/data/${ALLOC_SET:-rq_a_allocations}" --output "$output" --bootstrap-draws 500 "$@"
fi
x14="$result_root/extensions/x14_mcs.json"
if [[ ! -f "$x14" ]]; then
  "$python_bin" -m study_truth_transport_v2.runners.analyze_x14_mcs \
    --model-id "$model_id" --prompt-root "$study_root/data/${PROMPT_SET:-prompts_v1}" \
    --cache-root "$run_root/caches/$model_id" --layer-selection "$result_root/layer_selection.json" \
    --allocation-root "$study_root/data/${ALLOC_SET:-rq_a_allocations}" --output "$x14"
fi
