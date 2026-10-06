#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 RUN_ROOT PYTHON MODEL_PATH MODEL_ID GPU_INDEX" >&2
  exit 2
fi

run_root=$1
python_bin=$2
model_path=$3
model_id=$4
gpu_index=$5
study_root="$run_root/study_truth_transport_v2"
result_root="$run_root/results/$model_id"
cd "$run_root"

direction_names=(zero_overlap_source full_overlap_source target_native difficulty language_identity \
                 random_00 random_01 random_02 random_03 random_04)

for specification in "layer_minus_2:-2:final" "layer_plus_2:2:final" "all_statement_tokens:0:all-statement"; do
  IFS=: read -r name offset scope <<< "$specification"
  output="$result_root/steering_sensitivity_$name"
  # Resume support (DEVIATIONS.md, 2026-10-04 v5 resume entry): skip finished outputs, remove own partial ones.
  if [[ ! -f "$output/COMPLETE" ]]; then
  [[ "$output" == "$result_root"/* ]] && rm -rf "$output"
  "$run_root/study_truth_transport_v2/runners/gpu_ok.sh" "$gpu_index" "$name" || exit 1
  flock -n "${GPU_LOCK:-/tmp/truth_transport_gpu.lock}" env CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
    -m study_truth_transport_v2.runners.run_steering \
    --model "$model_path" \
    --prompt-root "$study_root/data/${PROMPT_SET:-prompts_v1}" \
    --steering-dir "$result_root/steering_vectors" \
    --output "$output" \
    --batch-size 32 --test-groups 50 \
    --block-offset "$offset" --token-scope "$scope" \
    --direction-names "${direction_names[@]}"
  fi
  [[ -f "$result_root/steering_sensitivity_${name}_summary.json" ]] || "$python_bin" -m study_truth_transport_v2.runners.analyze_steering \
    --steering-output "$output" \
    --output "$result_root/steering_sensitivity_${name}_summary.json" \
    --bootstrap-draws 1000
done

neutral_output="$result_root/steering_neutral_flores"
if [[ ! -f "$neutral_output/COMPLETE" ]]; then
[[ "$neutral_output" == "$result_root"/* ]] && rm -rf "$neutral_output"
"$run_root/study_truth_transport_v2/runners/gpu_ok.sh" "$gpu_index" "neutral control" || exit 1
flock -n "${GPU_LOCK:-/tmp/truth_transport_gpu.lock}" env CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
  -m study_truth_transport_v2.runners.run_neutral_steering \
  --model "$model_path" \
  --neutral-root "$study_root/data/flores_neutral_v1" \
  --steering-dir "$result_root/steering_vectors" \
  --output "$neutral_output" \
  --batch-size 16
fi

echo "complete steering sensitivities $model_id $(date -u +%FT%TZ)"
