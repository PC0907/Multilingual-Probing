#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 RUN_ROOT PYTHON MODEL_PATH MODEL_ID GPU_INDEX BATCH_SIZE" >&2
  exit 2
fi

run_root=$1
python_bin=$2
model_path=$3
model_id=$4
gpu_index=$5
batch_size=$6
study_root="$run_root/study_truth_transport_v2"
output_root="$run_root/behavior/$model_id"
log_root="$run_root/logs/${model_id}-behavior"
mkdir -p "$output_root" "$log_root"

for language in en de ar hi fr es; do
  output="$output_root/$language"
  if [[ -f "$output/COMPLETE" ]]; then
    echo "skip complete behavior $model_id $language"
    continue
  fi
  if [[ -e "$output" ]]; then
    # resume (DEVIATIONS.md 2026-10-04): an interrupted run's own partial output is removed and redone
    echo "removing own partial output: $output" >&2
    rm -rf "$output"
  fi
  "$run_root/study_truth_transport_v2/runners/gpu_ok.sh" "$gpu_index" "behavior $language" || exit 1
  echo "start behavior $model_id $language $(date -u +%FT%TZ)"
  flock -n "${GPU_LOCK:-/tmp/truth_transport_gpu.lock}" env CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
    "$study_root/runners/score_behavior.py" \
    --input "$study_root/data/${PROMPT_SET:-prompts_v1}/$language.behavior.json" \
    --model "$model_path" \
    --output "$output" \
    --batch-size "$batch_size" 2>&1 | tee "$log_root/$language.log"
  echo "finish behavior $model_id $language $(date -u +%FT%TZ)"
done

echo "all behavioral scores complete for $model_id $(date -u +%FT%TZ)"
