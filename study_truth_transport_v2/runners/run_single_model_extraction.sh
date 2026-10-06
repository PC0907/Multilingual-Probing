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
cache_root="$run_root/caches/$model_id"
log_root="$run_root/logs/$model_id"
mkdir -p "$cache_root" "$log_root"

if [[ ! -d "$model_path" ]]; then
  echo "missing model: $model_path" >&2
  exit 1
fi
"$run_root/study_truth_transport_v2/runners/gpu_ok.sh" "$gpu_index" extraction || exit 1

for language in en de ar hi fr es; do
  output="$cache_root/$language"
  if [[ -f "$output/COMPLETE" ]]; then
    echo "skip complete $model_id $language"
    continue
  fi
  if [[ -e "$output" ]]; then
    # resume (DEVIATIONS.md 2026-10-04): an interrupted run's own partial output is removed and redone
    echo "removing own partial output: $output" >&2
    rm -rf "$output"
  fi
  echo "start $model_id $language $(date -u +%FT%TZ)"
  flock -n "${GPU_LOCK:-/tmp/truth_transport_gpu.lock}" env CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
    "$study_root/runners/extract_prompted_states.py" \
    --input "$study_root/data/${PROMPT_SET:-prompts_v1}/$language.activation.json" \
    --model "$model_path" \
    --output "$output" \
    --batch-size "$batch_size" \
    --hash-caches 2>&1 | tee "$log_root/$language.log"
  echo "finish $model_id $language $(date -u +%FT%TZ)"
done

echo "all extraction caches complete for $model_id $(date -u +%FT%TZ)"
