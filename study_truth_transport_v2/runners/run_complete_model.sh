#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 RUN_ROOT PYTHON MODEL_PATH MODEL_ID GPU_INDEX EXTRACTION_BATCH BEHAVIOR_BATCH" >&2
  exit 2
fi

run_root=$1
python_bin=$2
model_path=$3
model_id=$4
gpu_index=$5
extraction_batch=$6
behavior_batch=$7
study_root="$run_root/study_truth_transport_v2"
result_root="$run_root/results/$model_id"
log_root="$run_root/logs"
mkdir -p "$result_root" "$log_root"
cd "$run_root"

# Resume support (DEVIATIONS.md, 2026-10-04 v5 resume entry): a step whose final output exists is skipped,
# so a queue stopped by a busy GPU can restart; an interrupted step's own partial directory is removed first.
todo() { [[ ! -e "$1" ]]; }
clean() { if [[ "$1" == "$result_root"/* ]]; then rm -rf "$1"; fi; }
gpu_idle() {
  "$run_root/study_truth_transport_v2/runners/gpu_ok.sh" "$gpu_index" "$1" || exit 1
}

"$study_root/runners/run_single_model_extraction.sh" \
  "$run_root" "$python_bin" "$model_path" "$model_id" "$gpu_index" "$extraction_batch"

if todo "$result_root/layer_selection.json"; then
  "$python_bin" -m study_truth_transport_v2.runners.select_layer \
    --model-id "$model_id" \
    --prompt-root "$study_root/data/${PROMPT_SET:-prompts_v1}" \
    --cache-root "$run_root/caches/$model_id" \
    --output "$result_root/layer_selection.json"
fi

if todo "$result_root/rq_a/summary.json"; then
  clean "$result_root/rq_a"
  "$python_bin" -m study_truth_transport_v2.runners.analyze_rq_a \
    --model-id "$model_id" \
    --prompt-root "$study_root/data/${PROMPT_SET:-prompts_v1}" \
    --cache-root "$run_root/caches/$model_id" \
    --layer-selection "$result_root/layer_selection.json" \
    --allocation-root "$study_root/data/${ALLOC_SET:-rq_a_allocations}" \
    --output-dir "$result_root/rq_a"
fi

if todo "$result_root/rq_a/inference.json"; then
  "$python_bin" -m study_truth_transport_v2.runners.analyze_rq_a_inference \
    --model-id "$model_id" \
    --prompt-root "$study_root/data/${PROMPT_SET:-prompts_v1}" \
    --cache-root "$run_root/caches/$model_id" \
    --layer-selection "$result_root/layer_selection.json" \
    --allocation-root "$study_root/data/${ALLOC_SET:-rq_a_allocations}" \
    --allocation-results "$result_root/rq_a/allocation_results.jsonl" \
    --output "$result_root/rq_a/inference.json" \
    --bootstrap-draws 1000 --permutation-draws 10000
fi

if todo "$result_root/rq_c.json"; then
  "$python_bin" -m study_truth_transport_v2.runners.analyze_rq_c \
    --model-id "$model_id" \
    --prompt-root "$study_root/data/${PROMPT_SET:-prompts_v1}" \
    --cache-root "$run_root/caches/$model_id" \
    --layer-selection "$result_root/layer_selection.json" \
    --output "$result_root/rq_c.json"
fi

"$study_root/runners/run_single_model_behavior.sh" \
  "$run_root" "$python_bin" "$model_path" "$model_id" "$gpu_index" "$behavior_batch"

if todo "$result_root/behavior_aggregate.json"; then
  "$python_bin" -m study_truth_transport_v2.runners.aggregate_behavior \
    --model-id "$model_id" \
    --behavior-root "$run_root/behavior/$model_id" \
    --output "$result_root/behavior_aggregate.json"
fi

if todo "$result_root/rq_b.json"; then
  "$python_bin" -m study_truth_transport_v2.runners.analyze_rq_b \
    --model-id "$model_id" \
    --prompt-root "$study_root/data/${PROMPT_SET:-prompts_v1}" \
    --cache-root "$run_root/caches/$model_id" \
    --layer-selection "$result_root/layer_selection.json" \
    --behavior "$result_root/behavior_aggregate.json" \
    --allocation-root "$study_root/data/${ALLOC_SET:-rq_a_allocations}" \
    --output "$result_root/rq_b.json"
fi

if todo "$result_root/rq_b_sensitivities.json"; then
  "$python_bin" -m study_truth_transport_v2.runners.analyze_rq_b_sensitivities \
    --model-id "$model_id" \
    --prompt-root "$study_root/data/${PROMPT_SET:-prompts_v1}" \
    --cache-root "$run_root/caches/$model_id" \
    --layer-selection "$result_root/layer_selection.json" \
    --behavior "$result_root/behavior_aggregate.json" \
    --output "$result_root/rq_b_sensitivities.json"
fi

if todo "$result_root/rq_d.json"; then
  "$python_bin" -m study_truth_transport_v2.runners.analyze_rq_d \
    --model-id "$model_id" \
    --prompt-root "$study_root/data/${PROMPT_SET:-prompts_v1}" \
    --cache-root "$run_root/caches/$model_id" \
    --layer-selection "$result_root/layer_selection.json" \
    --behavior "$result_root/behavior_aggregate.json" \
    --allocation-root "$study_root/data/${ALLOC_SET:-rq_a_allocations}" \
    --output "$result_root/rq_d.json"
fi

if todo "$result_root/rq_a_controls.json"; then
  "$python_bin" -m study_truth_transport_v2.runners.analyze_rq_a_controls \
    --model-id "$model_id" \
    --prompt-root "$study_root/data/${PROMPT_SET:-prompts_v1}" \
    --cache-root "$run_root/caches/$model_id" \
    --layer-selection "$result_root/layer_selection.json" \
    --allocation-root "$study_root/data/${ALLOC_SET:-rq_a_allocations}" \
    --output "$result_root/rq_a_controls.json" \
    --random-draws 1000 --permutation-draws 200
fi

if todo "$result_root/all_layers.json"; then
  "$python_bin" -m study_truth_transport_v2.runners.analyze_all_layers \
    --model-id "$model_id" \
    --prompt-root "$study_root/data/${PROMPT_SET:-prompts_v1}" \
    --cache-root "$run_root/caches/$model_id" \
    --layer-selection "$result_root/layer_selection.json" \
    --output "$result_root/all_layers.json"
fi

if todo "$result_root/steering_vectors/manifest.json"; then
  clean "$result_root/steering_vectors"
  "$python_bin" -m study_truth_transport_v2.runners.prepare_steering \
    --model-id "$model_id" \
    --prompt-root "$study_root/data/${PROMPT_SET:-prompts_v1}" \
    --cache-root "$run_root/caches/$model_id" \
    --layer-selection "$result_root/layer_selection.json" \
    --behavior "$result_root/behavior_aggregate.json" \
    --allocation-root "$study_root/data/${ALLOC_SET:-rq_a_allocations}" \
    --output-dir "$result_root/steering_vectors"
fi

if todo "$result_root/steering_full/COMPLETE"; then
  clean "$result_root/steering_full"
  gpu_idle steering
  flock -n "${GPU_LOCK:-/tmp/truth_transport_gpu.lock}" env CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
    -m study_truth_transport_v2.runners.run_steering \
    --model "$model_path" \
    --prompt-root "$study_root/data/${PROMPT_SET:-prompts_v1}" \
    --steering-dir "$result_root/steering_vectors" \
    --output "$result_root/steering_full" \
    --batch-size 32 --test-groups 100
fi

if todo "$result_root/steering_summary.json"; then
  "$python_bin" -m study_truth_transport_v2.runners.analyze_steering \
    --steering-output "$result_root/steering_full" \
    --output "$result_root/steering_summary.json" \
    --bootstrap-draws 1000
fi

if [[ -z "${SKIP_TRANSLATION_SENSITIVITY:-}" ]]; then
"$python_bin" -m study_truth_transport_v2.runners.analyze_translation_sensitivity \
  --model-id "$model_id" \
  --prompt-root "$study_root/data/${PROMPT_SET:-prompts_v1}" \
  --cache-root "$run_root/caches/$model_id" \
  --layer-selection "$result_root/layer_selection.json" \
  --behavior "$result_root/behavior_aggregate.json" \
  --allocation-root "$study_root/data/${ALLOC_SET:-rq_a_allocations}" \
  --group-metadata "$study_root/data/derived_group_metadata.json" \
  --output "$result_root/translation_sensitivity.json"
fi

"$study_root/runners/run_steering_sensitivities.sh" \
  "$run_root" "$python_bin" "$model_path" "$model_id" "$gpu_index"

echo "complete model pipeline $model_id $(date -u +%FT%TZ)"
