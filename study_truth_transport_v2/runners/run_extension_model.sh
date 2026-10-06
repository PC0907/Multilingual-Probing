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
result_root="$run_root/results/$model_id"
extension_root="$result_root/extensions"
mkdir -p "$extension_root"
cd "$run_root"

check_gpu() {
  "$run_root/study_truth_transport_v2/runners/gpu_ok.sh" "$gpu_index" extension || exit 1
}

if [[ ! -f "$extension_root/alignment_baselines.json" ]]; then
  check_gpu
  flock -n "${GPU_LOCK:-/tmp/truth_transport_gpu.lock}" env CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
    -m study_truth_transport_v2.runners.analyze_alignment_baselines \
    --model-id "$model_id" --prompt-root "$study_root/data/${PROMPT_SET:-prompts_v1}" \
    --cache-root "$run_root/caches/$model_id" --layer-selection "$result_root/layer_selection.json" \
    --output "$extension_root/alignment_baselines.json" --device cuda
fi

if [[ ! -f "$extension_root/lsi_latent/COMPLETE" ]]; then
  check_gpu
  flock -n "${GPU_LOCK:-/tmp/truth_transport_gpu.lock}" env CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
    -m study_truth_transport_v2.runners.run_lsi_latent_probe \
    --model-id "$model_id" --prompt-root "$study_root/data/${PROMPT_SET:-prompts_v1}" \
    --cache-root "$run_root/caches/$model_id" --layer-selection "$result_root/layer_selection.json" \
    --output-dir "$extension_root/lsi_latent" --batch-size 4096
fi

if [[ ! -f "$extension_root/lsi_latent/inference.json" ]]; then
  check_gpu
  flock -n "${GPU_LOCK:-/tmp/truth_transport_gpu.lock}" env CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
    -m study_truth_transport_v2.runners.analyze_lsi_inference \
    --model-id "$model_id" --prompt-root "$study_root/data/${PROMPT_SET:-prompts_v1}" \
    --cache-root "$run_root/caches/$model_id" --layer-selection "$result_root/layer_selection.json" \
    --lsi-dir "$extension_root/lsi_latent" --output "$extension_root/lsi_latent/inference.json" \
    --bootstrap-draws 1000 --batch-size 4096
fi

if [[ ! -d "$model_path" ]]; then
  echo "model checkpoint unavailable after cache-only extensions: $model_path" >&2
  exit 3
fi

# Preflight: every frozen-rule compacted record must fit under this model's
# own tokenizer before any MMMLU cache is written (DEVIATIONS.md, 2 Oct 2026).
"$python_bin" - "$study_root/data/mmmlu_external_v1_fit512/$model_id" "$model_path" <<'PY'
import json, sys
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
root, model = Path(sys.argv[1]), sys.argv[2]
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
limit = min(512, int(AutoConfig.from_pretrained(model, local_files_only=True).max_position_embeddings))
for lang in ("en", "de", "ar", "hi", "fr", "es"):
    rows = json.loads((root / f"{lang}.json").read_text(encoding="utf-8"))
    longest = max(len(ids) for ids in tokenizer([r["sentence"] for r in rows], add_special_tokens=True)["input_ids"])
    if longest > limit:
        sys.exit(f"MMMLU preflight failed: {lang} has {longest} > {limit} tokens")
print("MMMLU preflight passed", root)
PY

for language in en de ar hi fr es; do
  output="$extension_root/mmmlu_cache_fit512/$language"
  if [[ -f "$output/COMPLETE" ]]; then
    continue
  fi
  if [[ -e "$output" ]]; then
    # resume (DEVIATIONS.md 2026-10-04): an interrupted run's own partial output is removed and redone
    echo "removing own partial output: $output" >&2
    rm -rf "$output"
  fi
  check_gpu
  block=$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_block_number"])' \
    "$result_root/layer_selection.json")
  flock -n "${GPU_LOCK:-/tmp/truth_transport_gpu.lock}" env CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
    -m study_truth_transport_v2.runners.extract_selected_layer \
    --input "$study_root/data/mmmlu_external_v1_fit512/$model_id/$language.json" --model "$model_path" \
    --output "$output" --block-number "$block" --batch-size "$batch_size"
done

if [[ ! -f "$extension_root/mmmlu_transfer.json" ]]; then
  "$python_bin" -m study_truth_transport_v2.runners.analyze_mmmlu_transfer \
    --model-id "$model_id" --train-prompt-root "$study_root/data/${PROMPT_SET:-prompts_v1}" \
    --train-cache-root "$run_root/caches/$model_id" --layer-selection "$result_root/layer_selection.json" \
    --external-prompt-root "$study_root/data/mmmlu_external_v1_fit512/$model_id" \
    --external-cache-root "$extension_root/mmmlu_cache_fit512" \
    --output "$extension_root/mmmlu_transfer.json" --bootstrap-draws 1000
fi

if [[ ! -d "$result_root/steering_vectors" ]]; then
  "$python_bin" -m study_truth_transport_v2.runners.prepare_steering \
    --model-id "$model_id" --prompt-root "$study_root/data/${PROMPT_SET:-prompts_v1}" \
    --cache-root "$run_root/caches/$model_id" --layer-selection "$result_root/layer_selection.json" \
    --behavior "$result_root/behavior_aggregate.json" --allocation-root "$study_root/data/${ALLOC_SET:-rq_a_allocations}" \
    --output-dir "$result_root/steering_vectors"
fi

behavior_dir="$extension_root/steering_damage_grid"
if [[ ! -f "$behavior_dir/COMPLETE" ]]; then
  check_gpu
  flock -n "${GPU_LOCK:-/tmp/truth_transport_gpu.lock}" env CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
    -m study_truth_transport_v2.runners.run_steering \
    --model "$model_path" --prompt-root "$study_root/data/${PROMPT_SET:-prompts_v1}" \
    --steering-dir "$result_root/steering_vectors" --output "$behavior_dir" \
    --batch-size "$batch_size" --test-groups 100 \
    --direction-names zero_overlap_source target_native random_00 random_01 random_02 random_03 random_04 \
    --alphas -4 -2 -1 -0.5 -0.25 -0.125 0.125 0.25 0.5 1 2 4
fi

neutral_dir="$extension_root/neutral_continuation_damage_grid"
if [[ ! -f "$neutral_dir/COMPLETE" ]]; then
  check_gpu
  flock -n "${GPU_LOCK:-/tmp/truth_transport_gpu.lock}" env CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" \
    -m study_truth_transport_v2.runners.run_neutral_continuation_steering \
    --model "$model_path" --neutral-root "$study_root/data/flores_neutral_v1" \
    --steering-dir "$result_root/steering_vectors" --output "$neutral_dir" \
    --batch-size "$batch_size" \
    --direction-names zero_overlap_source target_native random_00 random_01 random_02 random_03 random_04 \
    --alphas -4 -2 -1 -0.5 -0.25 -0.125 0.125 0.25 0.5 1 2 4
fi

if [[ ! -f "$extension_root/damage_budgeted_steering.json" ]]; then
  "$python_bin" -m study_truth_transport_v2.runners.analyze_damage_budgeted_steering \
    --behavior-dir "$behavior_dir" --neutral-dir "$neutral_dir" \
    --output "$extension_root/damage_budgeted_steering.json" --damage-budget 0.05
fi

"$python_bin" - "$extension_root" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
required = [root / "alignment_baselines.json", root / "lsi_latent/results.json",
            root / "lsi_latent/inference.json",
            root / "mmmlu_transfer.json", root / "damage_budgeted_steering.json"]
for path in required:
    value = json.loads(path.read_text())
    assert value and path.stat().st_size > 100
for lang in ("en", "de", "ar", "hi", "fr", "es"):
    assert (root / "mmmlu_cache_fit512" / lang / "COMPLETE").exists()
print("verified extension artifacts", root)
PY

echo "extension complete $model_id $(date -u +%FT%TZ)"
