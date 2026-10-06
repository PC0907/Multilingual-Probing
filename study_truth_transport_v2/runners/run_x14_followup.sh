#!/usr/bin/env bash
# X14 for models whose v3 activation cache was already deleted (protocol/X14_PREREGISTRATION.md).
# Regenerates the cache (must reproduce stored RQ-C exactly), runs X14, deletes checkpoint and cache.
set -euo pipefail
if [[ $# -ne 4 ]]; then echo "usage: $0 RUN_ROOT PYTHON GPU_INDEX HF_TOKEN_PATH" >&2; exit 2; fi
run_root=$1; python_bin=$2; gpu=$3; token=$4
export PROMPT_SET=prompts_v3 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
study="$run_root/study_truth_transport_v2"; r="$study/runners"
cd "$run_root"
x14_model() {
  local repo=$1 rev=$2 id=$3 batch=$4; shift 4
  local res="$run_root/results/$id" path="$run_root/models/${id}-${rev:0:7}"
  [[ -f "$res/extensions/x14_mcs.json" ]] && { echo "skip $id"; return 0; }
  [[ -f "$res/layer_selection.json" ]] || { echo "no v3 results for $id" >&2; return 1; }
  if [[ ! -f "$run_root/caches/$id/VERIFIED" ]]; then
    [[ -d "$path" ]] || HF_TOKEN_PATH="$token" "$python_bin" -m study_truth_transport_v2.runners.download_model \
      --repository "$repo" --revision "$rev" --output "$path" --minimum-free-gib 10 "$@"
    "$r/regenerate_and_verify_cache.sh" "$run_root" "$python_bin" "$path" "$id" "$gpu" "$batch"
  fi
  rm -rf "$path"
  "$python_bin" -m study_truth_transport_v2.runners.analyze_x14_mcs --model-id "$id" --prompt-root "$study/data/$PROMPT_SET" \
    --cache-root "$run_root/caches/$id" --layer-selection "$res/layer_selection.json" \
    --allocation-root "$study/data/${ALLOC_SET:-rq_a_allocations}" --output "$res/extensions/x14_mcs.json"
  rm -rf "$run_root/caches/$id"
  echo "X14 complete $id $(date -u +%FT%TZ)"
}
x14_model Qwen/Qwen3-8B-Base 49e3418fbbbca6ecbdf9608b4d22e5a407081db4 qwen3-8b-base 16
x14_model google/gemma-7b ff6768d9368919a1f025a54f9f5aa0ee591730bb gemma-7b 8 --ignore-pattern '*.gguf'
x14_model mistralai/Mistral-7B-v0.3 caa1feb0e54d415e2df31207e5f4e273e33509b1 mistral-7b-v0.3 16 --ignore-pattern consolidated.safetensors
x14_model swiss-ai/Apertus-8B-2509 3162c99675aa588097cecd4a24b9aa1f712af477 apertus-8b-2509 16
x14_model Qwen/Qwen3-0.6B-Base da87bfb608c14b7cf20ba1ce41287e8de496c0cd qwen3-0.6b-base 16
x14_model Qwen/Qwen3-1.7B-Base ea980cb0a6c2ae4b936e82123acc929f1cec04c1 qwen3-1.7b-base 16
x14_model Qwen/Qwen3-4B-Base 906bfd4b4dc7f14ee4320094d8b41684abff8539 qwen3-4b-base 16
x14_model Qwen/Qwen3-8B b968826d9c46dd6066d109eabc6255188de91218 qwen3-8b 16
x14_model allenai/OLMo-2-1124-7B 7df9a82518afdecae4e8c026b27adccc8c1f0032 olmo-2-1124-7b 16
x14_model Qwen/Qwen3-14B-Base 0b0bd3732e2c374d483664439ea334928b65f304 qwen3-14b-base 8
echo "X14 followup complete $(date -u +%FT%TZ)"
