#!/usr/bin/env bash
# Regenerate a deleted all-layer cache with the frozen extractor and require it to
# reproduce the stored RQ-C AUROC and cosine to within 1e-6.
set -euo pipefail
if [[ $# -ne 6 ]]; then echo "usage: $0 RUN_ROOT PYTHON MODEL_PATH MODEL_ID GPU_INDEX BATCH" >&2; exit 2; fi
run_root=$1; python_bin=$2; model_path=$3; model_id=$4; gpu_index=$5; batch=$6
study_root="$run_root/study_truth_transport_v2"
result_root="$run_root/results/$model_id"
cache_root="$run_root/caches/$model_id"
cd "$run_root"
if [[ -f "$cache_root/VERIFIED" ]]; then exit 0; fi
"$study_root/runners/run_single_model_extraction.sh" "$run_root" "$python_bin" "$model_path" "$model_id" "$gpu_index" "$batch"
check="$result_root/extensions/x7_rq_c_reproduction.json"
mkdir -p "$result_root/extensions"
rm -f "$check"
"$python_bin" -m study_truth_transport_v2.runners.analyze_rq_c \
  --model-id "$model_id" --prompt-root "$study_root/data/${PROMPT_SET:-prompts_v1}" \
  --cache-root "$cache_root" --layer-selection "$result_root/layer_selection.json" --output "$check"
"$python_bin" - "$result_root/rq_c.json" "$check" <<'PY'
import json, sys
stored, rerun = (json.load(open(path))["cells"] for path in sys.argv[1:3])
key = lambda cell: (cell["source"], cell["target"])
stored = {key(c): c for c in stored}
worst = max(abs(stored[key(c)][m] - c[m]) for c in rerun for m in ("auroc", "direction_cosine"))
print(json.dumps({"rq_c_reproduction_max_abs_diff": worst}))
if worst > 1e-6:
    sys.exit(f"Regenerated cache does not reproduce stored RQ-C (max diff {worst})")
PY
touch "$cache_root/VERIFIED"
