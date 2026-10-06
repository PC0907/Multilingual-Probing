#!/usr/bin/env bash
# usage: gpu_ok.sh GPU_INDEX [STEP_LABEL]; exit 0 if a step of ours may start on this GPU.
# Never usable if any other user has a process on it (whatever its memory reading: a job that has just started
# shows little memory). If only our own processes are there (the other v5 lane), it must have at least
# GPU_MIN_FREE_MIB free (default 1024 used at most when GPU_MIN_FREE_MIB is unset, i.e. an idle GPU).
gpu=$1; label=${2:-step}; me=$(id -u)
while IFS=, read -r pid; do
  pid=${pid// /}; [[ -z "$pid" ]] && continue
  uid=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')
  if [[ -n "$uid" && "$uid" != "$me" ]]; then
    echo "GPU $gpu is no longer idle before $label: process $pid of another user" >&2; exit 1
  fi
done < <(nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
read -r used total < <(nvidia-smi --id="$gpu" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
if [[ -n "${GPU_MIN_FREE_MIB:-}" ]]; then
  (( total - used >= GPU_MIN_FREE_MIB )) && exit 0
  echo "GPU $gpu is no longer idle before $label: $((total - used)) MiB free < ${GPU_MIN_FREE_MIB} MiB" >&2; exit 1
fi
(( used <= 1024 )) && exit 0
echo "GPU $gpu is no longer idle before $label: ${used} MiB used" >&2; exit 1
