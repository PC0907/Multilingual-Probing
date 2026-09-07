#!/bin/bash
#SBATCH --job-name=probe-extract-de
#SBATCH --partition=A40short
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

cd "${SLURM_SUBMIT_DIR:-$HOME/multilingual-probing}"
source setup_env.sh

set -euo pipefail

echo "=== node ==="
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo
echo "=== gpu visible to torch ==="
python -c "
import torch, sys
if not torch.cuda.is_available():
    sys.exit('FATAL: no GPU visible to torch; the run would fall back to CPU')
print(torch.cuda.get_device_name(0))
"

echo
echo "=== extraction ==="
python scripts/02_extract.py \
    --model meta-llama/Llama-3.1-8B \
    --index data/processed/de/index.csv \
    --out   data/processed/de/llama-3.1-8b \
    --pooling last \
    --batch-size 16 \
    --max-length 128 \
    --dtype bfloat16

echo
echo "=== done ==="
ls -la data/processed/de/llama-3.1-8b | head