#!/bin/bash
#SBATCH --job-name=probe-extract-de
#SBATCH --partition=A100devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "$HOME/multilingual-probing"
source .venv/bin/activate

export HF_HOME="$HOME/.cache/huggingface"
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN before submitting}"
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/02_extract.py \
    --model meta-llama/Llama-3.1-8B \
    --index data/processed/de/index.csv \
    --out   data/processed/de/llama-3.1-8b \
    --pooling last \
    --batch-size 16 \
    --max-length 128 \
    --dtype bfloat16