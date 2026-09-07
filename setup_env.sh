#!/bin/bash
# Environment setup for multilingual-probing on Bender.
#
#   source setup_env.sh
#
# Safe to source interactively and from a SLURM job script under --export=NONE.
#
# DUAL STACK
# ----------
# Bender installs its EasyBuild software twice. login01 and the A40 nodes use
# the INTEL stack; the A100 nodes use AMD. Loading the wrong one gives
# "illegal instruction" at runtime.
#
# The stack is selected here from the CPU vendor rather than from a flag, so
# the same script is correct on the login node, on A40 and on A100 without
# being told which. Each stack gets its own venv (.venv-intel / .venv-amd)
# because a venv built against one stack's Python is not valid under the other.
#
# `source /etc/profile` comes first: under --export=NONE the module system is
# not initialised, and `module` is undefined without it.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: source this file, do not execute it:"
    echo "    source ${BASH_SOURCE[0]}"
    exit 1
fi

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_MODULE="Python/3.12.3"
CUDA_MODULE="CUDA/12.4.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

# --- module system ---------------------------------------------------------
export LC_ALL="${LC_ALL:-C.UTF-8}"
export LANG="${LANG:-C.UTF-8}"
source /etc/profile

# --- stack selection by CPU vendor ----------------------------------------
if grep -qi "AuthenticAMD" /proc/cpuinfo; then
    STACK="AMD"
    STACK_PATH="/software/easybuild-AMD_A100/modules/all"
    OTHER_PATH="/software/easybuild-INTEL_A40/modules/all"
    VENV="$PROJECT_ROOT/.venv-amd"
else
    STACK="INTEL"
    STACK_PATH="/software/easybuild-INTEL_A40/modules/all"
    OTHER_PATH="/software/easybuild-AMD_A100/modules/all"
    VENV="$PROJECT_ROOT/.venv-intel"
fi

module purge 2>/dev/null || true
module unuse "$OTHER_PATH" 2>/dev/null || true
module use   "$STACK_PATH"

module load "$PYTHON_MODULE" || { echo "ERROR: cannot load $PYTHON_MODULE"; return 1; }
module load "$CUDA_MODULE"   || { echo "ERROR: cannot load $CUDA_MODULE";   return 1; }

echo "stack  : $STACK ($(hostname))"

# --- virtualenv ------------------------------------------------------------
if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "creating venv: $VENV"
    python3 -m venv "$VENV" || return 1
    source "$VENV/bin/activate"
    pip install --quiet --upgrade pip setuptools wheel

    # torch explicitly, before requirements.txt. Resolved as a transitive
    # dependency it can silently become a CPU-only wheel, which runs fine and
    # takes hours instead of minutes.
    echo "installing torch (cu124)"
    pip install --quiet torch --index-url "$TORCH_INDEX" || return 1

    if [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
        echo "installing requirements.txt"
        pip install --quiet -r "$PROJECT_ROOT/requirements.txt" || return 1
    fi
else
    source "$VENV/bin/activate"
fi

# --- environment -----------------------------------------------------------
# Bender has no scratch; home is 100 GB. The HF cache is shared with other
# projects deliberately, since the Llama weights are ~16 GB.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
# Project root, NOT root/src: this project imports as `src.data.load_json`.
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

mkdir -p "$PROJECT_ROOT/logs"

echo "python : $(python --version 2>&1)"
python - <<'PY'
try:
    import torch
    print(f"torch  : {torch.__version__}  cuda_build={torch.version.cuda}  "
          f"available={torch.cuda.is_available()}")
    if torch.version.cuda is None:
        print("WARNING: CPU-only torch. Something overrode the cu124 install --")
        print("check whether requirements.txt pins its own torch version.")
except ImportError:
    print("torch  : NOT INSTALLED")
PY
echo "venv   : $VENV"
echo "(cuda available=False on the login node is expected -- no GPU there)"