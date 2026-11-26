#!/usr/bin/env bash
# Build the environment as a CONDA PREFIX on /scratch. Run under sbatch by
# setup_env.sh; not called directly.
#
# WHY CONDA AND NOT python3 -m venv
# ---------------------------------
# Two independent failures on hpc01 killed the venv approach, and conda fixes
# both:
#
# 1. MISSING PYTHON HEADERS -> vLLM cannot start at all.
#    Triton (vLLM's JIT) compiles a C extension at runtime:
#      triton/backends/nvidia/driver.c:9:10: fatal error: Python.h: No such
#      file or directory
#    then "EngineCore failed to start". Measured 2026-11-06: no system
#    interpreter here has usable headers -
#      /usr/include/python3.12/Python.h   absent
#      /usr/include/python3.11/           exists but Python.h absent
#    and python3.12-devel cannot be installed without root. A conda prefix
#    ships $PREFIX/include/python3.*/Python.h, so triton compiles.
#
# 2. THE VENV WAS NOT PORTABLE ACROSS NODE TYPES.
#    `python3 -m venv` symlinks bin/python3.12 -> /usr/bin/python3.12. That
#    exists on the compute and A100 nodes but NOT on the H100/H200 nodes or
#    the login node, so those jobs died instantly with
#      slurm_script: line 63: python: command not found   (exit 127)
#    A conda prefix on shared /scratch carries its own interpreter and runs
#    identically everywhere.
#
# GitHub release downloads are blocked here (SSL_ERROR_SYSCALL to
# release-assets.githubusercontent.com), so uv / python-build-standalone are
# not options. PyPI works, and conda comes from a module.
set -euo pipefail

TSARCH="${TSARCH:?TSARCH must be set (export it before sbatch)}"
PREFIX="$TSARCH/conda"
PYVER="${TSARCH_PYVER:-3.12}"
export PIP_CACHE_DIR="$TSARCH/pip"

echo "=== node: $(hostname) ==="

CONDA_MOD="${TSARCH_CONDA_MOD:-anaconda3-2022.05-gcc-11.2.0-od5lltp}"
module purge 2>/dev/null || true
module load "$CONDA_MOD" 2>/dev/null || {
  echo "ABORT: cannot load '$CONDA_MOD'. Available:"
  module avail 2>&1 | tr -s ' ' '\n' | grep -i conda | sort -u | sed 's/^/  /'
  exit 1
}
command -v conda >/dev/null || { echo "ABORT: conda not on PATH after module load"; exit 1; }
echo "conda: $(conda --version 2>&1)"

# conda-forge only: the defaults channel carries Anaconda's commercial terms,
# and conda-forge has the newer Python builds anyway. --override-channels so
# a system .condarc cannot reintroduce defaults.
if [[ ! -x "$PREFIX/bin/python" ]]; then
  echo "=== creating conda prefix at $PREFIX (python $PYVER) ==="
  conda create -y -p "$PREFIX" -c conda-forge --override-channels \
    "python=$PYVER" pip
else
  echo "=== conda prefix already exists, reusing ==="
fi

export PATH="$PREFIX/bin:$PATH"
hash -r
echo "python : $(command -v python)  $(python --version 2>&1)"

# THE check this whole file exists for
INC="$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
echo "headers: $INC"
if [[ ! -f "$INC/Python.h" ]]; then
  echo "ABORT: $INC/Python.h is missing, so triton cannot compile its driver"
  echo "       and vLLM will fail at engine startup exactly as before."
  exit 1
fi
echo "        Python.h present - triton can compile"

python -m pip install --quiet --upgrade pip wheel

echo "=== framework requirements ==="
python -m pip install --quiet 'numpy>=1.24' 'pandas>=2.0' 'scipy>=1.10' \
                             'pyyaml>=6.0' 'pytest>=7.4'

echo "=== vLLM (brings its own torch) ==="
python -m pip install --quiet vllm

echo "=== chronos-forecasting ==="
python -m pip install --quiet chronos-forecasting

# Rewrite env.sh to activate THIS prefix. Jobs source it, so it is the single
# place that decides which interpreter the whole experiment uses.
cat > "$TSARCH/env.sh" <<ENVEOF
# source this at the top of every job. Written by slurm/_build_conda.sh.
SCRATCH="\${SCRATCH:-/scratch/\$USER}"
export TSARCH="\${TSARCH:-$TSARCH}"

# conda prefix on shared /scratch: same interpreter on every node type, and
# it carries its own Python.h so triton can compile. See _build_conda.sh.
module purge 2>/dev/null || true
module load $CONDA_MOD 2>/dev/null || true
export PATH="\$TSARCH/conda/bin:\$PATH"
hash -r 2>/dev/null || true

export HF_HOME="\$TSARCH/hf"
export HUGGINGFACE_HUB_CACHE="\$TSARCH/hf/hub"
export TRANSFORMERS_CACHE="\$TSARCH/hf/transformers"
export PIP_CACHE_DIR="\$TSARCH/pip"
export TORCH_HOME="\$TSARCH/torch"
export TRITON_CACHE_DIR="\$TSARCH/triton"
export VLLM_CACHE_ROOT="\$TSARCH/vllm"
export XDG_CACHE_HOME="\$TSARCH/.cache"
export TMPDIR="\$TSARCH/tmp"
mkdir -p "\$TMPDIR" "\$TRITON_CACHE_DIR"

if [[ -s "\$HOME/.hf_token" ]]; then
  HF_TOKEN="\$(cat "\$HOME/.hf_token")"; export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="\$HF_TOKEN"
fi
if [[ "\${TSARCH_OFFLINE:-0}" == "1" ]]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi

export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TSARCH_CACHE="\$TSARCH/cache"
ENVEOF

echo
echo "=== versions ==="
python - <<'PY'
import importlib
for m in ("numpy", "pandas", "scipy", "yaml", "torch", "vllm"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:10} {getattr(mod, '__version__', '?')}")
    except Exception as exc:                        # noqa: BLE001
        print(f"  {m:10} MISSING ({type(exc).__name__})")
try:
    import torch
    print(f"  torch cuda build: {torch.version.cuda}")
except Exception:                                   # noqa: BLE001
    pass
PY
echo "=== conda env ready at $PREFIX; env.sh rewritten ==="
