#!/usr/bin/env bash
# The venv build, run under srun by setup_env.sh. Not called directly.
#
# WHY IT PROBES FOR AN INTERPRETER INSTEAD OF LOADING A MODULE
# ------------------------------------------------------------
# Measured on hpc01, node19, 2026-11-06:
#
#   module load python-3.10.8-gcc-11.2.0-dlcmq7k   -> exits 0, module listed
#   its bin on PATH: /apps/spack/.../python-3.10.8-dlcmq7k.../bin
#   python    -> command not found
#   python3   -> /usr/bin/python3   = 3.6.8
#   python3.10-> command not found
#
# The module is loaded and its bin directory is on PATH, but that directory
# contains no python binary: the spack install is broken on the compute
# nodes. `module load` reporting success therefore proves nothing, which is
# why the version is now checked rather than assumed.
#
# The compute nodes do ship /usr/bin/python3.12, so no module is needed at
# all. This probes candidates newest-first and takes the first that is in
# [3.9, 3.13) - the range where numpy>=1.24, pandas>=2.0, scipy>=1.10 and
# vLLM all have wheels.
#
# Override with TSARCH_PYTHON=/path/to/python if a cluster change makes the
# probe wrong.
#
# WHY IT ABORTS RATHER THAN FALLING BACK
# --------------------------------------
# The first version did `module load A || module load B || true`. Both
# modules were absent, the `|| true` swallowed it, and the venv was built
# against Python 3.6. The failure then surfaced forty lines later as
# "No matching distribution found for numpy>=1.24", naming neither the
# missing module nor the wrong interpreter.
set -euo pipefail

TSARCH="${TSARCH:?TSARCH must be set (export it before srun)}"
export PIP_CACHE_DIR="$TSARCH/pip"

echo "=== node: $(hostname) ==="

pick_python() {
  if [[ -n "${TSARCH_PYTHON:-}" ]]; then
    echo "$TSARCH_PYTHON"; return
  fi
  local c
  for c in /usr/bin/python3.12 python3.12 /usr/bin/python3.11 python3.11 \
           /usr/bin/python3.10 python3.10 /usr/bin/python3.9 python3.9 \
           python3; do
    command -v "$c" >/dev/null 2>&1 || continue
    # in [3.9, 3.13): below 3.9 the wheels do not exist, and 3.13+ is ahead
    # of what vLLM ships for
    "$c" -c 'import sys; sys.exit(0 if (3,9) <= sys.version_info[:2] < (3,13) else 1)' \
      2>/dev/null || continue
    command -v "$c"; return
  done
  return 1
}

PY="$(pick_python || true)"
if [[ -z "$PY" ]]; then
  echo "ABORT: no interpreter in [3.9, 3.13) on $(hostname)."
  echo "Candidates examined:"
  for c in /usr/bin/python3.12 python3.12 /usr/bin/python3.11 python3.11 \
           /usr/bin/python3.10 python3.10 python3; do
    printf '  %-26s %s\n' "$c" \
      "$(command -v "$c" >/dev/null 2>&1 && "$c" --version 2>&1 || echo absent)"
  done
  echo "Set TSARCH_PYTHON=/path/to/python and re-run slurm/setup_env.sh."
  exit 1
fi

echo "interpreter : $PY  ($("$PY" --version 2>&1))"

# a half-built venv from a previous failed attempt would shadow the new one
rm -rf "$TSARCH/env"
"$PY" -m venv "$TSARCH/env"
# shellcheck disable=SC1091
source "$TSARCH/env/bin/activate"
echo "venv python : $(python --version 2>&1)"
python -m pip install --quiet --upgrade pip wheel

echo "=== framework requirements ==="
python -m pip install --quiet 'numpy>=1.24' 'pandas>=2.0' 'scipy>=1.10' \
                             'pyyaml>=6.0' 'pytest>=7.4'

echo "=== vLLM (pulls its own torch build) ==="
# vLLM pins a specific torch; install it FIRST and let it choose, or pip
# resolves torch twice and the second pass can silently replace the CUDA
# build with a CPU-only one.
python -m pip install --quiet vllm

echo "=== chronos-forecasting ==="
python -m pip install --quiet chronos-forecasting

echo
echo "=== versions ==="
python - <<'PY'
import importlib
for m in ("numpy", "pandas", "scipy", "yaml", "torch", "vllm"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:10} {getattr(mod, '__version__', '?')}")
    except Exception as exc:                        # noqa: BLE001
        print(f"  {m:10} MISSING ({type(exc).__name__}: {exc})")
try:
    import torch
    print(f"  torch cuda build: {torch.version.cuda}")
except Exception:                                   # noqa: BLE001
    pass
PY
echo "=== venv built at $TSARCH/env ==="
