#!/usr/bin/env bash
# Source this BEFORE slurm/serve_vllm.sh:
#
#     source slurm/pick_port.sh
#     source slurm/serve_vllm.sh "$MODEL"
#
# WHY. Loopback (127.0.0.1) is shared by every job on a node. The H200
# partition is ONE node with eight cards, so two of our jobs - two rungs of
# the H3 array, or H3 beside H1 - routinely land together, and with the old
# fixed port 8000 the second server cannot bind while its client talks to
# the first job's server. slurm/c2_run.sbatch has done this since cycle 2;
# the single-purpose scripts (h1, h1_all, h2, h3, c3_h2_strata) had not.
#
# The client learns the port from TSARCH_VLLM_PORT, which changes the
# connection and NOT the cache key (models/providers.py), so cached answers
# replay whatever port a job happened to get.
#
# An explicit VLLM_PORT in the environment is respected.
if [[ -z "${VLLM_PORT:-}" ]]; then
  VLLM_PORT="$(python - <<'PY'
import os, socket
start = 8000 + int(os.environ.get("SLURM_JOB_ID", "0") or 0) % 1000
for port in list(range(start, start + 200)) + list(range(9000, 9200)):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
    except OSError:
        continue
    finally:
        s.close()
    print(port)
    break
PY
)"
fi
if [[ -z "${VLLM_PORT:-}" ]]; then
  echo "[port] no free port found on $(hostname)" >&2
  return 1 2>/dev/null || exit 1
fi
export VLLM_PORT TSARCH_VLLM_PORT="$VLLM_PORT"
echo "[port] this job serves and connects on 127.0.0.1:${VLLM_PORT}"
