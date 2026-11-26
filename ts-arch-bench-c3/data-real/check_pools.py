"""Check every real-data config against the pool it declares.

    python data-real/check_pools.py

Reads each `configs/*.yaml` with `data.name: real`, hashes the
`windows.npz` it points at and compares with `data.pool_sha256`. Touches
no model and writes nothing. Exit status 1 if anything does not match, so
it can sit in front of a submission.
"""
from __future__ import annotations

import glob
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.config import load_config  # noqa: E402


def main() -> int:
    bad = 0
    seen = 0
    for path in sorted(glob.glob(os.path.join(ROOT, "configs", "*.yaml"))):
        try:
            cfg = load_config(path)
        except Exception:               # not a run config (e.g. a fragment)
            continue
        if not getattr(cfg, "is_real", False):
            continue
        seen += 1
        npz = os.path.join(str(cfg.pool_dir(ROOT)), "windows.npz")
        want = str(cfg.data.get("pool_sha256") or "")
        if not os.path.exists(npz):
            got, ok = "MISSING", False
        else:
            got = hashlib.sha256(open(npz, "rb").read()).hexdigest()
            ok = got == want
        bad += not ok
        print(f"{'ok      ' if ok else 'MISMATCH'}  "
              f"{os.path.basename(path):28} {cfg.data.get('pool_dir'):32} "
              f"declared {want[:16]}  on disk {got[:16]}")
    if not seen:
        print("no real-data configs found")
        return 1
    if bad:
        print(f"\n{bad} config(s) do not match their pool. Do NOT edit the "
              "hash in the config: restore the pool that was shipped with "
              "the code (data-real/pool_*/).")
        return 1
    print(f"\nall {seen} real-data configs match their pools")
    return 0


if __name__ == "__main__":
    sys.exit(main())
