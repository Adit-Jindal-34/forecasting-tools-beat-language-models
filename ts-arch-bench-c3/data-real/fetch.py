"""Re-download the Monash `.tsf` files and check their hashes.

Idempotent: a file already present is left alone, so this is safe to run on
the cluster as a prepare step.

The first four back `pool_real_n300` (cycle 3's H1 L-sweep). `weather` and
`fred_md` were added for the two cycle-3 pools built by `build_pool_c3.py`:
both are in Chronos Benchmark II, and between them they carry the only
weak-seasonality series long enough for the L=512 geometry.

Run:  python data-real/fetch.py
"""
from __future__ import annotations

import hashlib
import io
import os
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")

# (zenodo record, zip name, first 16 hex of the zip's sha256)
SOURCES = {
    "m4_hourly": (4656589, "m4_hourly_dataset.zip", "18085bd3c34e41cd"),
    "nn5_daily": (4656117, "nn5_daily_dataset_without_missing_values.zip",
                  "14d504384a97f773"),
    "electricity_hourly": (4656140, "electricity_hourly_dataset.zip",
                           "eff447075dde68dc"),
    "traffic_hourly": (4656132, "traffic_hourly_dataset.zip",
                       "3db12ba866a9c9d3"),
    # cycle 3 (build_pool_c3.py). Hashes recorded from the downloads of
    # 2026-11-14; a mismatch means the Zenodo record changed and must not
    # be accepted silently.
    "weather": (4654822, "weather_dataset.zip", "adc9195b4b842673"),
    "fred_md": (4654833, "fred_md_dataset.zip", "305c0edd2b5e9715"),
}


def main() -> None:
    os.makedirs(RAW, exist_ok=True)
    for key, (rec, zip_name, want) in SOURCES.items():
        tsf = os.path.join(RAW, zip_name.replace(".zip", ".tsf"))
        if os.path.exists(tsf):
            print(f"{key:20} present, skipped")
            continue
        url = f"https://zenodo.org/records/{rec}/files/{zip_name}?download=1"
        print(f"{key:20} downloading...", end=" ", flush=True)
        blob = urllib.request.urlopen(url, timeout=300).read()
        got = hashlib.sha256(blob).hexdigest()[:16]
        if got != want:
            raise SystemExit(
                f"\n{key}: sha256 mismatch - expected {want}, got {got}. "
                "The record changed upstream; do not use it silently.")
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            z.extract(z.namelist()[0], RAW)
        print(f"{len(blob) / 1e6:6.2f} MB  sha256:{got}  ok")


if __name__ == "__main__":
    main()
