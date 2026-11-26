"""One table across every model that ran H2-C2 (amendment A16). CPU only.

    python scripts/c2_cross_model.py [--results results] [--out results/c2_cross_model]

Rows come from:
  * results/c2_h3/raw_results.csv   - the seven Qwen2.5 rungs (L=128 is
                                      H2's setting, same items)
  * results/c2_h2x_*/raw_results.csv - the extra models (A16)

For each model: the arms' means on the three family cells, the confirmatory
gaps with bootstrap CIs and raw Wilcoxon p, the harness and bare model
against seasonal naive on forecasting, coverage (share answered), and
routing. DESCRIPTIVE: the per-model confirmatory verdicts are in each
model's own analysis.json; this table only lines them up. It applies no
correction across models and draws no pooled verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runners import _c2 as X                     # noqa: E402
from src.config import load_config               # noqa: E402
from src.pipeline import make_samples            # noqa: E402

CELLS = ("forecasting", "trend_description", "event_explanation")
QWEN = ["qwen2_5_0_5b", "qwen2_5_1_5b", "qwen2_5_3b", "qwen2_5_7b",
        "qwen2_5_14b", "qwen2_5_32b", "qwen2_5_72b"]


def boot(d, n=5000, seed=20261111):
    d = np.asarray(d, float)
    m = np.random.default_rng(seed).choice(d, (n, len(d))).mean(1)
    return np.percentile(m, [2.5, 97.5])


def pval(d):
    d = np.asarray(d, float)
    return float(wilcoxon(d).pvalue) if np.any(d != 0) else float("nan")


def load_runs(res: Path) -> dict[str, pd.DataFrame]:
    runs = {}
    h3 = res / "c2_h3" / "raw_results.csv"
    if h3.exists():
        df = pd.read_csv(h3, low_memory=False)
        for bb in QWEN:
            sub = df[df.backbone == bb]
            if len(sub):
                runs[bb] = sub
    for d in sorted(res.glob("c2_h2x_*")):
        f = d / "raw_results.csv"
        if f.exists():
            runs[d.name.replace("c2_h2x_", "")] = pd.read_csv(
                f, low_memory=False)
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--pool", default=None,
                    help="data/c2_n100 directory (default: the config's)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = Path(args.results)
    out = Path(args.out) if args.out else res / "c2_cross_model"
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(ROOT / "configs" / "c2_h2.yaml")
    if args.pool:
        cfg.data["pool_dir"] = args.pool
    samples = make_samples(cfg, ROOT)
    base = X.forecast_baseline_scores(cfg, samples, 128).pivot(
        index="sample_id", columns="baseline", values="score")
    snaive = base["seasonal_naive_estimated_period"]

    runs = load_runs(res)
    if not runs:
        print("no results found under", res)
        return 1
    rows = []
    for model, df in runs.items():
        r = dict(model=model, family="Qwen2.5" if model in QWEN else "other")
        for qt in CELLS:
            p = df[df.query_type == qt].pivot_table(
                index="sample_id", columns="system", values="score")
            for arm in ("bare_llm", "bare_features", "harness"):
                r[f"{qt}:{arm}"] = float(p[arm].mean())
            d = (p.harness - p.bare_llm).to_numpy()
            lo, hi = boot(d)
            r[f"{qt}:gap"] = float(d.mean())
            r[f"{qt}:gap_ci"] = f"[{lo:+.3f},{hi:+.3f}]"
            r[f"{qt}:gap_p"] = pval(d)
            if qt == "forecasting":
                j = p.join(snaive)
                for arm in ("bare_llm", "harness"):
                    dd = (j[arm] - j["seasonal_naive_estimated_period"]).to_numpy()
                    lo, hi = boot(dd)
                    r[f"{arm}_vs_snaive"] = float(dd.mean())
                    r[f"{arm}_vs_snaive_ci"] = f"[{lo:+.3f},{hi:+.3f}]"
                    r[f"{arm}_vs_snaive_p"] = pval(dd)
        r["coverage_bare"] = float(df[df.system == "bare_llm"].answered.mean())
        rt = df[(df.system == "harness") & (df.route_source == "llm")]
        r["routing_functional"] = float(rt.route_equivalent.fillna(0).mean()) \
            if len(rt) else float("nan")
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(out / "cross_model.csv", index=False)

    pd.set_option("display.width", 220)
    print("=== forecasting: score, and against seasonal naive (est. period)")
    print(t[["model", "forecasting:bare_llm", "forecasting:harness",
             "bare_llm_vs_snaive", "bare_llm_vs_snaive_ci",
             "harness_vs_snaive", "forecasting:gap", "forecasting:gap_ci"]]
          .round(3).to_string(index=False))
    print("\n=== descriptive cells: harness - bare (bare_features beside it)")
    print(t[["model", "trend_description:bare_llm",
             "trend_description:bare_features", "trend_description:harness",
             "trend_description:gap_ci", "event_explanation:bare_llm",
             "event_explanation:bare_features", "event_explanation:harness",
             "event_explanation:gap_ci"]].round(3).to_string(index=False))
    print("\n=== coverage (bare arm) and functional routing (harness)")
    print(t[["model", "coverage_bare", "routing_functional"]].round(3)
          .to_string(index=False))

    n = len(t)
    summary = dict(
        n_models=n,
        harness_beats_bare_on_forecasting=int(
            ((t["forecasting:gap"] > 0) & (t["forecasting:gap_p"] < 0.05)).sum()),
        bare_below_snaive=int(
            ((t.bare_llm_vs_snaive < 0) & (t.bare_llm_vs_snaive_p < 0.05)).sum()),
        bare_not_distinguishable_from_snaive=int((t.bare_llm_vs_snaive_p >= 0.05).sum()),
        bare_above_snaive=int(
            ((t.bare_llm_vs_snaive > 0) & (t.bare_llm_vs_snaive_p < 0.05)).sum()),
        note="Raw, uncorrected p-values per model; descriptive counts only.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2),
                                      encoding="utf-8")
    print("\n=== counts across", n, "models (raw p < 0.05, uncorrected):")
    for k, v in summary.items():
        if k not in ("n_models", "note"):
            print(f"  {k:<40} {v}")
    print(f"\nwrote {out}/cross_model.csv, summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
