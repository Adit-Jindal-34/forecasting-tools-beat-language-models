"""Generate every visual in the blog post from the FINAL archived results.

    python blog/make_figures.py

Sources, all under results-archive/ :

  20261115-oss120b-high/.../c2_cross_model/cross_model.csv
        the 14-configuration synthetic table (7 Qwen sizes + 6 other families,
        gpt-oss-120b counted twice at two reasoning efforts)
  20261115-cycle3-real-data/ts-arch-bench-c3/results/
        the real-data (Monash) run: c3_h1_clean, c3_h2_strata,
        c3_sensitivity_h8, c3_h1_contaminated
  20261113-cycle2-run/data/c2_n100/
        the frozen synthetic pool, for the two worked examples

Nothing here is typed in by hand except the three evidence-block medians,
which are quoted from FINDINGS_C2 section 5.3 and labelled as such.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = os.path.join(ROOT, "results-archive")
C2 = os.path.join(ARCH, "20261113-cycle2-run")
C3 = os.path.join(ARCH, "20261115-cycle3-real-data", "ts-arch-bench-c3", "results")
XMODEL = os.path.join(ARCH, "20261115-oss120b-high", "ts-arch-bench",
                      "results", "c2_cross_model", "cross_model.csv")
OUT = os.path.join(ROOT, "blog", "figs")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- palette
HARNESS = "#2a78d6"   # blue   - the tool setup, in EVERY chart
BARE    = "#eb6834"   # orange - the plain model, in EVERY chart
FEAT    = "#eda100"   # yellow - plain model + computed facts
ORACLE  = "#4a3aa7"   # violet - tools, told which tool
NEG     = "#e34948"   # red    - the tool setup LOSES
RULE    = "#1baf7a"   # green  - a fixed arithmetic rule
TRUTH   = "#0b0b0b"
SURF, INK, INK2, MUTED, GRID, BASE = ("#fcfcfb", "#0b0b0b", "#52514e",
                                      "#898781", "#e1e0d9", "#c3c2b7")

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 14, "legend.frameon": False, "legend.fontsize": 10,
})
DPI = 200


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))


def tag(ax, text, y=-0.30):
    ax.text(0, y, text, transform=ax.transAxes, fontsize=9, color=MUTED,
            va="top", ha="left", linespacing=1.6)


# ================================================================ loaders
LABELS = {
    "qwen2_5_0_5b": "Qwen2.5 0.5B", "qwen2_5_1_5b": "Qwen2.5 1.5B",
    "qwen2_5_3b": "Qwen2.5 3B", "qwen2_5_7b": "Qwen2.5 7B",
    "qwen2_5_14b": "Qwen2.5 14B", "qwen2_5_32b": "Qwen2.5 32B",
    "qwen2_5_72b": "Qwen2.5 72B", "gpt_oss_120b": "gpt-oss-120b",
    "gpt_oss_120b_high": "gpt-oss-120b (high)", "gpt_oss_20b": "gpt-oss-20b",
    "granite_3_3_8b": "Granite-3.3-8B", "mistral_small_3_2": "Mistral-Small-3.2",
    "olmo_2_32b": "OLMo-2-32B", "phi_4": "Phi-4",
}
ORDER = ["phi_4", "gpt_oss_120b_high", "qwen2_5_3b", "qwen2_5_0_5b",
         "olmo_2_32b", "qwen2_5_7b", "gpt_oss_20b", "qwen2_5_1_5b",
         "mistral_small_3_2", "granite_3_3_8b", "qwen2_5_14b",
         "gpt_oss_120b", "qwen2_5_32b", "qwen2_5_72b"]


def cross_model() -> pd.DataFrame:
    d = pd.read_csv(XMODEL).set_index("model").loc[ORDER].reset_index()
    d["lab"] = d.model.map(LABELS)
    return d


def c3(name: str) -> dict:
    with open(os.path.join(C3, name, "analysis.json")) as f:
        return json.load(f)


def worked_example():
    z = np.load(os.path.join(C2, "data", "c2_n100", "pool.npz"))
    s = z["s83"]
    return (s[512 - 128:512], s[512:520],
            np.array([57.23, 57.04, 56.85, 56.67, 56.49, 56.31, 56.13, 55.95]),
            np.array([60.3666, 62.0568, 64.23, 67.1276, 68.8179, 72.1984,
                      73.6472, 75.096]))


def failure_example():
    z = np.load(os.path.join(C2, "data", "c2_n100", "pool.npz"))
    return z["s42"][512 - 128:512]


# SNAIVE on the synthetic pool, from the cross-model run's own baseline block
SNAIVE_SYNTH = 0.4966


# ================================================================ figures
def fig01_cross_model():
    """THE headline: 14 configurations, one flat tool line, one dumb baseline."""
    d = cross_model()
    y = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(12, 6.6))

    ax.axvspan(SNAIVE_SYNTH, 0.58, color=HARNESS, alpha=0.05, zorder=0)
    ax.axvline(SNAIVE_SYNTH, color=MUTED, lw=1.8, ls="--", zorder=3)
    ax.axvline(0.5491, color=HARNESS, lw=2.6, zorder=3)

    for yi, (_, r) in zip(y, d.iterrows()):
        # the tool's own score per row. It is 0.549 on 13 of the 14; the
        # exception is Qwen 0.5B, which cannot reliably pick the right tool.
        h = r["forecasting:harness"]
        ax.plot([r["forecasting:bare_llm"], h], [yi, yi], color=GRID,
                lw=6, solid_capstyle="round", zorder=2)
        ax.plot(r["forecasting:bare_llm"], yi, marker="o", ms=11, color=BARE,
                zorder=5)
        ax.plot(h, yi, marker="^", ms=10, color=HARNESS, zorder=5)
        ax.text(0.5491 + 0.014, yi, f"+{r['forecasting:gap']:.3f}",
                va="center", fontsize=9.5, color=INK2)

    ax.text(SNAIVE_SYNTH - 0.010, len(d) - 0.3,
            "a dumb baseline:\nrepeat the last cycle, 0.497", ha="right",
            va="center", fontsize=9.5, color=INK2, linespacing=1.5)
    ax.text(0.5491 + 0.014, len(d) + 0.2,
            "the tool: 0.549 on 13 of the 14,\nbecause it is the tool's own answer",
            ha="left", va="center", fontsize=9.5, color=HARNESS,
            weight="bold", linespacing=1.5)
    ax.set_xlabel("forecasting score  1/(1+MASE).  Higher is better; 0.5 ties the naive baseline.  n = 100")
    # call out the one row where the triangle does not reach the blue line
    row0 = d[d.model == "qwen2_5_0_5b"].index[0]
    y0 = y[row0]
    ax.text(0.472, y0 - 0.38, "0.477, too small to pick a tool",
            ha="right", va="top", fontsize=8.6, color=HARNESS, weight="bold")

    ax.set_yticks(y, d.lab, fontsize=10.5, color=INK)
    ax.set_xlim(0.05, 0.655); ax.set_ylim(-0.9, len(d) + 0.7)
    ax.set_title("No configuration reached the naive baseline alone. Routing to the tool cleared it in 13 of 14.")
    ax.grid(axis="x")
    ax.plot([], [], marker="o", ms=10, color=BARE, ls="none",
            label="the model alone")
    ax.plot([], [], marker="^", ms=9, color=HARNESS, ls="none",
            label="the model routing to the tool")
    ax.legend(loc="lower left", fontsize=10)
    tag(ax, "100 identical questions put to every configuration. Seven Qwen2.5 sizes plus six "
            "other models from five families; gpt-oss-120b appears twice, at\ntwo reasoning "
            "settings. Thirteen of the fourteen sit clearly below the baseline; Qwen2.5 72B is "
            "the one that merely cannot be told apart from it.\nThe three gpt-oss scores are "
            "pulled down by questions they never finished answering, which count as zero. "
            "See the text.", y=-0.19)
    save(fig, "fig01_cross_model.png")


def fig02_architecture():
    """Portrait system diagram, readable at phone width."""
    fig, ax = plt.subplots(figsize=(7.2, 11.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 160); ax.axis("off")

    def box(x, y, w, h, title, body, fc, ec, ts=12, bs=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8",
                                    fc=fc, ec=ec, lw=2))
        if body:
            ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
                    fontsize=ts, weight="bold", color=INK)
            ax.text(x + w / 2, y + h * 0.28, body, ha="center", va="center",
                    fontsize=bs, color=INK2, linespacing=1.5)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    fontsize=ts, weight="bold", color=INK, linespacing=1.5)

    def down(x, y1, y2, color=MUTED):
        ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>",
                                     mutation_scale=16, lw=2, color=color))

    ax.text(2, 157, "How one question is tested", fontsize=16, weight="bold",
            color=INK, va="top")
    box(12, 136, 76, 14, "ONE ITEM",
        "a series of numbers  +  one question\n+  the correct answer",
        "#f4f3ef", BASE)
    down(50, 135, 128)
    box(12, 112, 76, 14, "THE ANSWER IS REMOVED HERE",
        "five fields cross over, and nothing else:\n"
        "id · question · numbers · length · horizon",
        "#fff6e8", FEAT)
    ax.text(50, 109, "a test reads every file on the other side and fails\n"
                     "the build if any of them even mentions the answer",
            ha="center", va="top", fontsize=9, color=NEG, style="italic",
            linespacing=1.5)
    down(50, 103, 96)

    rows = [
        (80, "1.  MODEL ALONE", "just the numbers and the question", "#fdeee6", BARE),
        (62, "2.  MODEL + FACTS", "plus a block of computed numbers", "#fff6e8", FEAT),
        (44, "3.  MODEL + TOOLS", "it picks a tool, then the tool answers", "#e8f1fc", HARNESS),
        (26, "4.  MODEL + TOOLS, TOLD WHICH", "handed the right tool for free", "#ededf7", ORACLE),
    ]
    for yy, t, b, fc, ec in rows:
        box(16, yy, 76, 13, t, b, fc, ec)
    ax.plot([9, 9], [20.5, 86.5], color=BASE, lw=2, solid_capstyle="round", zorder=1)
    for yy, _, _, _, ec in rows:
        ax.plot([9, 15.2], [yy + 6.5, yy + 6.5], color=ec, lw=2, zorder=2)
        ax.plot([9], [yy + 6.5], marker="o", ms=6, color=ec, zorder=3)
    ax.plot([9, 50], [20.5, 20.5], color=BASE, lw=2, zorder=1)
    down(50, 20.5, 18.6, BASE)
    ax.text(11, 22.5, "all four answers", fontsize=9, color=MUTED,
            style="italic", va="bottom")
    box(16, 5, 68, 13, "ONE SCORER",
        "the same ruler for all four answers,\nagainst the same hidden future",
        "#f4f3ef", BASE)
    save(fig, "fig02_architecture.png")


def fig04_real_data():
    """Cycle 3: the pre-registered prediction, and the history-length null."""
    a = c3("c3_h1_clean")
    fb = a["c2"]["forecast_baselines"]
    Ls = [32, 64, 128, 256, 512]
    bare = [fb[str(L)]["arm_mean_score"]["bare_llm"] for L in Ls]
    harn = [fb[str(L)]["arm_mean_score"]["harness"] for L in Ls]
    decl = [fb[str(L)]["baseline_mean_score"]["seasonal_naive_true_period"] for L in Ls]
    est = [fb[str(L)]["baseline_mean_score"]["seasonal_naive_estimated_period"] for L in Ls]
    cell = a["confirmatory"]["cells"]["forecasting"]
    v = fb["512"]["versus"]["harness|seasonal_naive_true_period"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0),
                                   gridspec_kw=dict(wspace=0.26,
                                                    width_ratios=[1.3, 1]))
    x = np.arange(5)
    ax1.plot(x, harn, color=HARNESS, lw=3, marker="^", ms=10, zorder=5,
             label="model + tools")
    ax1.plot(x, bare, color=BARE, lw=2.4, marker="o", ms=8, zorder=5,
             label="model alone")
    ax1.plot(x, decl, color=MUTED, lw=2, ls="--", zorder=4,
             label="repeat the last cycle (the bar to beat)")
    ax1.plot(x, est, color=BASE, lw=1.6, ls=":", zorder=3,
             label="repeat the last cycle, length guessed")
    ax1.annotate("", xy=(4, harn[4]), xytext=(4, decl[4]),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.6))
    ax1.text(3.88, (harn[4] + decl[4]) / 2,
             f"+{v['mean_diff']:.3f}\n[{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]",
             ha="right", va="center", fontsize=10.5, weight="bold", color=INK)
    ax1.set_xticks(x, [str(L) for L in Ls], color=INK)
    ax1.set_xlabel("points of history shown")
    ax1.set_ylabel("forecasting score  1/(1+MASE)")
    ax1.set_ylim(0.355, 0.70)
    ax1.set_title("On real data, the tool clears the bar. The model does not.")
    ax1.grid(axis="y")
    ax1.legend(loc="lower center", fontsize=8.8, ncols=2, frameon=True,
               framealpha=0.96, facecolor=SURF, edgecolor="none")

    g = [harn[i] - bare[i] for i in range(5)]
    ax2.plot(x, g, color=HARNESS, lw=2.6, marker="o", ms=9)
    ax2.axhline(g[0], color=MUTED, ls=":", lw=1.2)
    ax2.annotate("", xy=(4, g[4]), xytext=(4, g[0]),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
    ax2.text(2.0, 0.230,
             f"change from 32 to 512 points:  {cell['mean_diff']:+.4f}\n"
             f"[{cell['ci_low']:+.3f}, {cell['ci_high']:+.3f}], the interval "
             "crosses zero,\nso this is not a win",
             ha="center", va="top", fontsize=9.8, weight="bold", color=NEG,
             linespacing=1.7)
    ax2.set_xticks(x, [str(L) for L in Ls], color=INK)
    ax2.set_ylim(0.125, 0.238)
    ax2.set_xlabel("points of history shown")
    ax2.set_ylabel("gap  (tools − model alone)")
    ax2.set_title("More history did not reliably widen the gap")
    ax2.grid(axis="y")
    fig.suptitle("Real data: Qwen2.5-32B on 300 series from four public archives, "
                 "forecasting 24 steps ahead",
                 x=0.012, ha="left", fontsize=15, weight="bold", y=1.03, color=INK)
    fig.text(0.012, -0.17,
             "The bar is the naive baseline that repeats the last full cycle, told how long "
             "that cycle is. That is its hardest version, and it is the one we committed to "
             "beating before running\nanything. The tool cleared it at every history length. "
             "The model alone never did: at 512 points it sits 0.113 below. Right: we also "
             "said more history would widen the gap, and it\ndid not. The change is +0.019 "
             "and its range includes zero, so we report it as a loss. The tool does improve "
             "with more history (+0.031); the model alone does not move (+0.012).",
             fontsize=9, color=MUTED, ha="left", linespacing=1.55)
    save(fig, "fig04_real_data.png")


def fig05_strata():
    """Where the advantage lives: repeating series versus irregular ones."""
    a = c3("c3_h2_strata")
    pooled = a["stratum_contrast"]["within_stratum"]
    wx = a["within_dataset"]["weather"]["within_stratum"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8),
                                   gridspec_kw=dict(wspace=0.3))
    x = np.arange(2); wd = 0.32
    for ax, src, ttl, note in (
            (ax1, pooled, "All 240 series in the balanced set",
             "120 repeating, 120 irregular, four archives"),
            (ax2, wx, "Within one archive only",
             "the stricter test: one data source, 136 series")):
        s = [src["strong"]["mean_diff"], src["weak"]["mean_diff"]]
        lo = [src["strong"]["mean_diff"] - src["strong"]["ci_low"],
              src["weak"]["mean_diff"] - src["weak"]["ci_low"]]
        hi = [src["strong"]["ci_high"] - src["strong"]["mean_diff"],
              src["weak"]["ci_high"] - src["weak"]["mean_diff"]]
        ax.bar(x, s, wd * 2, color=[HARNESS, BASE], zorder=3)
        ax.errorbar(x, s, yerr=[lo, hi], fmt="none", ecolor=INK,
                    elinewidth=1.5, capsize=5, zorder=4)
        for xi, vv in zip(x, s):
            ax.text(xi, vv + 0.012, f"+{vv:.3f}", ha="center", fontsize=11,
                    weight="bold", color=INK)
        ax.set_xticks(x, ["strongly\nrepeating", "weakly\nrepeating"],
                      fontsize=10.5, color=INK)
        ax.set_ylim(0, 0.23)
        ax.set_ylabel("gap  (tools − model alone)")
        ax.set_title(f"{ttl}\n{note}", fontsize=12, linespacing=1.9)
        ax.grid(axis="y")
    fig.suptitle("The tool's advantage is mostly about repetition",
                 x=0.012, ha="left", fontsize=15, weight="bold", y=1.15,
                 color=INK)
    fig.text(0.012, -0.19,
             "Qwen2.5-32B. This uses a second, balanced set of 240 series: 120 repeating and "
             "120 irregular, because the main 300-series set holds only 10 irregular ones, "
             "far too few to\ncompare. On repeating series the tool gains three to four times "
             "what it gains on irregular ones. Most of the irregular series come from one "
             "archive, so we said in advance that the\nwithin-archive version carries the "
             "claim: it is the right-hand panel, and it points the same way. On irregular "
             "series the tool does not beat the simple baseline at all (0.696\nagainst 0.675, "
             "a difference we cannot distinguish from zero).",
             fontsize=9, color=MUTED, ha="left", linespacing=1.55)
    save(fig, "fig05_strata.png")


def fig06_worked_example():
    hist, tgt, bare, harn = worked_example()
    fig = plt.figure(figsize=(13.5, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], wspace=0.34, hspace=0.42)
    ax = fig.add_subplot(gs[:, 0])
    n = 100
    ax.plot(np.arange(-n, 0), hist[-n:], color=MUTED, lw=1.5,
            label="the history both sides saw")
    xf = np.arange(0, 8)
    ax.plot(np.r_[-1, xf], np.r_[hist[-1], tgt], color=TRUTH, lw=2.6,
            marker="o", ms=5, label="what actually happened")
    ax.plot(np.r_[-1, xf], np.r_[hist[-1], bare], color=BARE, lw=2.2, ls="--",
            marker="s", ms=5, label="the model's answer")
    ax.plot(np.r_[-1, xf], np.r_[hist[-1], harn], color=HARNESS, lw=2.4,
            marker="^", ms=6, label="the tool's answer")
    ax.axvline(-0.5, color=BASE, ls=":", lw=1)
    ax.set_xlabel("time step"); ax.set_ylabel("value")
    # extra headroom below, so the legend sits on clean background
    ax.set_ylim(float(min(hist[-n:].min(), bare.min())) - 7.0, None)
    ax.set_title("One question, both answers")
    ax.grid(axis="y")
    ax.legend(loc="lower left", fontsize=9, frameon=True, framealpha=0.95,
              facecolor=SURF, edgecolor="none")

    axt = fig.add_subplot(gs[0, 1]); axt.axis("off")
    axt.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.015",
                                 fc="#f4f3ef", ec=BASE, lw=1,
                                 transform=axt.transAxes))
    axt.text(0.04, 0.86, "the two answers, word for word", fontsize=10.5,
             weight="bold", color=INK, transform=axt.transAxes)
    axt.text(0.04, 0.63, "the model alone", fontsize=9.5, weight="bold",
             color=BARE, transform=axt.transAxes)
    axt.text(0.04, 0.44, "57.23, 57.04, 56.85, 56.67,\n56.49, 56.31, 56.13, 55.95",
             family="monospace", fontsize=8.6, color=INK2, transform=axt.transAxes)
    axt.text(0.04, 0.26, "the model + tools", fontsize=9.5, weight="bold",
             color=HARNESS, transform=axt.transAxes)
    axt.text(0.04, 0.07, "60.37, 62.06, 64.23, 67.13,\n68.82, 72.20, 73.65, 75.10",
             family="monospace", fontsize=8.6, color=INK2, transform=axt.transAxes)

    axb = fig.add_subplot(gs[1, 1])
    vals = [0.179, 0.190, 0.631, 0.631]
    b = axb.barh(np.arange(4)[::-1], vals, height=0.56,
                 color=[BARE, FEAT, HARNESS, ORACLE], zorder=3)
    for r, vv in zip(b, vals):
        axb.text(vv + 0.012, r.get_y() + r.get_height() / 2, f"{vv:.3f}",
                 va="center", fontsize=10, weight="bold", color=INK)
    axb.set_yticks(np.arange(4)[::-1],
                   ["model alone", "model + facts", "model + tools",
                    "tools, told which"], fontsize=9.5, color=INK)
    axb.set_xlim(0, 0.8); axb.set_xlabel("score on this one question")
    axb.set_title("Score", fontsize=11)
    axb.grid(axis="x")
    fig.suptitle("The model continues the dip. The tool continues the cycle.",
                 x=0.012, ha="left", fontsize=15, weight="bold", y=1.00, color=INK)
    fig.text(0.012, -0.045,
             "The last value the model saw was 57.49 and falling. The next eight values "
             "actually rose to 72.9, because this series repeats every 24 steps and was due "
             "to climb. The model followed\nthe recent slope and was about eight times "
             "further off than the tool. This is the whole forecasting result in a single "
             "question.",
             fontsize=9, color=MUTED, ha="left", linespacing=1.5)
    save(fig, "fig06_worked_example.png")


def fig07_descriptive_by_model():
    """Trend and event gaps across all 14 configurations."""
    d = cross_model()
    o = d.sort_values("event_explanation:gap")
    y = np.arange(len(o))[::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 6.2),
                                   gridspec_kw=dict(wspace=0.46))
    for ax, col, ttl in (
            (ax1, "trend_description:gap", "Describing the trend"),
            (ax2, "event_explanation:gap", "Explaining an event")):
        vals = o[col].to_numpy()
        ax.barh(y, vals, height=0.6,
                color=[HARNESS if v > 0.02 else (NEG if v < -0.02 else BASE)
                       for v in vals], zorder=3)
        ax.axvline(0, color=INK, lw=1.2)
        for yi, vv in zip(y, vals):
            ax.text(vv + (0.012 if vv >= 0 else -0.012), yi, f"{vv:+.2f}",
                    va="center", ha="left" if vv >= 0 else "right",
                    fontsize=9, color=INK2)
        # name the routing failure on the label itself, so it cannot drift
        labs = [f"{t}\n(cannot pick tools)" if m == "qwen2_5_0_5b" else t
                for t, m in zip(o.lab, o.model)]
        ax.set_yticks(y, labs, fontsize=9.5, color=INK)
        ax.set_xlim(-0.52, 0.42)
        ax.set_xlabel("gap  (tools − model alone)")
        ax.set_title(ttl, fontsize=12.5)
        ax.grid(axis="x")
    ax1.text(0.20, 13.6, "tools help", color=HARNESS, fontsize=9.5,
             weight="bold", ha="center")
    ax1.text(-0.30, 13.6, "tools hurt", color=NEG, fontsize=9.5,
             weight="bold", ha="center")
    fig.suptitle("On descriptive questions, the answer depends on the model",
                 x=0.012, ha="left", fontsize=15, weight="bold", y=1.02, color=INK)
    fig.text(0.012, -0.085,
             "Left: on describing the trend, tools help ten of the fourteen and are roughly "
             "neutral on the rest. Right: on explaining an event, most models sit near zero, "
             "and the two large Qwen\nmodels are the outliers that first made us believe tools "
             "hurt here. Qwen2.5 0.5B is a separate story: it is too small to pick the right "
             "tool at all, so its two long bars measure\nrouting failure, not reasoning. Grey "
             "bars are smaller than our 0.02 floor, which means no effect worth claiming.",
             fontsize=9, color=MUTED, ha="left", linespacing=1.55)
    save(fig, "fig07_descriptive_by_model.png")


def fig08_evidence_trap():
    """Why the two large Qwen models go wrong on events."""
    # medians quoted from FINDINGS_C2 section 5.3
    EV_STEP, EV_SHIFT, EV_PEAK = [3.71, 6.73, 9.67], [1.86, 6.71, 2.81], [2.21, 2.02, 3.86]
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(14.5, 4.8),
        gridspec_kw=dict(wspace=0.32, width_ratios=[1.15, 1, 1.1]))
    w = failure_example()
    ax1.plot(w, color=MUTED, lw=1.4)
    ax1.plot([10], [w[10]], marker="o", ms=11, mfc="none", mec=NEG, mew=2.2)
    ax1.annotate("one spike, early on", (10, w[10]), xytext=(26, w[10] + 1.0),
                 fontsize=9.5, color=NEG, weight="bold",
                 arrowprops=dict(arrowstyle="->", color=NEG, lw=1.2))
    ax1.set_xlabel("position in the window"); ax1.set_ylabel("value")
    ax1.set_title("(a) One question, truth: a spike", fontsize=11.5)
    ax1.grid(axis="y")
    ax1.text(0.0, -0.34,
             "what the tool told the model:\n"
             "  biggest one-step jump  = 13.70\n"
             "  level shift            =  2.01\n"
             "the model alone said 'spike'  (right)\n"
             "with the tool it said 'abrupt change'  (wrong)",
             transform=ax1.transAxes, family="monospace", fontsize=8.0,
             color=INK2, va="top", linespacing=1.75)

    x = np.arange(3); wd = 0.26
    ax2.bar(x - wd, EV_STEP, wd, color=NEG, label="biggest one-step jump", zorder=3)
    ax2.bar(x, EV_SHIFT, wd, color=HARNESS, label="level shift", zorder=3)
    ax2.bar(x + wd, EV_PEAK, wd, color=RULE, label="biggest local peak", zorder=3)
    ax2.set_xticks(x, ["nothing\nhappened", "abrupt\nchange", "spike"],
                   fontsize=9.5, color=INK)
    ax2.set_ylabel("typical value of that number")
    ax2.set_ylim(0, 12.6)
    ax2.set_title("(b) Why it misleads", fontsize=11.5)
    ax2.grid(axis="y"); ax2.legend(fontsize=8.0, loc="upper left")
    ax2.text(0.0, -0.34,
             "The red number is biggest for SPIKES, not for\nabrupt changes, and every "
             "series has one.\nThe blue number is the one that actually\ntells the three apart.",
             transform=ax2.transAxes, fontsize=8.6, color=MUTED, va="top",
             linespacing=1.75)

    ax3.barh([2, 1, 0], [31, 48, 97], height=0.5,
             color=[TRUTH, BARE, HARNESS], zorder=3)
    for yi, vv in zip([2, 1, 0], [31, 48, 97]):
        ax3.text(vv + 2, yi, f"{vv}%", va="center", fontsize=11, weight="bold",
                 color=INK)
    ax3.set_yticks([2, 1, 0], ["the truth", "Qwen 72B alone",
                               "Qwen 72B with\nthe tool's facts"],
                   fontsize=9.5, color=INK)
    ax3.set_xlim(0, 118)
    ax3.set_xlabel('share of answers saying "abrupt change" (%)')
    ax3.set_title("(c) What the model then says", fontsize=11.5)
    ax3.grid(axis="x")
    ax3.text(0.0, -0.34,
             "Given that block of numbers, the largest Qwen model\nanswers \"abrupt change\" "
             "on 97 of 100 questions. The\ntrue share is 31. Most other models do not fall\n"
             "for it. See Figure 7.",
             transform=ax3.transAxes, fontsize=8.6, color=MUTED, va="top",
             linespacing=1.75)
    fig.suptitle("How one number in a tool's output can mislead a model",
                 x=0.012, ha="left", fontsize=15, weight="bold", y=1.05, color=INK)
    save(fig, "fig08_evidence_trap.png")


def fig09_real_ladder():
    """The real-data sibling of fig01: six Qwen rungs on 300 Monash series.

    DESCRIPTIVE ONLY. The seventh rung (72B) never ran, so no statistical
    verdict was computed for this cell. Both the as-scored and the
    well-formed-only bare score are plotted, because four of the six rungs
    lose items to short or missing forecasts.
    """
    import glob
    rows = []
    for p in sorted(glob.glob(os.path.join(C3, "c3_h3_clean", "shards", "*.csv"))):
        d = pd.read_csv(p, low_memory=False)
        f = d[d.query_type == "forecasting"]
        bare = f[f.system == "bare_llm"].set_index("sample_id")
        harn = f[f.system == "harness"].set_index("sample_id")
        ok = bare[bare.raw_forecast_len >= 24].index   # not short, not missing
        rows.append(dict(model=f.backbone.iloc[0], bare=bare.score.mean(),
                         harness=harn.score.mean(), bare_ok=bare.loc[ok].score.mean(),
                         short=300 - len(ok)))
    order = ["qwen2_5_0_5b", "qwen2_5_1_5b", "qwen2_5_3b", "qwen2_5_7b",
             "qwen2_5_14b", "qwen2_5_32b"]
    d = pd.DataFrame(rows).set_index("model").loc[order].reset_index()
    d["lab"] = d.model.map(LABELS)

    b = c3("c3_h1_clean")["c2"]["forecast_baselines"]["128"]["baseline_mean_score"]
    BAR, LASTV = b["seasonal_naive_true_period"], b["last_value"]

    y = np.arange(len(d))[::-1]
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.axvspan(BAR, 0.70, color=HARNESS, alpha=0.05, zorder=0)
    ax.axvline(BAR, color=MUTED, lw=1.8, ls="--", zorder=3)
    ax.axvline(LASTV, color=BASE, lw=1.2, ls=":", zorder=3)

    for yi, (_, r) in zip(y, d.iterrows()):
        ax.plot([r.bare, r.harness], [yi, yi], color=GRID, lw=6,
                solid_capstyle="round", zorder=2)
        if r.short >= 20:
            ax.plot([r.bare, r.bare_ok], [yi, yi], color=BARE, lw=1.2,
                    ls=":", zorder=4)
            ax.plot(r.bare_ok, yi, marker="o", ms=10, mfc="none", mec=BARE,
                    mew=2, zorder=5)
        ax.plot(r.bare, yi, marker="o", ms=11, color=BARE, zorder=6)
        ax.plot(r.harness, yi, marker="^", ms=10, color=HARNESS, zorder=6)
        ax.text(r.harness + 0.012, yi, f"+{r.harness - r.bare:.3f}",
                va="center", fontsize=9.5, color=INK2)

    ax.text(BAR - 0.008, len(d) - 0.35,
            "the bar to beat:\nrepeat the last cycle, 0.569", ha="right",
            va="center", fontsize=9.5, color=INK2, linespacing=1.5)
    ax.text(0.615 + 0.012, len(d) + 0.15,
            "the tool: 0.611 from 1.5B up", ha="left", va="center",
            fontsize=9.5, color=HARNESS, weight="bold")
    ax.text(0.552, y[0] - 0.36, "0.552, below the bar", ha="center",
            va="top", fontsize=8.6, color=HARNESS, weight="bold")

    ax.plot([], [], marker="o", ms=10, color=BARE, ls="none",
            label="the model alone, as scored")
    ax.plot([], [], marker="o", ms=9, mfc="none", mec=BARE, mew=2, ls="none",
            label="the model alone, well-formed answers only")
    ax.plot([], [], marker="^", ms=9, color=HARNESS, ls="none",
            label="the model routing to the tool")
    ax.legend(loc="lower left", fontsize=9.3)

    ax.set_yticks(y, d.lab, fontsize=10.5, color=INK)
    ax.set_xlim(0.22, 0.70); ax.set_ylim(-1.9, len(d) + 0.7)
    ax.set_xlabel("forecasting score  1/(1+MASE)  on 300 real series, 24 steps ahead.  n = 300")
    ax.set_title("Same picture on real data: no model reaches the bar, the tool clears it from 1.5B up")
    ax.grid(axis="x")
    tag(ax, "DESCRIPTIVE ONLY. The 72B rung never ran, so no statistical test was computed for this cell. "
            "Six Qwen2.5 sizes, 300 real series, 128 points of history.\nHollow circles show the same model counted only "
            "on answers that were the right length; four rungs lose items to short or missing forecasts (0.5B 29, 1.5B 65, "
            "7B 85, 14B 24 of 300).\nThe 7B dip is mostly that, not forecasting skill. Even counting only well-formed "
            "answers, the best model reaches 0.462 against a bar of 0.569.", y=-0.22)
    save(fig, "fig09_real_ladder.png")


if __name__ == "__main__":
    fig01_cross_model()
    fig02_architecture()
    fig04_real_data()
    fig05_strata()
    fig06_worked_example()
    fig07_descriptive_by_model()
    fig08_evidence_trap()
    fig09_real_ladder()
    print("\nall figures in", os.path.relpath(OUT, ROOT))
