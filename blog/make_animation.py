"""The blog's required animation: one real item walked through the whole
pipeline, from the generated series to the scored result.

    python blog/make_animation.py

Writes blog/figs/anim_pipeline.gif (~9 s, 2 fps-ish, loops forever).
Every number shown is item b0083 / b0042 from the archived cycle-2 run.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCH = os.path.join(ROOT, "results-archive", "20261113-cycle2-run")
OUT = os.path.join(ROOT, "blog", "figs", "anim_pipeline.gif")

HARNESS, BARE, FEAT, ORACLE = "#2a78d6", "#eb6834", "#eda100", "#4a3aa7"
NEG, RULE, TRUTH = "#e34948", "#1baf7a", "#0b0b0b"
SURF, INK, INK2, MUTED, GRID, BASE = ("#fcfcfb", "#0b0b0b", "#52514e",
                                      "#898781", "#e1e0d9", "#c3c2b7")
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "axes.edgecolor": BASE,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})

z = np.load(os.path.join(ARCH, "data", "c2_n100", "pool.npz"))
FULL = z["s83"]
HIST, TGT = FULL[512 - 128:512], FULL[512:520]
PRED_BARE = np.array([57.23, 57.04, 56.85, 56.67, 56.49, 56.31, 56.13, 55.95])
PRED_HARN = np.array([60.3666, 62.0568, 64.23, 67.1276, 68.8179, 72.1984,
                      73.6472, 75.096])
DEN = 2.199596905291801

STEPS = [
    ("1.  Generate", "A series built from a known recipe: slowly falling,\n"
                     "repeats every 24 steps, no sudden events.\n"
                     "The recipe IS the answer — and it never leaves\n"
                     "this step."),
    ("2.  Ask", "One of 100 wordings of the same question is drawn.\n"
                "The wording varies; what is being asked does not.\n"
                "\"Give the next 8 values, comma-separated.\""),
    ("3.  Hide the answer", "Five fields cross over and nothing else: id,\n"
                            "question, numbers, length, horizon. A checker reads\n"
                            "every file on the other side and fails the build if\n"
                            "any of them mentions the answer."),
    ("4.  Four setups", "The model alone gets 128 numbers in the prompt.\n"
                        "The tool setup gets the same, but first asks the model\n"
                        "which tool to use — and then calls that tool."),
    ("5.  Answer", "model alone:   57.23, 57.04, ... 55.95  (continues the dip)\n"
                   "model + tools: picks `forecast`; the tool returns\n"
                   "               60.37, 62.06, ... 75.10  (continues the cycle)"),
    ("6.  Score", "Both answers are scored with the same metric against the\n"
                  "same hidden future, on a 0-to-1 scale where 0.5 ties\n"
                  "the dumb baseline.  model alone 0.18, model + tools 0.63"),
    ("7.  Repeat everything", "100 questions per cell, then the whole test again\n"
                              "for 14 model setups, and again on 300 real series.\n"
                              "The tool beats the model in all 14."),
]
FRAMES_PER_STEP = 6
N = len(STEPS) * FRAMES_PER_STEP

fig = plt.figure(figsize=(11, 6.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.72], hspace=0.52,
                      left=0.075, right=0.975, top=0.84, bottom=0.08)
axs = fig.add_subplot(gs[0])
axd = fig.add_subplot(gs[1])
fig.suptitle("One question, end to end", x=0.075, ha="left",
             fontsize=15, weight="bold", y=0.965, color=INK)
_counter = fig.text(0.975, 0.965, "", ha="right", fontsize=10, color=MUTED,
                    va="center")


def draw(frame):
    step = min(frame // FRAMES_PER_STEP, len(STEPS) - 1)
    sub = frame % FRAMES_PER_STEP
    title, body = STEPS[step]

    axs.clear(); axd.clear()
    axs.grid(axis="y"); axs.set_xlabel("time step", fontsize=9, color=INK2)
    axs.set_ylabel("series value", fontsize=9, color=INK2)
    axs.set_xlim(-131, 36); axs.set_ylim(48.5, 86)
    axs.tick_params(labelsize=8.5)

    # the series is revealed in step 1, then stays
    if step == 0:
        k = int(len(HIST) * (sub + 1) / FRAMES_PER_STEP)
        axs.plot(np.arange(-128, -128 + k), HIST[:k], color=MUTED, lw=1.5)
        axs.plot(np.arange(-128, 8)[128:], np.full(8, np.nan))
        axs.text(-128, 83.5, "the recipe also records: falling, repeats every 24 steps, no event",
                 fontsize=8.5, color=NEG, style="italic")
    else:
        axs.plot(np.arange(-128, 0), HIST, color=MUTED, lw=1.5,
                 label="the history both sides saw")

    if step >= 1:
        axs.axvspan(-0.5, 8, color=BASE, alpha=0.22)
        axs.text(3.7, 85.2, "the 8 steps\nbeing asked for", fontsize=8,
                 color=INK2, ha="center", va="top")
    if step == 2:
        axs.text(-128, 83.5, "the numbers and the question cross over. The recipe does not.",
                 fontsize=8.5, color=HARNESS, style="italic")
    if step >= 4:
        xf = np.arange(0, 8)
        kk = 8 if step > 4 else int(8 * (sub + 1) / FRAMES_PER_STEP)
        if kk:
            axs.plot(np.r_[-1, xf[:kk]], np.r_[HIST[-1], PRED_BARE[:kk]],
                     color=BARE, lw=2.1, ls="--", marker="s", ms=4,
                     label="the model's answer")
            axs.plot(np.r_[-1, xf[:kk]], np.r_[HIST[-1], PRED_HARN[:kk]],
                     color=HARNESS, lw=2.3, marker="^", ms=5,
                     label="the tool's answer")
    if step >= 5:
        axs.plot(np.r_[-1, np.arange(8)], np.r_[HIST[-1], TGT], color=TRUTH,
                 lw=2.4, marker="o", ms=4, label="what actually happened")
        axs.annotate("score 0.18", (7, PRED_BARE[-1]), xytext=(9.2, 53.0),
                     color=BARE, fontsize=9.5, weight="bold")
        axs.annotate("score 0.63", (7, PRED_HARN[-1]), xytext=(9.2, 76.5),
                     color=HARNESS, fontsize=9.5, weight="bold")
    if step >= 1:
        axs.legend(loc="lower left", fontsize=8, ncols=2, frameon=True,
                   framealpha=0.95, facecolor=SURF, edgecolor="none")

    # ---- the pipeline strip
    axd.set_xlim(0, 100); axd.set_ylim(0, 34); axd.axis("off")
    names = ["generate", "ask", "hide key", "four setups", "answer", "score",
             "repeat"]
    for i, nm in enumerate(names):
        x = 1.5 + i * 14.0
        done, now = i < step, i == step
        fc = "#e8f1fc" if now else ("#f4f3ef" if done else SURF)
        ec = HARNESS if now else (BASE if done else GRID)
        axd.add_patch(FancyBboxPatch((x, 24), 11.5, 7,
                                     boxstyle="round,pad=0.4", fc=fc, ec=ec,
                                     lw=2.0 if now else 1.0))
        axd.text(x + 5.75, 27.5, nm, ha="center", va="center", fontsize=8.6,
                 weight="bold" if now else "normal",
                 color=INK if (now or done) else MUTED)
        if i < 6:
            axd.add_patch(FancyArrowPatch((x + 12.1, 27.5), (x + 13.9, 27.5),
                                          arrowstyle="-|>", mutation_scale=9,
                                          lw=1.1,
                                          color=HARNESS if done else GRID))
    axd.text(0, 18, title, fontsize=14, weight="bold", color=INK, va="top")
    axd.text(0, 13.2, body, fontsize=10, color=INK2, va="top", linespacing=1.7)

    _counter.set_text(f"step {step + 1} of {len(STEPS)}")
    return []


anim = FuncAnimation(fig, draw, frames=N, interval=180, blit=False)
anim.save(OUT, writer=PillowWriter(fps=5), dpi=96)
print("wrote", os.path.relpath(OUT, ROOT),
      f"({os.path.getsize(OUT) / 1e6:.1f} MB)")
