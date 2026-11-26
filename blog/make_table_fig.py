"""Render a table as a PNG in the same style as the post's figures."""
import sys, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, INK, INK2, MUTED, GRID, BASE = ("#fcfcfb", "#0b0b0b", "#52514e",
                                      "#898781", "#e1e0d9", "#c3c2b7")
plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURF,
                     "savefig.facecolor": SURF})

def render(headers, rows, widths, out, align=None, wrap=None,
           fw=7.28, fs=11.5, pad=0.055, row_h=0.30, head_h=0.40):
    align = align or ["left"] * len(headers)
    wrap = wrap or [999] * len(headers)
    # wrap each cell, work out how many lines each row needs
    body = []
    for r in rows:
        cells = [textwrap.wrap(c, wrap[i]) or [""] for i, c in enumerate(r)]
        body.append((cells, max(len(c) for c in cells)))
    head = [textwrap.wrap(h, wrap[i]) or [""] for i, h in enumerate(headers)]
    head_lines = max(len(h) for h in head)

    line = 0.235                      # inches per text line
    total = head_h + head_lines * line + sum(head_h * 0 + n * line + 0.16
                                             for _, n in body) + 0.12
    fig = plt.figure(figsize=(fw, total), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, total)

    x = []
    acc = pad
    for w in widths:
        x.append(acc); acc += w
    def place(cells, y, weight, color):
        for i, lines in enumerate(cells):
            for j, ln in enumerate(lines):
                xi = x[i] if align[i] == "left" else x[i] + widths[i] / 2
                ax.text(xi, y - j * line, ln, fontsize=fs, color=color,
                        weight=weight, va="top",
                        ha="left" if align[i] == "left" else "center")

    y = total - 0.22
    ax.plot([pad, 1 - pad], [y + 0.06, y + 0.06], color=INK, lw=1.5)   # top rule
    place(head, y - 0.04, "bold", INK)
    y -= head_lines * line + 0.17
    ax.plot([pad, 1 - pad], [y + 0.04, y + 0.04], color=BASE, lw=1.0)  # header rule
    for cells, n in body:
        y -= 0.10
        place(cells, y, "normal", INK2)
        y -= n * line + 0.06
    ax.plot([pad, 1 - pad], [y + 0.08, y + 0.08], color=INK, lw=1.5)   # bottom rule
    fig.savefig(out, dpi=200, facecolor=SURF)
    print("wrote", out, f"({fw*200:.0f} x {total*200:.0f} px)")

render(
    ["Question type", "Where the answer comes from", "Carries a conclusion"],
    [["Forecasting", "the held-out continuation of the series", "yes"],
     ["Trend description", "the generator recipe", "yes"],
     ["Event explanation", "the generator recipe", "yes"],
     ["Mean, spike count, maximum, percent change, autocorrelation, largest step",
      "an exact calculation on the visible numbers", "no"]],
    widths=[0.33, 0.40, 0.17],
    align=["left", "left", "center"],
    wrap=[25, 32, 12],
    out="table1_question_types.png")

# compact variant: the six arithmetic types named in the caption instead
render(
    ["Question type", "Where the answer comes from", "Carries a conclusion"],
    [["Forecasting", "the held-out continuation of the series", "yes"],
     ["Trend description", "the generator recipe", "yes"],
     ["Event explanation", "the generator recipe", "yes"],
     ["Six arithmetic types", "an exact calculation on the visible numbers", "no"]],
    widths=[0.33, 0.40, 0.17],
    align=["left", "left", "center"],
    wrap=[25, 32, 12],
    out="table1_question_types_compact.png")

# Table 2: the four setups
render(
    ["Setup", "What it receives"],
    [["Model alone", "the numbers and the question"],
     ["Model and statistics",
      "the same, plus a block of computed statistics about the series"],
     ["Model and tools",
      "the same, but it first picks a tool from a menu, then calls it"],
     ["Model and tools, told which",
      "the same, but handed the correct tool without choosing"]],
    widths=[0.36, 0.54],
    align=["left", "left"],
    wrap=[28, 46],
    out="table2_four_setups.png")

# Table 3: effect of the parser correction
render(
    ["Configuration", "Replies with no usable forecast", "Score before", "Score after"],
    [["Phi-4", "3 of 100", "0.085", "0.364"],
     ["Mistral-Small-3.2", "1 of 100", "0.398", "0.417"],
     ["Qwen2.5 0.5B", "4 of 100", "0.335", "0.338"],
     ["gpt-oss-120b, high effort", "51 of 100", "0.248", "0.248"],
     ["Qwen2.5 7B", "13 of 100", "0.373", "0.375"],
     ["gpt-oss-20b", "12 of 100", "0.392", "0.392"],
     ["The remaining eight", "0 or 1 of 100", "unchanged within 0.002", ""]],
    widths=[0.28, 0.27, 0.19, 0.14],
    align=["left", "left", "left", "left"],
    wrap=[26, 16, 30, 12],
    out="table3_parser_correction.png")

# Table 4: the history-length sweep on the 300 real series
render(
    ["History shown", "Model alone", "Model and statistics", "Model and tools",
     "Naive, cycle given", "Naive, cycle guessed"],
    [["32",  "0.4442", "0.4403", "0.6079", "0.5688", "0.4960"],
     ["64",  "0.4840", "0.4582", "0.6214", "0.5688", "0.5825"],
     ["128", "0.4570", "0.4251", "0.6107", "0.5688", "0.5641"],
     ["256", "0.4833", "0.4409", "0.6285", "0.5688", "0.5592"],
     ["512", "0.4563", "0.4035", "0.6385", "0.5688", "0.5645"]],
    widths=[0.12, 0.13, 0.16, 0.17, 0.15, 0.15],
    align=["left"] * 6,
    wrap=[8, 8, 11, 9, 9, 10],
    out="table4_history_sweep.png")

# Table 5: the history-length test on both datasets
render(
    ["Dataset", "Change in the gap", "95 percent interval",
     "Result against the 0.02 threshold"],
    [["Generated, 300 series", "+0.0227", "+0.0049 to +0.0406", "passes, by 0.0027"],
     ["Real, 300 series", "+0.0185", "\u22120.0062 to +0.0436", "does not pass"]],
    widths=[0.26, 0.14, 0.24, 0.24],
    align=["left", "left", "left", "left"],
    wrap=[21, 10, 19, 19], fs=10.5,
    out="table5_history_test.png")
