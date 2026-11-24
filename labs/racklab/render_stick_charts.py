"""Render both published hardware charts from the public benchmark data."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


PAGE = Path(__file__).resolve().parents[2] / "docs/hardware/benchmarks"
BACKGROUND = "#241923"
INK = "#fff1dd"
MUTED = "#d2bfca"
LINE = "#51404e"
MINT = "#a3e8b9"
GOLD = "#f2c78e"
CPU = "#c4b3ee"


def figure(title, subtitle):
    fig = plt.figure(figsize=(8, 5.8), facecolor=BACKGROUND)
    fig.text(.075, .9, title, fontsize=26, weight="bold", color=INK)
    fig.text(.078, .838, subtitle, fontsize=13, color=MUTED)
    return fig


def axes(fig, position):
    ax = fig.add_axes(position, facecolor=BACKGROUND)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0, labelsize=12, colors=MUTED, pad=10)
    ax.set_axisbelow(True)
    return ax


def decision_chart(data):
    fig = figure("A 131,072-token decision", "MagicEightBall · Windows host + physical NCS2")
    ax = axes(fig, [.15, .17, .75, .57])
    timing = data["magiceightball"]["ncs2_support"]["timing"]
    points = ("1024", "32768", "131072")
    x = range(len(points))
    for key, color, label in (
        ("total_ms", MINT, "Host + NCS2"),
        ("cpu_reference_ms", CPU, "CPU reference"),
    ):
        values = [timing[size][key]["p50"] for size in points]
        ax.plot(x, values, color=color, linewidth=2.8, marker="o", markersize=8, label=label)
        dy = 11 if key == "total_ms" else -17
        for index, value in enumerate(values):
            offset = (28, -3) if key == "cpu_reference_ms" and index == 0 else (0, dy)
            ax.annotate(f"{value:.2f}", (index, value), xytext=offset,
                        textcoords="offset points", ha="center", color=color, fontsize=12, weight="bold")
    ax.set_xlim(-.27, 2.25)
    ax.set_ylim(0, 42)
    ax.set_yticks(range(0, 41, 10))
    ax.set_xticks(list(x), [f"{int(size):,}" for size in points])
    ax.grid(axis="y", color=LINE, linewidth=.8)
    ax.set_ylabel("Median wall time (ms)", color=MUTED, fontsize=12)
    ax.set_xlabel("Needle tokens per document", color=MUTED, fontsize=12, labelpad=10)
    ax.legend(loc="upper left", frameon=False, labelcolor=INK, fontsize=12)
    fig.savefig(PAGE / "decision-latency.png", dpi=165)
    plt.close(fig)


def room_chart(data):
    fig = figure("Listening on both sticks", "BreakShot · same 464 annotated DESED room clips")
    readings = data["breakshot"]["room"]
    labels = ["CPU", "NCS1", "NCS2"]
    panels = (
        (.17, .36, "detected_events", "Speech events detected ↑", readings["expected_events"], 200),
        (.62, .28, "false_wakes", "False wakes ↓", 110, 25),
    )
    for left, width, metric, title, limit, step in panels:
        ax = axes(fig, [left, .15, width, .56])
        ax.set_title(title, loc="left", color=INK, fontsize=14, weight="bold", pad=12)
        ax.set_ylim(-.58, 2.58)
        ax.set_xlim(0, limit * 1.06)
        ax.xaxis.set_major_locator(MultipleLocator(step))
        ax.set_yticks([2, 1, 0], labels if left < .5 else [])
        ax.grid(axis="x", color=LINE, linewidth=.8)
        for y, label, color in zip([2, 1, 0], labels, [CPU, GOLD, MINT]):
            value = readings[label.lower()][metric]
            ax.barh(y, limit, height=.48, color="#352634")
            ax.barh(y, value, height=.48, color=color)
            ax.text(value - limit * .018, y, str(value), va="center", ha="right",
                    fontsize=13, color="#211422", weight="bold")
    fig.text(.17, .053, f"{readings['expected_events']} annotated events", color=MUTED, fontsize=12)
    fig.savefig(PAGE / "breakshot-room.png", dpi=165)
    plt.close(fig)


def search_chart(data):
    fig = figure("Finding the source", "BankShot · same five editorial queries across three runtimes")
    ax = axes(fig, [.21, .18, .70, .57])
    bankshot = data["bankshot"]
    cases = ("lockout", "lathe", "exit", "recall", "recorder")
    for offset, lane, color, label in zip((.19, 0, -.19), ("cpu", "ncs1", "ncs2"),
                                          (CPU, GOLD, MINT), ("CPU", "NCS1", "NCS2")):
        times = {case["id"]: case["median_ms"] for case in bankshot["lanes"][lane]}
        ax.scatter([times[case] for case in cases], [index + offset for index in range(len(cases))],
                   s=56, color=color, label=label, zorder=3)
    ax.set_yticks(range(len(cases)), [case.title() for case in cases])
    ax.set_xlim(0, 17)
    ax.set_ylim(4.5, -.5)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.grid(axis="both", linewidth=.8, color=LINE)
    ax.set_xlabel("Median warm search time (ms)", color=MUTED, fontsize=12, labelpad=12)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.02), frameon=False,
              labelcolor=INK, fontsize=11, ncols=3)
    fig.savefig(PAGE / "bankshot-latency.png", dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    evidence = json.loads((PAGE / "data.json").read_text(encoding="utf8"))
    decision_chart(evidence)
    room_chart(evidence)
    search_chart(evidence)
