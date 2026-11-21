"""Render a resident runtime comparison from matched MagicEightBall fixtures."""

import math

from matplotlib.patches import Circle, Rectangle

from render_size_charts import BOLD, COPPER, INK, LINES, MINT, MONO, SOFT, TEAL, canvas, label, panel, read_report, save


NATIVE_REPORT = "compact-gather-physical-ncs1.json"
PORTABLE_REPORT = "portable-cpu-comparison.json"
PROFILES = (
    ("onnx", "ONNX radix", "ONNX Runtime CPU"),
    ("fp32", "FP32 dense", "ONNX Runtime CPU"),
    ("gguf", "GGUF", "CPU table reader"),
)


def latency(wall_time):
    p50, p95 = wall_time["p50"], wall_time["p95"]
    if not 0 < p50 <= p95 or not math.isfinite(p95):
        raise ValueError("Expected positive wall-time percentiles")
    return p50, p95


def observations():
    native_report = read_report(NATIVE_REPORT)
    native = native_report["replays"]["compact"]
    cpu = read_report(PORTABLE_REPORT)
    for key in ("fixture_contract_sha256", "corpus_sha256", "truth_table_assignments", "full_context"):
        if native[key] != cpu[key]:
            raise ValueError(f"Runtime reports disagree on {key}")
    veto = native["full_context"][-1]
    if (not native_report["physical_device_verified"] or native["status"] != "passed"
            or native["truth_table_assignments"] != 128 or veto["tokens"] != 131_072
            or veto["name"] != "full_late_veto" or veto["result"]["action"] is not None):
        raise ValueError("Expected the verified NCS1 run with 128 decisions and the full-context veto")

    short = native["timings"]["short_128_vector"]
    long = native["timings"]["full_late_veto_one"]
    if short["openvino_requests"] != 1 or long["openvino_requests"] != 1:
        raise ValueError("Unexpected NCS1 head call count")
    rows = [("Compact NCS1", "OpenVINO MYRIAD", short["openvino_requests"],
             latency(short["wall_ms"]), latency(long["wall_ms"]))]

    for key, name, engine in PROFILES:
        profile = cpu["profiles"][key]
        workloads = profile["workloads"]
        short = workloads["short_128_native"]
        if "same_workload_as" in short:
            short = workloads[short["same_workload_as"]]
        long = workloads["full_late_veto_one"]
        if (short["documents"] != 128 or short["head_calls"] != 128 // profile["batch_capacity"]
                or long["documents"] != 1 or long["head_calls"] != 1):
            raise ValueError(f"Unexpected CPU workload for {key}")
        rows.append((name, engine, short["head_calls"], latency(short["summary"]["wall_ms"]),
                     latency(long["summary"]["wall_ms"])))
    return rows, native["repeats"], cpu["repeats"]


def axis_ticks(values):
    largest = max(p95 for _, p95 in values)
    target = largest / 4
    unit = 10 ** math.floor(math.log10(target))
    step = next(unit * factor for factor in (1, 2, 2.5, 5, 10) if unit * factor >= target)
    return [step * index for index in range(math.ceil(largest / step) + 1)]


def lane(ax, x, width, values, row_centers):
    ticks = axis_ticks(values)
    ceiling = ticks[-1]
    for tick in ticks:
        pos = x + tick / ceiling * width
        ax.plot([pos, pos], [411, 775], color=LINES, lw=.7, zorder=1)
        label(ax, pos, 798, f"{tick:g}", 13, SOFT, MONO, ha="center")

    for index, (y, (p50, p95)) in enumerate(zip(row_centers, values)):
        accent = MINT if index == 0 else TEAL
        median_x, tail_x = (x + value / ceiling * width for value in (p50, p95))
        label(ax, x + 3, y - 16, f"{p50:.2f} p50", 17, MINT if index == 0 else INK, MONO)
        label(ax, x + 175, y - 16, f"{p95:.2f} p95", 17, COPPER, MONO)
        ax.add_patch(Rectangle((x, y + 10), median_x - x, 11, facecolor=accent, edgecolor="none", zorder=2))
        ax.plot([median_x, tail_x], [y + 16, y + 16], color=COPPER, lw=2, zorder=3)
        ax.add_patch(Circle((median_x, y + 16), 4.5, facecolor=INK, edgecolor="none", zorder=4))
        ax.plot([tail_x, tail_x], [y + 8, y + 24], color=COPPER, lw=1.6, zorder=5)


def render():
    rows, native_repeats, cpu_repeats = observations()
    fig, ax = canvas(1010, 391, .48)
    label(ax, 81, 72, "MAGICEIGHTBALL / RUNTIME", 18, COPPER, MONO)
    label(ax, 73, 158, "Decision runtime", 66, INK, BOLD)
    label(ax, 81, 231, "One support policy across four execution paths", 24, MINT, BOLD)

    panel(ax, 72, 303, 1456, 533)
    label(ax, 110, 353, "RUNTIME", 15, COPPER, MONO)
    label(ax, 110, 385, "Calls per short workload", 17, SOFT)
    label(ax, 487, 353, "128 short decisions", 27, INK, BOLD)
    label(ax, 487, 385, "Wall time in ms per batch", 17, SOFT)
    label(ax, 965, 353, "131,072-token late veto", 27, INK, BOLD)
    label(ax, 965, 385, "Fresh document · 1 head call · ms", 17, SOFT)
    ax.plot([928, 928], [330, 807], color=LINES, lw=.9)

    row_centers = (460, 550, 640, 730)
    for index, (y, (name, engine, calls, _, _)) in enumerate(zip(row_centers, rows)):
        accent = MINT if index == 0 else INK
        label(ax, 110, y - 16, name, 24, accent, BOLD)
        label(ax, 110, y + 12, engine, 15, SOFT)
        label(ax, 334, y + 12, f"{calls} head {'call' if calls == 1 else 'calls'}", 13, accent, MONO)
    lane(ax, 487, 392, [row[3] for row in rows], row_centers)
    lane(ax, 965, 493, [row[4] for row in rows], row_centers)

    ax.plot([89, 1511], [871, 871], color=LINES, lw=1)
    ax.add_patch(Circle((89, 871), 4, facecolor=MINT, edgecolor="none", zorder=3))
    label(ax, 89, 900, "Resident runtimes. Wall time includes host text preparation, execution and decoding.", 17, INK)
    label(ax, 89, 932, f"Same host, separate sessions · {native_repeats} NCS1 and {cpu_repeats} CPU timed repeats per workload.", 16, SOFT)
    label(ax, 89, 969, f"Sources  {NATIVE_REPORT}  ·  {PORTABLE_REPORT}", 15, COPPER, MONO)
    save(fig, "runtime-comparison.webp")


if __name__ == "__main__":
    render()
