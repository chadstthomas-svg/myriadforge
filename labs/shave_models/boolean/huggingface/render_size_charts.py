"""Render the MagicEightBall size cards using the workbench artwork and measured artifacts."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image, ImageOps


HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
LAB = HERE.parent
MEASUREMENTS = LAB / "measurements"
WIDTH = 1600
BG = "#100c17"
PANEL = "#1c1424"
INK = "#fff5e8"
SOFT = "#c1b5cb"
MINT = "#a4f8ba"
TEAL = "#67b6a4"
COPPER = "#fdc589"
LINES = "#453449"
WINDOWS_FONTS = Path("C:/Windows/Fonts")


def font(face, fallback):
    path = WINDOWS_FONTS / face
    return FontProperties(fname=path) if path.is_file() else FontProperties(family=fallback)


REGULAR = font("segoeui.ttf", "DejaVu Sans")
BOLD = font("segoeuib.ttf", "DejaVu Sans:bold")
MONO = font("consola.ttf", "DejaVu Sans Mono")


def read_report(name):
    return json.loads((MEASUREMENTS / name).read_bytes())


def observations():
    compact = read_report("compact-gather-physical-ncs1.json")
    comparison = read_report("ncs1-cpu-comparison.json")["results"]
    diagram = comparison["tree-cpu"]["artifacts"]["generated_binary_decision_diagram"]["bytes"]
    lookup = comparison["table-cpu"]["artifacts"]["generated_uint8_truth_table"]["bytes"]
    compact_archive = (LAB / "bundles/magiceightball-ncs1-compact.m8b").stat().st_size
    tokenizer = (LAB / "needle3-tokenizer.m8t.xz").stat().st_size
    return {
        "native_executables": [
            ["MagicEightBall compact", compact["candidate_bytes"], "compact-gather-physical-ncs1.json"],
            ["Stock radix Gather", compact["stock_bytes"], "compact-gather-physical-ncs1.json"],
            *[[label, read_report(report)["bytes"]["compiled_blob"], report] for label, report in (
                ("Packed convolution", "fast-packed-1.json"), ("Lookup", "fast-lookup-1.json"),
                ("Boolean gates", "fast-gates-1.json"), ("Dense reference", "fast-dense-1.json"),
            )],
        ],
        "portable_files": [
            ["ONNX radix Gather", (LAB / "bundles/magiceightball-radix/model.onnx").stat().st_size,
             "https://huggingface.co/tablinum/MagicEightBall-ONNX/blob/main/verification.json"],
            ["GGUF", 1024, "https://huggingface.co/tablinum/MagicEightBall-GGUF/blob/main/verification.json"],
            ["FP32 dense reference", (LAB / "bundles/dense/model.onnx").stat().st_size,
             "https://huggingface.co/tablinum/MagicEightBall-FP32/blob/main/verification.json"],
        ],
        "decision_data": [["CPU decision diagram", diagram], ["CPU lookup table", lookup]],
        "deployment": {"compact_archive": compact_archive, "shared_tokenizer": tokenizer},
        "published_examples": [
            ["Bonsai", "300 B", "Search ranking", "Bing ranking example",
             "https://proceedings.mlr.press/v70/kumar17a/kumar17a.pdf"],
            ["FastGRNN", "1 KB", "Wake-word recognition", "Wakeword-2, Table 3",
             "https://proceedings.neurips.cc/paper_files/paper/2018/file/ab013ca67cf2d50796b0c11d1b8bc95d-Paper.pdf"],
            ["fastText.zip", "16 KiB", "Text classification", "Pruned model, Table 2",
             "https://arxiv.org/pdf/1612.03651"],
        ],
    }


def canvas(height, art_height, focus):
    fig = plt.figure(figsize=(WIDTH / 100, height / 100), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(height, 0))
    ax.set_axis_off()
    with Image.open(ASSETS / "workbench.webp") as poster:
        art = ImageOps.fit(poster.convert("RGB"), (WIDTH, art_height), centering=(0.5, focus))
        ax.imshow(art, extent=(0, WIDTH, art_height, 0), aspect="auto")
    x = np.linspace(0, 1, WIDTH)
    y = np.linspace(0, 1, art_height)
    shade = np.zeros((art_height, WIDTH, 4))
    shade[:, :, :3] = np.array([16, 12, 23]) / 255
    shade[:, :, 3] = np.maximum(.88 * np.clip(1 - (x - .19) / .48, 0, 1)[None, :],
                                np.clip((y - .71) / .29, 0, 1)[:, None])
    ax.imshow(shade, extent=(0, WIDTH, art_height, 0), aspect="auto")
    return fig, ax


def label(ax, x, y, message, size, color=INK, face=REGULAR, **align):
    ax.text(x, y, message, color=color, fontsize=size, fontproperties=face, va="center", **align)


def panel(ax, x, y, width, height):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0,rounding_size=20",
                                facecolor=PANEL, edgecolor=LINES, linewidth=.9))


def strip(ax, x, y, width, text):
    ax.plot([x, x + width], [y, y], color=LINES, lw=1, solid_capstyle="round")
    ax.add_patch(Circle((x, y), 4, facecolor=MINT, edgecolor="none", zorder=3))
    label(ax, x + 1, y + 29, text, 15, SOFT)


def bars(ax, rows, y, step, ceiling, ticks, first=MINT):
    x, width = 489, 878
    for tick in ticks:
        pos = x + tick / ceiling * width
        ax.plot([pos, pos], [y - 24, y + (len(rows) - 1) * step + 27], color=LINES, lw=.75, zorder=1)
        label(ax, pos, y + (len(rows) - 1) * step + 46, f"{tick:,}", 13, SOFT, MONO, ha="center")
    for index, (name, value, *_) in enumerate(rows):
        at = y + index * step
        strong = index == 0
        label(ax, 110, at, name, 19, INK if strong else SOFT, BOLD if strong else REGULAR)
        length = value / ceiling * width
        ax.add_patch(Rectangle((x, at - 9), length, 18, color=first if strong else TEAL,
                               ec="none", alpha=1 if strong else .67, zorder=2))
        ax.add_patch(Circle((x + length, at), 4.5, facecolor=INK if strong else TEAL,
                            edgecolor="none", zorder=3))
        label(ax, x + length + 16, at, f"{value:,} B", 18, MINT if strong else INK, MONO)


def save(fig, name):
    fig.savefig(ASSETS / name, pil_kwargs={"quality": 92, "method": 6})
    plt.close(fig)


def native(data):
    fig, ax = canvas(960, 414, .48)
    compact = data["native_executables"][0][1]
    saved = data["native_executables"][1][1] - compact
    label(ax, 80, 75, "MAGICEIGHTBALL / NCS1", 18, COPPER, MONO)
    label(ax, 72, 177, f"{compact:,} bytes", 95, MINT, BOLD)
    label(ax, 81, 267, "Native decision executable on the stick", 24, INK, BOLD)
    label(ax, 82, 315, f"{saved:,} bytes smaller than its padded radix build", 18, SOFT)
    panel(ax, 72, 393, 1456, 494)
    label(ax, 110, 441, "Six NCS1 builds of the same support policy", 26, INK, BOLD)
    label(ax, 1369, 444, "BYTES", 15, COPPER, MONO, ha="right")
    bars(ax, data["native_executables"], 515, 53, 4500, (0, 1000, 2000, 3000, 4000))
    strip(ax, 89, 912, 1422, "Physical NCS1 replay. Sizes count native executables and exclude the shared tokenizer.")
    save(fig, "size-native-ncs1.webp")


def portable(data):
    fig, ax = canvas(910, 359, .52)
    label(ax, 80, 73, "MAGICEIGHTBALL / PORTABLE", 18, COPPER, MONO)
    label(ax, 76, 152, "One policy.", 52, INK, BOLD)
    label(ax, 76, 224, "Three CPU formats.", 48, MINT, BOLD)
    panel(ax, 72, 335, 1456, 282)
    label(ax, 110, 376, "Model file sizes", 25, INK, BOLD)
    label(ax, 1370, 376, "BYTES", 15, COPPER, MONO, ha="right")
    bars(ax, data["portable_files"], 442, 52, 2000, (0, 500, 1000, 1500, 2000))
    panel(ax, 72, 647, 1456, 173)
    label(ax, 110, 683, "CPU decision data only", 21, INK, BOLD)
    for x, (name, value) in zip((110, 796), data["decision_data"]):
        label(ax, x, 758, f"{value} B", 41, MINT, MONO)
        label(ax, x + 224, 757, name, 20, SOFT)
    strip(ax, 89, 850, 1422, "The CPU file and decision-data figures exclude the shared tokenizer and runner.")
    save(fig, "size-portable-cpu.webp")


def published(data):
    fig, ax = canvas(731, 345, .54)
    label(ax, 80, 70, "SIZE IN CONTEXT", 18, COPPER, MONO)
    label(ax, 76, 151, "Compact models", 54, INK, BOLD)
    label(ax, 76, 224, "Across workloads", 49, MINT, BOLD)
    compact = data["native_executables"][0][1]
    observed = ["MagicEightBall", f"{compact:,} B", "Support decisions", "Native NCS1 executable", ""]
    for index, (name, extent, workload, source, _) in enumerate([observed] + data["published_examples"]):
        x = 72 + index * 368
        panel(ax, x, 321, 352, 306)
        label(ax, x + 25, 366, name, 21, INK, BOLD)
        label(ax, x + 23, 443, extent, 42, MINT if index == 0 else COPPER, MONO)
        label(ax, x + 25, 517, workload, 17, INK)
        label(ax, x + 25, 562, source, 14, SOFT)
    strip(ax, 89, 661, 1422, "Sizes refer to different workloads and storage boundaries. Paper figures use their reported units.")
    save(fig, "size-published-models.webp")


if __name__ == "__main__":
    data = observations()
    native(data)
    portable(data)
    published(data)
    (HERE / "size-chart-data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf8")
