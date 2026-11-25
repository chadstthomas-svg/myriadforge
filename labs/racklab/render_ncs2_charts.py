"""Build public NCS2 charts and a compact summary from pinned measurements."""

import json
from statistics import median

from render_stick_charts import BACKGROUND, CPU, GOLD, INK, LINE, MINT, MUTED, PAGE, axes, figure, plt

MEASUREMENTS = PAGE.parents[2] / "tools/ncs2/measurements"


def report(name):
    return json.loads((MEASUREMENTS / name).read_text(encoding="utf8"))


def allocation_summary(mobile, allocation):
    trials = [row for row in mobile["trials"] if row["status"] == "ok" and row["allocation"] == allocation]
    return {"allocation": allocation, "runs": len(trials),
            "requests_per_second": median(row["requests_per_sec"] for row in trials),
            "single_request_p95_ms": median(row["p95_ms"] for row in trials)}


def summary():
    mobile = report("mobilenetv2-tuning.json")
    decoder = report("decoder-search-v1.json")
    embeddings = report("embeddinggemma2-ncs2-v1.json")
    memory = report("llm-residency.json")
    selected = decoder["tuned_qualification"]
    soak = next(row for row in decoder["tuning"]["steps"] if row["phase"] == "soak" and row["status"] == "ok")
    return {
        "schema": "racklab.ncs2-research/1",
        "measured_on": decoder["observed_date"],
        "device": selected["device"],
        "openvino": selected["runtime"],
        "mobilenetv2": {
            "stock": allocation_summary(mobile, mobile["default_allocation"]),
            "throughput_profile": allocation_summary(mobile, mobile["recommended"]["throughput"]),
        },
        "full_decoder": {
            "candidates": [{key: row[key] for key in (
                "checkpoint", "parameters", "myriad_weight_bytes", "matched", "comparisons",
                "clear_disagreements", "status",
            )} for row in decoder["candidates"]],
            "profile": decoder["tuning"]["profile"],
            "tuned_tokens_per_second": selected["decode_ms"]["myriad"]["tokens_per_second"],
            "soak": soak,
            "longer_windows": [{key: row[key] for key in (
                "context", "matched", "comparisons", "clear_disagreements", "status",
            )} for row in decoder["context_and_throughput"] if row["trial"].startswith("window-")],
        },
        "dense_probes": [{key: row[key] for key in ("fp16_constants_mib", "graph_count", "status")}
                         for row in memory["dense_probes"]],
        "embeddinggemma2": {
            "checkpoint": embeddings["checkpoint"],
            "trials": [{key: row[key] for key in (
                "mode", "text_count", "cpu_median_ms", "hybrid_median_ms", "min_embedding_cosine",
                "device_block", "parity", "host_text_parameters",
            )} for row in embeddings["recorded_projections"].values()],
        },
    }


def panel(ax, title, axis_label, labels, values, limit, color):
    ax.set_title(title, loc="left", color=INK, fontsize=14, weight="bold", pad=10)
    ax.set_xlim(0, limit)
    ax.set_yticks([1, 0], labels)
    ax.grid(axis="x", color=LINE, linewidth=.8)
    ax.set_xlabel(axis_label, color=MUTED, fontsize=12, labelpad=8)
    for y, value, fill in zip((1, 0), values, (GOLD, color)):
        ax.barh(y, value, height=.54, color=fill)
        ax.text(value - limit * .02, y, f"{value:.2f}", va="center", ha="right",
                color=BACKGROUND, weight="bold", fontsize=12)


def tuning_chart(data):
    fig = figure("NCS2 resource tuning", "Allocations show SHAVEs / CMX slices / streams")
    profile = data["full_decoder"]["profile"]
    mobile = data["mobilenetv2"]
    panel(axes(fig, [.25, .50, .64, .22]), "TinyStories 260K", "Warm token p95 (ms) ↓",
          ["16 / 16 / 1", "1 / 1 / 1"], [profile["stock_p95_ms"], profile["confirmation_p95_ms"]], 49, MINT)
    panel(axes(fig, [.25, .13, .64, .22]), "MobileNetV2", "Inferences per second ↑",
          ["16 / 16 / 1", "4 / 8 / 2"], [mobile["stock"]["requests_per_second"],
                                               mobile["throughput_profile"]["requests_per_second"]], 80, MINT)
    fig.savefig(PAGE / "ncs2-allocation-tuning.png", dpi=165)
    plt.close(fig)


def embedding_chart(data):
    fig = figure("EmbeddingGemma 2", "Same 15 texts · resident CPU and hybrid runs")
    ax = axes(fig, [.29, .19, .61, .50])
    head, both, _ = data["embeddinggemma2"]["trials"]
    for y, trial in enumerate((both, head)):
        for offset, key, label, fill in (
            (.17, "cpu_median_ms", "CPU", CPU),
            (-.17, "hybrid_median_ms", "Host + NCS2", MINT),
        ):
            value = trial[key] / 1000
            ax.barh(y + offset, value, height=.28, color=fill, label=label if y == 0 else None)
            ax.text(value + .035, y + offset, f"{value:.2f}", va="center", color=fill, fontsize=12, weight="bold")
    ax.set_ylim(-.48, 1.48)
    ax.set_xlim(0, 2.38)
    ax.set_yticks([1, 0], ["Output head", "Output +\nper-layer heads"])
    ax.grid(axis="x", color=LINE, linewidth=.8)
    ax.set_xlabel("Median end-to-end time (s)", color=MUTED, fontsize=12, labelpad=10)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, labelcolor=INK, fontsize=12, ncols=2)
    fig.savefig(PAGE / "ncs2-embedding-offload.png", dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    evidence = summary()
    (PAGE / "ncs2-research.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf8")
    tuning_chart(evidence)
    embedding_chart(evidence)
