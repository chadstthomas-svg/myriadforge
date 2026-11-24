"""Estimate Llama payload sizes or probe resident dense weights on an NCS2."""

from argparse import ArgumentParser, SUPPRESS
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from myriadforge.myriad_x import allocation_properties, require_myriad_x
from worker_process import run_worker

MIB = 2**20
DENSE_WIDTH = 4096
LAYER_MIB = DENSE_WIDTH**2 * 2 // MIB
MARKER = "MYRIADFORGE_DENSE_PROBE "


def llama_budget(config, context, budget_mib, reserve_mib):
    """Plan an FP16 payload for a bias-free, batch-one gated Llama decoder."""
    if config.get("model_type") != "llama" or config.get("attention_bias") or config.get("mlp_bias"):
        raise ValueError("Supply a bias-free Llama config.json")
    hidden, inner = config["hidden_size"], config["intermediate_size"]
    heads, layers = config["num_attention_heads"], config["num_hidden_layers"]
    kv_heads = config.get("num_key_value_heads", heads)
    head_dim = config.get("head_dim", hidden // heads)
    query_width, kv_width = heads * head_dim, kv_heads * head_dim
    if not 0 < context <= config["max_position_embeddings"]:
        raise ValueError("Context is outside the model's configured window")
    embeddings = config["vocab_size"] * hidden * (1 if config.get("tie_word_embeddings") else 2)
    per_block = 2 * hidden * (query_width + kv_width) + 3 * hidden * inner + 2 * hidden
    parameters = embeddings + layers * per_block + hidden
    fp16_weights = 2 * parameters
    fp16_cache = 2 * 2 * layers * context * kv_width
    remainder = (budget_mib - reserve_mib) * MIB - fp16_weights - fp16_cache
    return {
        "scope": "Analytical batch-one estimate. Reserve is an input assumption.",
        "context": context, "parameter_count": parameters,
        "fp16_weight_bytes": fp16_weights, "fp16_kv_bytes": fp16_cache,
        "ideal_int4_weight_bytes": (parameters + 1) // 2,
        "budget_mib": budget_mib, "reserve_mib": reserve_mib,
        "remainder_mib": round(remainder / MIB, 3), "fits_fp16_budget": remainder >= 0,
    }


def probe(payload_mib, graph_count):
    import numpy as np
    from openvino.runtime import Core, Model, get_version, opset8 as op

    core = Core()
    report = {
        "runtime": get_version(),
        "shape": [1, DENSE_WIDTH], "seed": 24, "fp16_constants_mib": payload_mib,
        "dense_layers": payload_mib // LAYER_MIB, "lane": "MYRIAD",
        "graph_count": graph_count, "fp16_constants_mib_per_graph": payload_mib // graph_count,
    }
    rng = np.random.default_rng(24)
    feed = rng.uniform(-.5, .5, size=(1, DENSE_WIDTH)).astype(np.float32)
    control = feed.copy()

    try:
        report["device"] = require_myriad_x(core)
        loaded = []
        report["compile_ms_per_graph"] = []
        for index in range(graph_count):
            signal = op.parameter([1, DENSE_WIDTH], np.float32)
            graph = signal
            for _ in range(report["dense_layers"] // graph_count):
                weights = (rng.standard_normal((DENSE_WIDTH, DENSE_WIDTH), dtype=np.float32) / 256).astype(np.float16)
                control += np.maximum(control @ weights.astype(np.float32), 0)
                embedded = op.convert(op.constant(weights), np.float32)
                graph = op.add(graph, op.relu(op.matmul(graph, embedded, False, False)))
            print(f"Compiling graph {index + 1}/{graph_count} on MYRIAD", file=sys.stderr, flush=True)
            start = perf_counter_ns()
            compiled = core.compile_model(Model([graph], [signal], "Resident dense weights"), "MYRIAD",
                                          allocation_properties(16, 16, 1))
            duration = (perf_counter_ns() - start) / 1e6
            report["compile_ms_per_graph"].append(duration)
            print(f"Compiled in {duration:.1f} ms", file=sys.stderr, flush=True)
            loaded.append((compiled, compiled.create_infer_request()))
        samples = []
        for _ in range(6):
            values = feed
            start = perf_counter_ns()
            for compiled, request in loaded:
                values = request.infer({compiled.input(0): values})[compiled.output(0)]
            samples.append((perf_counter_ns() - start) / 1e6)
            if not np.allclose(values, control, rtol=.02, atol=.02):
                raise ArithmeticError(
                    f"Dense graph disagreed with CPU, max abs error {np.max(np.abs(values-control)):.6g}"
                )
        report.update(status="ok", first_ms=samples[0], warm_median_ms=median(samples[1:]),
                      max_abs_error=float(np.max(np.abs(values - control))))
    except (RuntimeError, ArithmeticError) as error:
        report.update(status="error", detail=str(error).strip()[-1600:])
    return report


def bounded_probe(args):
    command = [sys.executable, str(Path(__file__).resolve()), "probe", str(args.payload_mib),
               "--graphs", str(args.graphs), "--worker"]
    context = {"fp16_constants_mib": args.payload_mib, "graph_count": args.graphs}
    try:
        result = run_worker(command, args.timeout)
    except subprocess.TimeoutExpired as error:
        partial = error.stderr or ""
        return {**context, "status": "timeout", "timeout_sec": args.timeout, "last_progress": partial[-600:]}
    outcomes = [row[len(MARKER):] for row in result.stdout.splitlines() if row.startswith(MARKER)]
    if not outcomes or result.returncode != 0:
        return {**context, "status": "error", "detail": (result.stderr or result.stdout)[-1600:]}
    return json.loads(outcomes[-1])


def main():
    parser = ArgumentParser(description=__doc__)
    mode = parser.add_subparsers(dest="mode", required=True)
    calc = mode.add_parser("estimate")
    calc.add_argument("config", type=Path)
    calc.add_argument("--context", type=int, default=2048)
    calc.add_argument("--budget-mib", type=int, default=1024)
    calc.add_argument("--reserve-mib", type=int, default=128)
    physical = mode.add_parser("probe")
    physical.add_argument("payload_mib", type=int)
    physical.add_argument("--graphs", type=int, choices=(1, 2), default=1)
    physical.add_argument("--timeout", type=float, default=60, help="wall-clock seconds per probe")
    physical.add_argument("--worker", action="store_true", help=SUPPRESS)
    physical.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.mode == "estimate":
        if not 0 <= args.reserve_mib < args.budget_mib:
            parser.error("Reserve must be smaller than the positive budget")
        config = json.loads(args.config.read_text(encoding="utf8"))
        report = llama_budget(config, args.context, args.budget_mib, args.reserve_mib)
    else:
        increment = LAYER_MIB * args.graphs
        if not increment <= args.payload_mib <= 1024 or args.payload_mib % increment:
            parser.error(f"Use a multiple of {increment} MiB up to 1024 MiB")
        if not args.timeout > 0:
            parser.error("Timeout must be positive")
        if args.worker:
            print(MARKER + json.dumps(probe(args.payload_mib, args.graphs)), flush=True)
            return 0
        report = bounded_probe(args)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2), flush=True)
    return 1 if report.get("status") in ("error", "timeout") else 0


if __name__ == "__main__":
    sys.exit(main())
