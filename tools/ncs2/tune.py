"""Tune resource allocations on one physical NCS2 with CPU output checks."""

from argparse import ArgumentParser, SUPPRESS
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from statistics import median
import subprocess
import sys
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from myriadforge.myriad_x import (
    allocation_properties, compile_myriad_x, model_fingerprint, prepare_static_model,
    require_myriad_x,
)
from worker_process import run_worker

BASELINE = (16, 16, 1)
DEFAULT_GRID = [(n, n, 1) for n in range(1, 17)] + [(n, n, 2) for n in range(1, 9)]
DEFAULT_GRID += [(s, c, 1) for s, c in ((4, 8), (4, 16), (8, 12), (8, 16), (12, 16))]
DEFAULT_GRID += [(4, 8, 2)]
FIELDS = ("shaves", "cmx_slices", "streams")
MARKER = "MYRIADFORGE_TUNING "


def allocation(value):
    try:
        result = tuple(map(int, value.split(":")))
        allocation_properties(*result)
    except (TypeError, ValueError) as error:
        raise ValueError("Use SHAVES:CMX:STREAMS within the Myriad X resource limits") from error
    return result


def inputs(model, seeds, file):
    import numpy as np

    ports = list(model.inputs)
    specs = [(port.get_any_name(), tuple(port.shape), port.element_type.to_dtype()) for port in ports]
    if file:
        rows = json.loads(file.read_text(encoding="utf8"))
        result = []
        for row in rows:
            if set(row) != {name for name, _, _ in specs}:
                raise ValueError("Each feed must have exactly the model's input names")
            result.append({name: np.asarray(row[name], dtype=dtype).reshape(shape)
                           for name, shape, dtype in specs})
        if not result:
            raise ValueError("At least one feed is required")
        return result

    result = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        result.append({
            name: (rng.uniform(0, 1, size=shape).astype(dtype) if np.issubdtype(dtype, np.floating)
                   else np.zeros(shape, dtype=dtype))
            for name, shape, dtype in specs
        })
    return result


def trial(args):
    import numpy as np
    from openvino.runtime import Core, get_version

    core = Core()
    name = require_myriad_x(core)
    started_compile = perf_counter_ns()
    device = compile_myriad_x(core, args.model, **dict(zip(FIELDS, args.worker)))
    compiled_at = perf_counter_ns()
    requests = [device.create_infer_request() for _ in range(args.in_flight)]
    static = prepare_static_model(core, args.model)
    feeds = inputs(static, (24, 42, 73), args.feeds)
    cpu = core.compile_model(static, "CPU", {"INFERENCE_NUM_THREADS": 1}).create_infer_request()
    expected = [[value.copy() for value in cpu.infer(feed).values()] for feed in feeds]
    if args.check == "top1" and any(
        value.ndim != 2 or value.shape[-1] < 2 for value in expected[0]
    ):
        raise ValueError("Top-1 checking requires classification logits shaped [batch, classes]")
    max_abs_error = 0.0

    def check(request, index):
        nonlocal max_abs_error
        for port, target in zip(device.outputs, expected[index]):
            observed = request.get_tensor(port).data
            if observed.shape != target.shape or not np.isfinite(observed).all():
                raise ArithmeticError("Output shape or values disagreed with CPU")
            difference = np.abs(observed.astype(np.float64) - target)
            max_abs_error = max(max_abs_error, float(difference.max(initial=0)))
            if args.check == "top1":
                matched = np.array_equal(observed.argmax(axis=-1), target.argmax(axis=-1))
            elif np.issubdtype(target.dtype, np.inexact):
                matched = np.allclose(observed, target, rtol=args.rtol, atol=args.atol)
            else:
                matched = np.array_equal(observed, target)
            if not matched:
                raise ArithmeticError(
                    f"CPU parity failed for feed {index} output {port.any_name}. "
                    f"Maximum absolute error {difference.max():.6g}, "
                    f"CPU top index {target.argmax()}, MYRIAD top index {observed.argmax()}"
                )

    for step in range(max(args.warmup, len(feeds))):
        index = step % len(feeds)
        requests[0].infer(feeds[index])
        check(requests[0], index)

    def queue(first, count):
        for offset in range(count):
            requests[offset].start_async(feeds[(first + offset) % len(feeds)])
        for offset in range(count):
            requests[offset].wait()

    queue(0, args.in_flight)
    for offset, request in enumerate(requests):
        check(request, offset % len(feeds))

    samples = []
    for step in range(args.runs):
        start = perf_counter_ns()
        requests[0].infer(feeds[step % len(feeds)])
        samples.append((perf_counter_ns() - start) / 1e6)
    start = perf_counter_ns()
    for step in range(0, args.runs, args.in_flight):
        queue(step, min(args.in_flight, args.runs - step))
    rate = args.runs * 1e9 / (perf_counter_ns() - start)
    check(requests[0], (args.runs - 1) // args.in_flight * args.in_flight % len(feeds))
    return {"status": "ok", "device": name, "runtime": get_version(), "samples_ms": samples,
            "compile_ms": (compiled_at - started_compile) / 1e6,
            "p50_ms": median(samples), "p95_ms": float(np.percentile(samples, 95)),
            "requests_per_sec": rate, "cpu_parity_feeds": len(feeds),
            "max_abs_error": max_abs_error}


def run_trial(args, config, phase):
    command = [
        sys.executable, str(Path(__file__).resolve()), str(args.model),
        "--worker", ":".join(map(str, config)), "--runs", str(args.runs),
        "--warmup", str(args.warmup), "--in-flight", str(args.in_flight),
        "--rtol", str(args.rtol), "--atol", str(args.atol), "--check", args.check,
    ]
    if args.feeds:
        command += ["--feeds", str(args.feeds)]
    try:
        proc = run_worker(command, args.timeout)
        payloads = [line[len(MARKER):] for line in proc.stdout.splitlines() if line.startswith(MARKER)]
        if payloads and proc.returncode == 0:
            result = json.loads(payloads[-1])
        else:
            result = {"status": "error", "detail": "worker exited %d" % proc.returncode,
                      "log_tail": (proc.stderr or proc.stdout)[-2000:]}
    except subprocess.TimeoutExpired as error:
        result = {"status": "timeout", "detail": "trial exceeded %s s" % error.timeout}
    result.update(allocation=dict(zip(FIELDS, config)), phase=phase)
    print(phase, ":".join(map(str, config)), result["status"],
          f'{result.get("p95_ms", 0):.3f} ms p95, {result.get("requests_per_sec", 0):.1f} req/s',
          flush=True)
    return result


def recommend(trials):
    scores = {}
    for run in trials:
        key = tuple(run["allocation"][field] for field in FIELDS)
        scores.setdefault(key, []).append(run)
    eligible = {key: rows for key, rows in scores.items()
                if len(rows) >= 3 and all(row["status"] == "ok" for row in rows)}
    if not eligible:
        return {}
    latency = min(eligible, key=lambda cfg: median(row["p95_ms"] for row in eligible[cfg]))
    throughput = max(eligible, key=lambda cfg: median(row["requests_per_sec"] for row in eligible[cfg]))
    return {key: dict(zip(FIELDS, config)) for key, config in (("latency", latency), ("throughput", throughput))}


def sweep(args):
    from openvino.runtime import Core, get_version

    identity = require_myriad_x(Core())
    grid = list(dict.fromkeys([BASELINE, *(args.alloc or DEFAULT_GRID)]))
    rng = random.Random(24)
    rng.shuffle(grid)
    report = {
        "schema": "myriadforge.ncs2-tuning/1", "date_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model.name, "model_sha256": model_fingerprint(args.model),
        "runtime": get_version(), "device": identity, "default_allocation": dict(zip(FIELDS, BASELINE)),
        "settings": {"warmup": args.warmup, "runs": args.runs, "in_flight": args.in_flight,
                     "atol": args.atol, "rtol": args.rtol, "check": args.check, "order_seed": 24,
                     "feed_sha256": model_fingerprint(args.feeds).get(".json") if args.feeds else None},
        "trials": [], "recommended": {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def measure(config, phase):
        report["trials"].append(run_trial(args, config, phase))
        partial = args.out.with_suffix(args.out.suffix + ".tmp")
        partial.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
        partial.replace(args.out)

    for config in grid:
        measure(config, "sweep")

    passed = [run for run in report["trials"] if run["status"] == "ok"]
    fastest = sorted(passed, key=lambda row: row["p95_ms"])[:3]
    busiest = sorted(passed, key=lambda row: -row["requests_per_sec"])[:3]
    finalists = [tuple(row["allocation"][field] for field in FIELDS) for row in fastest + busiest]
    finalists = list(dict.fromkeys([BASELINE, *finalists]))
    for confirmation in ("confirm-1", "confirm-2"):
        rng.shuffle(finalists)
        for config in finalists:
            measure(config, confirmation)
    report["recommended"] = recommend(report["trials"])
    partial = args.out.with_suffix(args.out.suffix + ".tmp")
    partial.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    partial.replace(args.out)
    print(json.dumps(report["recommended"]), flush=True)
    return 0 if report["recommended"] else 1


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="ONNX with embedded weights, or OpenVINO IR XML and BIN")
    parser.add_argument("--alloc", type=allocation, action="append", help="SHAVES:CMX:STREAMS, repeatable")
    parser.add_argument("--out", type=Path, default=Path("results/ncs2-tuning.json"))
    parser.add_argument("--feeds", type=Path, help="JSON array of named model input mappings")
    parser.add_argument("--check", choices=("numeric", "top1"), default="numeric")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--in-flight", type=int, choices=(1, 2), default=2)
    parser.add_argument("--atol", type=float, default=.01)
    parser.add_argument("--rtol", type=float, default=.01)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--worker", type=allocation, help=SUPPRESS)
    args = parser.parse_args()
    if args.warmup < 1 or args.runs < 5 or not args.atol >= 0 or not args.rtol >= 0 or args.timeout <= 0:
        parser.error("Use positive warmup and timeout, at least five runs, and nonnegative tolerances")
    if args.worker:
        try:
            result = trial(args)
        except (RuntimeError, ArithmeticError, ValueError) as error:
            result = {"status": "error", "detail": str(error).strip()[-1500:]}
        print(MARKER + json.dumps(result), flush=True)
        return 0
    return sweep(args)


if __name__ == "__main__":
    sys.exit(main())
