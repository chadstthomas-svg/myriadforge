"""NCS1 bridge runner - executes INSIDE the isolated OpenVINO 2020.3.2 venv (Py3.7).

The myriadforge main env never imports OpenVINO; devices/ncs1.py invokes this
script with the ncs1 venv python. It converts ONNX->FP16 IR via the 2020.3
Model Optimizer (through apps.common.ov.to_ir), runs the latency loop on
MYRIAD (Myriad 2, ma2450) or the CPU plugin, and writes myriadforge.result/1.

2020.3 is the last OpenVINO whose MYRIAD boot table accepts the Myriad 2 PID
0x2150. The Myriad 2 plugin has no SHAVE-count config key (fixed 12 SHAVEs),
so unlike the NCS2 bridge there is no --shaves sweep here.

Usage (from the ncs1 venv):
  python bridge_runner.py --model m.onnx --device myriad --warmup 5 --runs 50 --out results/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))   # stdlib-only myriadforge.bench.*
sys.path.insert(0, str(_REPO_ROOT))           # apps.common.ov (conversion)
sys.path.insert(0, str(_HERE))                # ncs1_paths

import ncs1_paths  # noqa: E402

ncs1_paths.setup()

from myriadforge.bench.metrics import percentile  # noqa: E402
from myriadforge.bench.schema import make_result, save_result  # noqa: E402


def _timed(infer_fn, warmup_runs: int, measured_runs: int) -> dict:
    for _ in range(warmup_runs):
        infer_fn()
    samples: list[float] = []
    for _ in range(measured_runs):
        t0 = time.perf_counter_ns()
        infer_fn()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    samples.sort()
    mean = sum(samples) / len(samples)
    var = sum((s - mean) ** 2 for s in samples) / (len(samples) - 1)
    return {
        "mean": mean, "p50": percentile(samples, 50), "p90": percentile(samples, 90),
        "p99": percentile(samples, 99), "stddev": var ** 0.5,
        "min": samples[0], "max": samples[-1], "runs": float(len(samples)),
    }


def _stats(samples: list[float]) -> dict:
    samples = sorted(samples)
    mean = sum(samples) / len(samples)
    var = sum((s - mean) ** 2 for s in samples) / max(len(samples) - 1, 1)
    return {
        "mean": mean, "p50": percentile(samples, 50), "p90": percentile(samples, 90),
        "p99": percentile(samples, 99), "stddev": var ** 0.5,
        "min": samples[0], "max": samples[-1], "runs": float(len(samples)),
    }


def _timed_pipelined(reqs, inputs: dict, warmup_runs: int, measured_runs: int) -> dict:
    """Async-pipelined throughput: keep N requests in flight, round-robin wait.

    Overlaps USB transfer of request k+1 with on-stick compute of request k.
    Metrics are inter-completion cadence (effective ms/infer), NOT single-shot
    latency — p50 ~ 1000/throughput. Measured sweet spot on ma2450 is N=2.
    """
    n = len(reqs)
    for r in reqs:
        r.async_infer(inputs)
    for r in reqs:
        r.wait()
    for _ in range(max(warmup_runs, 1)):
        for r in reqs:
            r.async_infer(inputs)
        for r in reqs:
            r.wait()
    for r in reqs:
        r.async_infer(inputs)
    samples: list[float] = []
    last = time.perf_counter_ns()
    processed = 0
    k = 0
    while processed < measured_runs:
        r = reqs[k % n]
        r.wait()
        now = time.perf_counter_ns()
        samples.append((now - last) / 1e6)
        last = now
        processed += 1
        r.async_infer(inputs)
        k += 1
    for r in reqs:
        r.wait()
    return _stats(samples)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", choices=["myriad", "cpu", "hetero"], default="myriad",
                    help="myriad=stick, cpu=OpenVINO CPU, hetero=HETERO:MYRIAD,CPU "
                         "(unsupported ops fall back to CPU)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--requests", type=int, default=2,
                    help="in-flight async infer requests: 1=sync latency, >1=pipelined "
                         "throughput (ma2450 sweet spot=2, ~+37%% vs sync)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--input-json", help="deterministic input values as JSON array")
    args = ap.parse_args()

    import numpy as np
    from openvino.inference_engine import IECore
    from apps.common.ov import to_ir

    model_path = Path(args.model)
    ie = IECore()
    available = list(ie.available_devices)
    ov_device = {"myriad": "MYRIAD", "cpu": "CPU", "hetero": "HETERO:MYRIAD,CPU"}[args.device]
    if args.device in ("myriad", "hetero") and not any(d.startswith("MYRIAD") for d in available):
        print(json.dumps({"error": f"no MYRIAD device; available: {available}"}))
        return 2

    t0 = time.perf_counter_ns()
    ir = to_ir(model_path)  # ONNX -> FP16 IR (cached), or .xml passthrough
    convert_ms = (time.perf_counter_ns() - t0) / 1e6

    net = ie.read_network(model=str(ir), weights=str(ir.with_suffix(".bin")))
    n_reqs = 1 if args.device == "hetero" else max(1, args.requests)
    t0 = time.perf_counter_ns()
    exec_net = ie.load_network(network=net, device_name=ov_device, config={},
                               num_requests=n_reqs)
    load_ms = (time.perf_counter_ns() - t0) / 1e6

    in_name = next(iter(net.inputs))
    out_name = next(iter(net.outputs))
    shape = list(net.inputs[in_name].shape)
    rng = np.random.default_rng(42)
    inp = rng.uniform(0.0, 1.0, size=shape).astype(np.float32)
    if args.input_json:
        inp = np.array(json.loads(args.input_json), dtype=np.float32).reshape(shape)
    inputs = {in_name: inp}

    t0 = time.perf_counter_ns()
    exec_net.requests[0].infer(inputs)  # first contact (single-shot)
    first_ms = (time.perf_counter_ns() - t0) / 1e6

    if n_reqs > 1:
        ss = _timed_pipelined(exec_net.requests, inputs, max(args.warmup - 1, 0), args.runs)
        mode = "pipelined-throughput"
    else:
        ss = _timed(lambda: exec_net.infer(inputs), max(args.warmup - 1, 0), args.runs)
        mode = "sync-latency"

    record = make_result(
        model={
            "name": model_path.stem, "file": model_path.name, "path": str(model_path),
            "format": model_path.suffix.lstrip("."),
            "size_mb": round(model_path.stat().st_size / 1024**2, 2),
        },
        device={
            "id": {"myriad": "ncs1-myriad", "cpu": "ncs1-ovcpu", "hetero": "ncs1-hetero"}[args.device],
            "lane": "ncs1",
            "kind": "host" if args.device == "cpu" else "usb-accel",
            "label": f"Intel NCS1 ({ov_device}, Myriad 2 ma2450)",
            "openvino_devices": available,
            "myriad_platform": "ma2450",
        },
        backend={
            "name": {"myriad": "openvino-myriad", "cpu": "openvino-cpu",
                     "hetero": "openvino-hetero"}[args.device],
            "runtime": "OpenVINO 2020.3.2 LTS",
            "accelerator": args.device,
            "shave_count": 12 if args.device in ("myriad", "hetero") else None,  # Myriad 2: fixed
            "shave_configurable": False,
            "usb_transport": args.device in ("myriad", "hetero"),
            "simulated": False,
            "pipelined": n_reqs > 1,
            "pipeline_requests": n_reqs,
        },
        config={"warmup_runs": args.warmup, "measured_runs": args.runs,
                "num_requests": n_reqs, "measurement_mode": mode},
        metrics={
            "convert_ms": round(convert_ms, 3),
            "init_ms": round(load_ms, 3),
            "first_inference_ms": round(first_ms, 3),
            "warmup_runs": args.warmup,
            "steady_state_ms": {k: round(v, 3) for k, v in ss.items()},
            "throughput_fps": round(1000.0 / ss["p50"], 2),
        },
    )
    out = save_result(record, Path(args.out))
    # Marker prefix: IECore emits XLink/BSL shutdown noise to stdout after this.
    print("NCS1RESULT:" + json.dumps({"ok": True, "result": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
