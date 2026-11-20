"""NCS2 bridge runner - executes INSIDE the isolated OpenVINO 2022.3.1 venv.

The myriadforge main env never imports OpenVINO; this script is invoked by
devices/ncs2.py via the venv's python. It converts ONNX->IR when needed
(OpenVINO 2022.3 API), runs the latency loop with per-iteration timing,
and writes a myriadforge.result/1 JSON. myriadforge.bench.schema is stdlib-only,
so it is importable here via sys.path without the main env's deps.

Usage (from the ncs2 venv):
  python bridge_runner.py --model m.onnx --device myriad --shaves 16 \
      --warmup 5 --runs 50 --out results/
  device: myriad | cpu (cpu = OpenVINO CPU plugin, functional truth)
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

# myriadforge source on sys.path so the stdlib-only schema module is importable
# inside this venv without installing myriadforge (tools/ncs2 -> tools -> repo root).
_REPO_SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(_REPO_SRC))

from myriadforge.bench.metrics import percentile  # noqa: E402 (path set above)
from myriadforge.bench.schema import make_result, save_result  # noqa: E402


def _timed(infer_fn, warmup_runs: int, measured_runs: int) -> dict[str, float]:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", choices=["myriad", "cpu"], default="myriad")
    ap.add_argument("--shaves", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--out", default="results")
    ap.add_argument("--input-json", help="deterministic input values as JSON array")
    args = ap.parse_args()

    import numpy as np  # available in the ncs2 venv (openvino dependency)
    from openvino.runtime import Core

    model_path = Path(args.model)
    core = Core()
    available = list(core.available_devices)
    ov_device = "MYRIAD" if args.device == "myriad" else "CPU"
    if args.device == "myriad" and not any(d.startswith("MYRIAD") for d in available):
        print(json.dumps({"error": f"no MYRIAD device; available: {available}"}))
        return 2

    # ONNX needs Model Optimizer conversion (2022.3 API); .xml is already IR.
    ir_path = model_path
    tmp = tempfile.TemporaryDirectory(prefix="ncs2-bridge-")
    if model_path.suffix.lower() == ".onnx":
        from openvino.tools.mo import convert_model
        ov_model = convert_model(str(model_path))
        t0 = time.perf_counter_ns()
        ir_path = Path(tmp.name) / (model_path.stem + ".xml")
        from openvino.runtime import serialize
        serialize(ov_model, str(ir_path))
        load_ms = (time.perf_counter_ns() - t0) / 1e6
    else:
        load_ms = None

    props = {"MYRIAD_NUMBER_OF_SHAVES": args.shaves} if args.device == "myriad" else {}
    t0 = time.perf_counter_ns()
    compiled = core.compile_model(str(ir_path), ov_device, props)
    init_ms = load_ms if load_ms is not None else (time.perf_counter_ns() - t0) / 1e6

    sig = next(iter(compiled.inputs))
    shape = sig.get_partial_shape()

    def _dim_len(d) -> int:
        try:
            return int(d.get_length())  # OpenVINO 2022.3 Dimension API
        except AttributeError:
            return int(d)  # newer API

    dims = [_dim_len(d) if d.is_static else 1 for d in shape]
    if not dims:
        dims = [1, 224, 224, 3]
    rng = np.random.default_rng(42)
    inp = rng.uniform(0.0, 1.0, size=dims).astype(np.float32)
    if args.input_json:
        vals = np.array(json.loads(args.input_json), dtype=np.float32)
        inp = vals.reshape(dims)

    infer_req = compiled.create_infer_request()
    infer_fn = lambda: infer_req.infer({sig: inp})  # noqa: E731

    t0 = time.perf_counter_ns()
    infer_fn()
    first_ms = (time.perf_counter_ns() - t0) / 1e6
    ss = _timed(infer_fn, max(args.warmup - 1, 0), args.runs)

    record = make_result(
        model={
            "name": model_path.stem,
            "file": model_path.name,
            "path": str(model_path),
            "format": model_path.suffix.lstrip("."),
            "size_mb": round(model_path.stat().st_size / 1024**2, 2),
        },
        device={
            "id": f"ncs2-{args.device}" if args.device == "myriad" else "openvino-cpu",
            "lane": "ncs2",
            "kind": "usb-accel" if args.device == "myriad" else "host",
            "label": f"Intel NCS2 ({ov_device})",
            "openvino_devices": available,
        },
        backend={
            "name": "openvino-myriad" if args.device == "myriad" else "openvino-cpu",
            "runtime": "OpenVINO 2022.3.1 LTS",
            "accelerator": "myriad" if args.device == "myriad" else "cpu",
            "shave_count": args.shaves if args.device == "myriad" else None,
            "usb_transport": args.device == "myriad",
            "simulated": False,
        },
        config={"warmup_runs": args.warmup, "measured_runs": args.runs,
                "shaves": args.shaves if args.device == "myriad" else None},
        metrics={
            "init_ms": round(init_ms, 3),
            "first_inference_ms": round(first_ms, 3),
            "warmup_runs": args.warmup,
            "steady_state_ms": {k: round(v, 3) for k, v in ss.items()},
            "throughput_fps": round(1000.0 / ss["p50"], 2),
        },
    )
    out = save_result(record, Path(args.out))
    print(json.dumps({"ok": True, "result": str(out)}))
    tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
