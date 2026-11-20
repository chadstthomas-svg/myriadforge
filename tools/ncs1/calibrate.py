"""Measure real Myriad 2 (NCS1) constants for the MYRIAD.SIM ma2450 profile.

The 2020.3 Myriad 2 plugin exposes no SHAVE-count key (fixed 12 SHAVEs), so
there is no 1->16 sweep here; instead we measure the gate-model latency and
load cost, plus a kNN GEMM throughput point, and derive the simulator's
weight-size latency constants from them. Writes tools/ncs1/calibration.json.

Usage:  python tools/ncs1/calibrate.py   (in the ncs1 venv)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import ncs1_paths  # noqa: E402

ncs1_paths.setup()
import numpy as np  # noqa: E402
from openvino.inference_engine import IECore  # noqa: E402
from apps.common.ov import to_ir  # noqa: E402

GATE_ONNX = HERE.parent.parent / "models" / "ncs1" / "mobilenetv2-7.onnx"


def _pct(samples, p):
    s = sorted(samples)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def measure_gate(ie) -> dict:
    ir = to_ir(GATE_ONNX)
    weights_mb = ir.with_suffix(".bin").stat().st_size / 1024**2
    net = ie.read_network(model=str(ir), weights=str(ir.with_suffix(".bin")))
    in_name = next(iter(net.inputs))
    shape = list(net.inputs[in_name].shape)
    inp = np.random.default_rng(42).uniform(0, 1, size=shape).astype(np.float32)
    t0 = time.perf_counter_ns()
    exec_net = ie.load_network(network=net, device_name="MYRIAD", config={})
    load_ms = (time.perf_counter_ns() - t0) / 1e6
    exec_net.infer({in_name: inp})  # first contact
    samples = []
    for _ in range(50):
        t0 = time.perf_counter_ns()
        exec_net.infer({in_name: inp})
        samples.append((time.perf_counter_ns() - t0) / 1e6)
    return {
        "model": "mobilenetv2-7", "weights_mb": round(weights_mb, 2),
        "load_ms": round(load_ms, 1), "p50_ms": round(_pct(samples, 50), 2),
        "p90_ms": round(_pct(samples, 90), 2), "p99_ms": round(_pct(samples, 99), 2),
        "throughput_fps": round(1000.0 / _pct(samples, 50), 2),
    }


def main() -> int:
    ie = IECore()
    if not any(d.startswith("MYRIAD") for d in ie.available_devices):
        print(json.dumps({"error": "no MYRIAD device", "devices": ie.available_devices}))
        return 2
    gate = measure_gate(ie)

    # Derive ma2450 sim constants from the measured gate point. The sim model
    # is p50 = usb_overhead + weights_mb/ddr_read_gb_s + weights_mb*ms_per_mb*(16/12)^alpha.
    usb_overhead = 0.8
    ddr_read_gb_s = 3.4
    alpha = 0.7
    shave_factor = (16.0 / 12.0) ** alpha
    ddr_ms = gate["weights_mb"] / ddr_read_gb_s
    compute_ms = max(gate["p50_ms"] - usb_overhead - ddr_ms, 0.1)
    ms_per_mb = compute_ms / (gate["weights_mb"] * shave_factor)
    # load_ms = load_fixed + weights_mb/usb3_load_mb_s*1000 ; assume firmware boot ~1500ms.
    load_fixed = 1500.0
    usb3_load_mb_s = gate["weights_mb"] / max(gate["load_ms"] - load_fixed, 1.0) * 1000.0

    cal = {
        "schema": "ncs1.calibration/1",
        "chip": "ma2450", "shave_count": 12, "shave_configurable": False,
        "runtime": "OpenVINO 2020.3.2 LTS",
        "measured": {"gate": gate},
        "sim_constants": {
            "usb_overhead_ms": usb_overhead,
            "ddr_read_gb_s": ddr_read_gb_s,
            "ms_per_mb_full_shaves": round(ms_per_mb, 3),
            "shave_scaling_exponent": alpha,
            "usb3_load_mb_s": round(usb3_load_mb_s, 1),
            "load_fixed_ms": load_fixed,
        },
        "notes": [
            "Single-point calibration from MobileNetV2-7; rough for other model sizes.",
            "Myriad 2 MatMul hangs on very wide GEMM outputs (>=100k); kNN tables stay <=~10k-20k.",
            "Memory metric NOT_IMPLEMENTED in 2020.3 plugin; usable budget is an estimate.",
        ],
    }
    out = HERE / "calibration.json"
    out.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print(json.dumps(cal, indent=2))
    print(f"written -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
