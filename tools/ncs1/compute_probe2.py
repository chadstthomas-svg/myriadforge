"""Remaining compute probes after a wedge recovery: conv1d, fp16-input, out-dir.

Run AFTER tools/ncs1/recover_f63b.ps1 / replug. Keep total MYRIAD loads under ~7
(the observed wedge threshold per power cycle).
"""
from __future__ import annotations

import faulthandler
import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import ncs1_paths  # noqa: E402
ncs1_paths.setup()
import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402
from openvino.inference_engine import IECore  # noqa: E402
from apps.common.ov import to_ir  # noqa: E402

faulthandler.enable()
ie = IECore()
FIXED_MS = 1.52
RNG = np.random.default_rng(0)


def save(model, path: Path) -> Path:
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def gap_model(path: Path, shape: list[int]):
    node = helper.make_node("GlobalAveragePool", ["input"], ["output"])
    graph = helper.make_graph(
        [node], "gap",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [shape[0], shape[1], 1, 1])],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    m.ir_version = 6
    return save(m, path)


def conv1d_model(path: Path, length: int = 16000):
    w1 = RNG.standard_normal((64, 8, 9)).astype(np.float32) * 0.05
    w2 = RNG.standard_normal((64, 64, 9)).astype(np.float32) * 0.05
    b1 = RNG.standard_normal(64).astype(np.float32) * 0.01
    b2 = RNG.standard_normal(64).astype(np.float32) * 0.01
    nodes = [
        helper.make_node("Conv", ["input", "w1", "b1"], ["c1"], kernel_shape=[9], pads=[4, 4]),
        helper.make_node("Relu", ["c1"], ["r1"]),
        helper.make_node("Conv", ["r1", "w2", "b2"], ["c2"], kernel_shape=[9], pads=[4, 4]),
        helper.make_node("GlobalAveragePool", ["c2"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes, "conv1d",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, length])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64, 1, 1])],
        initializer=[numpy_helper.from_array(w1, "w1"), numpy_helper.from_array(b1, "b1"),
                     numpy_helper.from_array(w2, "w2"), numpy_helper.from_array(b2, "b2")],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    m.ir_version = 6
    return save(m, path)


def outdir_model(path: Path, width: int):
    w = RNG.standard_normal((width, 8)).astype(np.float32) * 0.05
    b = RNG.standard_normal(width).astype(np.float32) * 0.01
    nodes = [helper.make_node("Gemm", ["input", "w", "b"], ["output"], transB=1)]
    graph = helper.make_graph(
        nodes, "outdir",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, width])],
        initializer=[numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    m.ir_version = 6
    return save(m, path)


def p50min(ir: Path, runs: int = 25, fp16: bool = False):
    net = ie.read_network(model=str(ir), weights=str(ir.with_suffix(".bin")))
    in_name = next(iter(net.inputs))
    shape = list(net.inputs[in_name].shape)
    base = np.random.default_rng(0).uniform(0, 1, size=shape)
    inp = base.astype(np.float16) if fp16 else base.astype(np.float32)
    ex = ie.load_network(network=net, device_name="MYRIAD", config={}, num_requests=1)
    req = ex.requests[0]
    req.infer({in_name: inp})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        req.infer({in_name: inp})
        times.append((time.perf_counter_ns() - t0) / 1e6)
    s = sorted(times)
    return {"p50": s[len(s) // 2], "min": s[0]}


def main() -> int:
    results = []
    with tempfile.TemporaryDirectory(prefix="ceil3-") as tmp:
        tmp = Path(tmp)
        print("=== 1D CNN: SKIPPED - Conv1d graphs wedge Myriad 2 at load (2/2, deterministic) ===", flush=True)
        results.append({"probe": "conv1d", "status": "WEDGE",
                        "note": "3D-input Conv wedges the firmware at boot; reshape to 2D Conv (kernel [1,9]) is the required transform"})
        print("  recorded as WEDGE finding", flush=True)

        print("=== FP16 input acceptance (gap_224) ===", flush=True)
        ir_g224 = to_ir(gap_model(tmp / "gap224.onnx", [1, 3, 224, 224]), cache_dir=tmp)
        try:
            r16 = p50min(ir_g224, runs=10, fp16=True)
            print(f"  FP16 input ACCEPTED | p50 {r16['p50']:.2f} ms", flush=True)
            results.append({"probe": "fp16-input", "status": "accepted", "p50_ms": round(r16["p50"], 2)})
        except Exception as exc:  # noqa: BLE001
            print(f"  FP16 input REJECTED: {str(exc)[:120]}", flush=True)
            results.append({"probe": "fp16-input", "status": "rejected", "error": str(exc)[:120]})

        print("=== output-direction bandwidth ===", flush=True)
        for width in (20000, 40000):
            ir_o = to_ir(outdir_model(tmp / f"out_{width}.onnx", width), cache_dir=tmp)
            r = p50min(ir_o, runs=25)
            out_ms = r["min"] - FIXED_MS
            bw = width * 4 / (out_ms / 1000) / 1e6 if out_ms > 0 else 0.0
            print(f"  out {width:>6} floats ({width * 4 / 1024:5.0f} KB): min {r['min']:6.2f} ms -> {bw:6.1f} MB/s", flush=True)
            results.append({"probe": f"out-dir {width}", "out_bytes": width * 4, "min_ms": r["min"], "mbps": round(bw, 1)})

    # merge into compute_curve.json (preserving the dense/fc probes from the prior run)
    curve_path = HERE / "compute_curve.json"
    try:
        curve = json.loads(curve_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        curve = {"schema": "myriad2.compute-curve/1", "probes": []}
    existing = {p.get("probe"): p for p in curve.get("probes", [])}
    for r in results:
        existing[r.get("probe", json.dumps(r)[:40])] = r
    curve["probes"] = list(existing.values())
    curve["date"] = time.strftime("%Y-%m-%d")
    curve_path.write_text(json.dumps(curve, indent=2), encoding="utf-8")
    print("\nmerged -> compute_curve.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
