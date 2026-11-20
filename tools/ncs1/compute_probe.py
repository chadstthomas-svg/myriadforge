"""Compute-ceiling probes v2: paired-gap method for exact compute isolation.

For each compute probe we ALSO bench a zero-compute gap model with the identical
input shape — subtracting them cancels dispatch and input transfer exactly, leaving
pure compute time -> true GFLOPS. (The plugin's own get_perf_counts() crashes, and
the naive 40 MB/s subtraction over-predicts transfer for large inputs.)

Probes:
  dense-conv  Conv(c_in,hw -> c_out) + GAP, paired with gap(c_in,hw)   : compute-dense
  fc-stack    1x1-Conv (== fully connected) stack + GAP, paired        : MobileNetV2-style 1x1 path
  conv1d      [1,8,L] audio-gate Conv1d stack + GAP, paired            : the 1-D fit-class
  fp16-input  gap fed float16 tensors                                  : plugin acceptance
  out-dir     tiny input -> wide Gemm output                           : output-direction rate

Writes tools/ncs1/compute_curve.json.
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
    """Zero-compute: input -> GlobalAveragePool. Same input bytes as its compute twin."""
    node = helper.make_node("GlobalAveragePool", ["input"], ["output"])
    graph = helper.make_graph(
        [node], "gap",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [shape[0], shape[1], 1, 1])],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    m.ir_version = 6
    return save(m, path)


def dense_conv_model(path: Path, c_in: int, hw: int, c_out: int):
    """Conv 3x3 c_in->c_out + GAP: the compute twin of gap(c_in, hw)."""
    w = RNG.standard_normal((c_out, c_in, 3, 3)).astype(np.float32) * 0.05
    b = RNG.standard_normal(c_out).astype(np.float32) * 0.01
    nodes = [
        helper.make_node("Conv", ["input", "w", "b"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("GlobalAveragePool", ["c"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes, "denseconv",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, c_in, hw, hw])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, c_out, 1, 1])],
        initializer=[numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    m.ir_version = 6
    return save(m, path)


def fc_stack_model(path: Path, d: int, depth: int):
    """1x1 Conv stack (== fully connected) + GAP: MobileNetV2's 1x1 path at scale."""
    inits, nodes = [], []
    prev = "input"
    for i in range(depth):
        w = RNG.standard_normal((d, d)).astype(np.float32) * 0.02
        inits.append(numpy_helper.from_array(w.reshape(d, d, 1, 1), f"w{i}"))
        nodes.append(helper.make_node("Conv", [prev, f"w{i}"], [f"c{i}"], kernel_shape=[1, 1]))
        nodes.append(helper.make_node("Relu", [f"c{i}"], [f"r{i}"]))
        prev = f"r{i}"
    w = RNG.standard_normal((10, d)).astype(np.float32) * 0.02
    inits.append(numpy_helper.from_array(w.reshape(10, d, 1, 1), "wout"))
    nodes.append(helper.make_node("Conv", [prev, "wout"], ["cout"], kernel_shape=[1, 1]))
    nodes.append(helper.make_node("GlobalAveragePool", ["cout"], ["output"]))
    graph = helper.make_graph(
        nodes, "fcstack",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, d, 1, 1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 1, 1])],
        initializer=inits,
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
    with tempfile.TemporaryDirectory(prefix="ceil2-") as tmp:
        tmp = Path(tmp)
        print("=== dense Conv2D (paired with same-input gap) ===", flush=True)
        for c_in, hw, c_out in ((32, 28, 64), (64, 56, 128), (128, 112, 128)):
            macs = c_out * c_in * 9 * hw * hw
            flops = 2 * macs
            ir_c = to_ir(dense_conv_model(tmp / f"dc_{c_in}_{hw}_{c_out}.onnx", c_in, hw, c_out), cache_dir=tmp)
            ir_g = to_ir(gap_model(tmp / f"gap_{c_in}_{hw}.onnx", [1, c_in, hw, hw]), cache_dir=tmp)
            rc, rg = p50min(ir_c), p50min(ir_g)
            compute = rc["min"] - rg["min"]
            gflops = flops / 1e9 / (compute / 1000) if compute > 0 else 0.0
            r = {"probe": f"dense-conv {c_in}x{hw}x{c_out}", "flops_m": round(flops / 1e6, 1),
                 "conv_p50": round(rc["p50"], 2), "gap_p50": round(rg["p50"], 2),
                 "compute_ms": round(compute, 2), "gflops": round(gflops, 1)}
            results.append(r)
            print(f"  {r['probe']:<26} conv {rc['p50']:7.2f} - gap {rg['p50']:6.2f} = compute {compute:7.2f} ms | {gflops:6.1f} GFLOPS", flush=True)

        print("=== FC stack (1x1 convs, MobileNetV2 1x1 path) ===", flush=True)
        for d, depth in ((1024, 8), (2048, 4)):
            flops = 2 * d * d * depth + d * 10 * 2
            ir_c = to_ir(fc_stack_model(tmp / f"fc_{d}_{depth}.onnx", d, depth), cache_dir=tmp)
            ir_g = to_ir(gap_model(tmp / f"gapfc_{d}.onnx", [1, d, 1, 1]), cache_dir=tmp)
            rc, rg = p50min(ir_c), p50min(ir_g)
            compute = rc["min"] - rg["min"]
            gflops = flops / 1e9 / (compute / 1000) if compute > 0 else 0.0
            r = {"probe": f"fc-stack d={d} x{depth}", "flops_m": round(flops / 1e6, 1),
                 "conv_p50": round(rc["p50"], 2), "gap_p50": round(rg["p50"], 2),
                 "compute_ms": round(compute, 2), "gflops": round(gflops, 1)}
            results.append(r)
            print(f"  {r['probe']:<26} conv {rc['p50']:7.2f} - gap {rg['p50']:6.2f} = compute {compute:7.2f} ms | {gflops:6.1f} GFLOPS", flush=True)

        print("=== 1D CNN (audio-gate fit-class) ===", flush=True)
        macs = 64 * 8 * 9 * 16000 + 64 * 64 * 9 * 16000
        flops = 2 * macs
        ir_c = to_ir(conv1d_model(tmp / "conv1d.onnx"), cache_dir=tmp)
        ir_g = to_ir(gap_model(tmp / "gap1d.onnx", [1, 8, 16000]), cache_dir=tmp)
        rc, rg = p50min(ir_c), p50min(ir_g)
        compute = rc["min"] - rg["min"]
        gflops = flops / 1e9 / (compute / 1000) if compute > 0 else 0.0
        r = {"probe": "conv1d 8->64x64 k9 L=16k", "flops_m": round(flops / 1e6, 1),
             "conv_p50": round(rc["p50"], 2), "gap_p50": round(rg["p50"], 2),
             "compute_ms": round(compute, 2), "gflops": round(gflops, 1)}
        results.append(r)
        print(f"  {r['probe']:<26} conv {rc['p50']:7.2f} - gap {rg['p50']:6.2f} = compute {compute:7.2f} ms | {gflops:6.1f} GFLOPS", flush=True)

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

    out = {"schema": "myriad2.compute-curve/1", "chip": "ma2450",
           "runtime": "OpenVINO 2020.3.2 LTS", "date": time.strftime("%Y-%m-%d"),
           "method": "paired-gap: compute = conv_p50.min - gap_p50.min (same input shape)",
           "probes": results}
    (HERE / "compute_curve.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote compute_curve.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
