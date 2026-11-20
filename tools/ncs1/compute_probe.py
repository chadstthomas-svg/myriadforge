"""Compute-ceiling probes: is 14.4 GFLOPS a Conv-kernel ceiling or a device-wide wall?

Uses the profile_split.py decomposition pattern: for each probe, wall time minus the
known fixed dispatch (~1.52 ms) and measured I/O transfer (~40 MB/s in, measured out)
yields the compute time -> effective GFLOPS.

Probes:
  dense-conv  Conv(C_in,H -> C_out) + GlobalAveragePool : compute-dense, tiny I/O
  mlp         Gemm stacks (d=1024/4096)                 : weight-memory-bound class
  conv1d      [1,8,L] audio-gate shape                  : the 1-D fit-class
  fp16-input  gap_224 fed float16 tensors               : does the plugin accept half bytes?
  out-dir     tiny input -> wide output                 : output-direction bandwidth

Output: prints the table and writes tools/ncs1/compute_curve.json.
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
FIXED_MS = 1.52          # measured dispatch cost (profile_split.py)
IN_MBPS = 40.7           # measured input-transfer rate (profile_split.py)

RNG = np.random.default_rng(0)


def save(model, path: Path) -> Path:
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return path


def conv2d_probe(path, c_in, hw, c_out):
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


def mlp_probe(path, d, depth):
    inits, nodes = [], []
    prev, prev_name = d, "input"
    for i in range(depth):
        w = RNG.standard_normal((d, d)).astype(np.float32) * 0.02
        inits.append(numpy_helper.from_array(w, f"w{i}"))
        nodes.append(helper.make_node("Gemm", [prev_name, f"w{i}"], [f"h{i}"], transB=1))
        nodes.append(helper.make_node("Relu", [f"h{i}"], [f"r{i}"]))
        prev, prev_name = d, f"r{i}"
    w = RNG.standard_normal((10, d)).astype(np.float32) * 0.02
    inits.append(numpy_helper.from_array(w, "wout"))
    nodes.append(helper.make_node("Gemm", [prev_name, "wout"], ["output"], transB=1))
    graph = helper.make_graph(
        nodes, "mlp",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, d])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])],
        initializer=inits,
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    m.ir_version = 6
    return save(m, path)


def conv1d_probe(path, length=16000):
    w1 = RNG.standard_normal((64, 8, 9)).astype(np.float32) * 0.05
    w2 = RNG.standard_normal((64, 64, 9)).astype(np.float32) * 0.05
    b1 = RNG.standard_normal(64).astype(np.float32) * 0.01
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
                     numpy_helper.from_array(w2, "w2")],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    m.ir_version = 6
    return save(m, path)


def outdir_probe(path, width):
    w = RNG.standard_normal((width, 8)).astype(np.float32) * 0.05
    nodes = [helper.make_node("Gemm", ["input", "w", "b"], ["output"], transB=1)]
    graph = helper.make_graph(
        nodes, "outdir",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, width])],
        initializer=[numpy_helper.from_array(w, "w"), numpy_helper.from_array(RNG.standard_normal(width).astype(np.float32) * 0.01, "b")],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    m.ir_version = 6
    return save(m, path)


def bench(ir: Path, runs: int = 25, in_bytes: int = 0, out_bytes: int = 0, flops: float = 0.0,
          fp16_input: bool = False):
    net = ie.read_network(model=str(ir), weights=str(ir.with_suffix(".bin")))
    in_name = next(iter(net.inputs))
    shape = list(net.inputs[in_name].shape)
    base = np.random.default_rng(0).uniform(0, 1, size=shape)
    inp16 = base.astype(np.float16) if fp16_input else base.astype(np.float32)
    t0 = time.perf_counter_ns()
    ex = ie.load_network(network=net, device_name="MYRIAD", config={}, num_requests=1)
    load_ms = (time.perf_counter_ns() - t0) / 1e6
    req = ex.requests[0]
    req.infer({in_name: inp16})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        req.infer({in_name: inp16})
        times.append((time.perf_counter_ns() - t0) / 1e6)
    s = sorted(times)
    p50, mn = s[len(s) // 2], s[0]
    in_ms = in_bytes / (IN_MBPS * 1e6) * 1000
    out_ms = out_bytes / (40e6) * 1000  # provisional until the out-dir probe refines it
    compute = mn - FIXED_MS - in_ms - out_ms
    gflops = flops / 1e9 / (compute / 1000) if compute > 0 else 0.0
    return {"load_ms": round(load_ms, 1), "p50_ms": round(p50, 2), "min_ms": round(mn, 2),
            "flops_m": round(flops / 1e6, 1), "in_bytes": in_bytes, "out_bytes": out_bytes,
            "compute_ms": round(compute, 2), "gflops_eff": round(gflops, 1)}


def main() -> int:
    probes = []
    with tempfile.TemporaryDirectory(prefix="ceil-") as tmp:
        tmp = Path(tmp)
        print("=== dense Conv2D (compute-dense, tiny I/O) ===", flush=True)
        for c_in, hw, c_out in ((32, 28, 64), (64, 56, 128), (128, 112, 128)):
            macs = c_out * c_in * 9 * hw * hw
            flops = 2 * macs
            p = conv2d_probe(tmp / f"dconv_{c_in}_{hw}_{c_out}.onnx", c_in, hw, c_out)
            ir = to_ir(p, cache_dir=tmp)
            r = bench(ir, runs=25, in_bytes=c_in * hw * hw * 4, out_bytes=c_out * 4, flops=flops)
            r["probe"] = f"dense-conv {c_in}x{hw}x{c_out}"
            probes.append(r)
            print(f"  {r['probe']:<26} {r['flops_m']:8.1f} MFLOP | p50 {r['p50_ms']:7.2f} ms | compute {r['compute_ms']:7.2f} ms | {r['gflops_eff']:6.1f} GFLOPS", flush=True)

        print("=== MLP (weight-memory-bound class) ===", flush=True)
        for d, depth in ((1024, 4), (4096, 2)):
            macs = depth * d * d + d * 10
            flops = 2 * macs
            wbytes = depth * d * d * 2 + d * 10 * 2
            p = mlp_probe(tmp / f"mlp_{d}_{depth}.onnx", d, depth)
            ir = to_ir(p, cache_dir=tmp)
            r = bench(ir, runs=25, in_bytes=d * 4, out_bytes=40, flops=flops)
            r["probe"] = f"mlp d={d} x{depth}"
            r["weight_mb"] = round(wbytes / 1024**2, 1)
            probes.append(r)
            print(f"  {r['probe']:<26} {r['flops_m']:8.1f} MFLOP | weights {r['weight_mb']:6.1f} MB | p50 {r['p50_ms']:7.2f} ms | compute {r['compute_ms']:7.2f} ms | {r['gflops_eff']:6.1f} GFLOPS", flush=True)

        print("=== 1D CNN (audio-gate fit-class) ===", flush=True)
        macs = 64 * 8 * 9 * 16000 + 64 * 64 * 9 * 16000
        flops = 2 * macs
        p = conv1d_probe(tmp / "conv1d.onnx")
        ir = to_ir(p, cache_dir=tmp)
        r = bench(ir, runs=25, in_bytes=8 * 16000 * 4, out_bytes=64 * 4, flops=flops)
        r["probe"] = "conv1d 8->64x64 k9 L=16k"
        probes.append(r)
        print(f"  {r['probe']:<26} {r['flops_m']:8.1f} MFLOP | p50 {r['p50_ms']:7.2f} ms | compute {r['compute_ms']:7.2f} ms | {r['gflops_eff']:6.1f} GFLOPS", flush=True)

        print("=== FP16 input acceptance (gap_224, halved bytes) ===", flush=True)
        p = conv2d_probe(tmp / "gap224fp16.onnx", 3, 224, 3)
        ir = to_ir(p, cache_dir=tmp)
        try:
            r = bench(ir, runs=15, in_bytes=3 * 224 * 224 * 2, fp16_input=True)
            print(f"  FP16 input ACCEPTED | p50 {r['p50_ms']:.2f} ms", flush=True)
            probes.append({"probe": "fp16-input", "status": "accepted", "p50_ms": r["p50_ms"]})
        except Exception as exc:  # noqa: BLE001
            print(f"  FP16 input REJECTED: {str(exc)[:120]}", flush=True)
            probes.append({"probe": "fp16-input", "status": "rejected", "error": str(exc)[:120]})

        print("=== output-direction bandwidth ===", flush=True)
        for width in (20000, 40000):
            macs = width * 8
            p = outdir_probe(tmp / f"out_{width}.onnx", width)
            ir = to_ir(p, cache_dir=tmp)
            r = bench(ir, runs=25, in_bytes=32, out_bytes=width * 4, flops=2 * macs)
            out_ms = r["min_ms"] - FIXED_MS
            bw = width * 4 / (out_ms / 1000) / 1e6
            print(f"  out {width:>6} floats ({width*4/1024:6.0f} KB): wall min {r['min_ms']:6.2f} ms -> {bw:6.1f} MB/s", flush=True)
            probes.append({"probe": f"out-dir {width}", "out_bytes": width * 4, "min_ms": r["min_ms"], "mbps": round(bw, 1)})

    out = {
        "schema": "myriad2.compute-curve/1",
        "chip": "ma2450", "runtime": "OpenVINO 2020.3.2 LTS",
        "fixed_ms": FIXED_MS, "in_mbps": IN_MBPS,
        "date": time.strftime("%Y-%m-%d"),
        "probes": probes,
    }
    (HERE / "compute_curve.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote compute_curve.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
