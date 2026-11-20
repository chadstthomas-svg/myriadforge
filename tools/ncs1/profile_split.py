"""Decompose the Myriad 2 per-inference cost: fixed overhead vs USB input vs compute.

Three models isolate the parts:
  relu_tiny   [1,8]          -> Relu : negligible transfer + compute => fixed USB/firmware overhead
  relu_224    [1,3,224,224]  -> Relu : same input bytes as MobileNetV2, ~0 compute => + input transfer
  mobilenetv2-7 [1,3,224,224]       : real 600 MFLOP model => + compute

Plus PERF_COUNT per-layer timings on MobileNetV2, compared against wall-clock, to see how
much the plugin itself attributes on-device vs leaves to host/USB.

Temporary diagnostic (not part of the app surface).
"""
import faulthandler
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
from onnx import TensorProto, helper  # noqa: E402
from openvino.inference_engine import IECore  # noqa: E402
from apps.common.ov import to_ir  # noqa: E402

faulthandler.enable()
ie = IECore()


def p50(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def bench_ir(ir: Path, runs: int = 30, perf: bool = False):
    net = ie.read_network(model=str(ir), weights=str(ir.with_suffix(".bin")))
    in_name = next(iter(net.inputs))
    shape = list(net.inputs[in_name].shape)
    inp = np.random.default_rng(0).uniform(0, 1, size=shape).astype(np.float32)
    cfg = {"PERF_COUNT": "YES"} if perf else {}
    t0 = time.perf_counter_ns()
    ex = ie.load_network(network=net, device_name="MYRIAD", config=cfg, num_requests=1)
    load_ms = (time.perf_counter_ns() - t0) / 1e6
    req = ex.requests[0]
    req.infer({in_name: inp})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        req.infer({in_name: inp})
        times.append((time.perf_counter_ns() - t0) / 1e6)
    return {
        "shape": shape, "load_ms": load_ms, "p50_ms": p50(times),
        "min_ms": min(times), "req": req, "in_name": in_name,
    }


def make_relu(path: Path, shape):
    node = helper.make_node("Relu", ["input"], ["output"])
    graph = helper.make_graph(
        [node], "relu",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    m.ir_version = 6
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def make_gap(path: Path, hw: int):
    """input [1,3,hw,hw] -> GlobalAveragePool -> [1,3,1,1]: full input transfer, ~0 output, ~0 compute."""
    node = helper.make_node("GlobalAveragePool", ["input"], ["output"])
    graph = helper.make_graph(
        [node], "gap",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, hw, hw])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 1, 1])],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    m.ir_version = 6
    onnx.checker.check_model(m)
    onnx.save(m, str(path))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="split-") as tmp:
        tmp = Path(tmp)
        print("=== building null models ===", flush=True)
        make_relu(tmp / "relu_tiny.onnx", [1, 8])
        make_gap(tmp / "gap_112.onnx", 112)
        make_gap(tmp / "gap_224.onnx", 224)
        tiny = to_ir(tmp / "relu_tiny.onnx", cache_dir=tmp)
        g112 = to_ir(tmp / "gap_112.onnx", cache_dir=tmp)
        g224 = to_ir(tmp / "gap_224.onnx", cache_dir=tmp)
        mobile = Path(r"D:\NTVStudy\models\ncs1\ir\mobilenetv2-7.xml")

        print("\n=== relu_tiny [1,8] (fixed overhead) ===", flush=True)
        rt = bench_ir(tiny, runs=30)
        print(f"  load {rt['load_ms']:.0f} ms | p50 {rt['p50_ms']:.2f} ms | min {rt['min_ms']:.2f}", flush=True)

        print("\n=== gap_112 [1,3,112,112] (input 150 KB) ===", flush=True)
        r1 = bench_ir(g112, runs=30)
        print(f"  load {r1['load_ms']:.0f} ms | p50 {r1['p50_ms']:.2f} ms | min {r1['min_ms']:.2f}", flush=True)

        print("\n=== gap_224 [1,3,224,224] (input 602 KB) ===", flush=True)
        r2 = bench_ir(g224, runs=30)
        print(f"  load {r2['load_ms']:.0f} ms | p50 {r2['p50_ms']:.2f} ms | min {r2['min_ms']:.2f}", flush=True)

        print("\n=== mobilenetv2-7 [1,3,224,224] (+ compute) ===", flush=True)
        mv = bench_ir(mobile, runs=40)
        print(f"  load {mv['load_ms']:.0f} ms | p50 {mv['p50_ms']:.2f} ms | min {mv['min_ms']:.2f}", flush=True)

        # decomposition on min latency (least noisy)
        fixed = rt["min_ms"]
        t112 = r1["min_ms"] - fixed
        t224 = r2["min_ms"] - fixed
        compute = mv["min_ms"] - r2["min_ms"]
        in224_bytes = 3 * 224 * 224 * 4
        in112_bytes = 3 * 112 * 112 * 4
        print("\n=== decomposition (min latency) ===", flush=True)
        print(f"  fixed USB+firmware round trip : {fixed:6.2f} ms  ({fixed/mv['min_ms']*100:4.1f}%)", flush=True)
        print(f"  input transfer 150 KB (112px) : {t112:6.2f} ms  => {in112_bytes/t112/1e6*1000:5.1f} MB/s", flush=True)
        print(f"  input transfer 602 KB (224px) : {t224:6.2f} ms  => {in224_bytes/t224/1e6*1000:5.1f} MB/s", flush=True)
        print(f"  compute+residual (600 MFLOP)  : {compute:6.2f} ms  ({compute/mv['min_ms']*100:4.1f}%)", flush=True)
        print(f"  => effective compute GFLOPS   : {600e6/(compute/1000)/1e9:5.1f}", flush=True)
        print(f"  MobileNetV2 total (min)       : {mv['min_ms']:6.2f} ms", flush=True)
        print("\n  NOTE: get_perf_counts() access-violates on this plugin build (unusable);", flush=True)
        print("        the decomposition above is the black-box substitute.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())