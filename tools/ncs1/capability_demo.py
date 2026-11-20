"""Capability-shim demo: an op the Myriad 2 can't run falls back to CPU via HETERO.

Builds a tiny Conv -> HardSigmoid graph (Conv is Myriad-supported; HardSigmoid is
not — a MobileNetV3-style activation). Pure MYRIAD rejects it; HETERO:MYRIAD,CPU
runs it (Conv on the stick, HardSigmoid on the host CPU). Proves the capability shim widens the runnable model set.
"""
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
from openvino.inference_engine import IECore  # noqa: E402
from apps.common.ov import to_ir  # noqa: E402


def build_conv_hardsigmoid(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(7)
    w = rng.standard_normal((8, 1, 3, 3)).astype(np.float32) * 0.1
    b = rng.standard_normal(8).astype(np.float32) * 0.01
    nodes = [
        helper.make_node("Conv", ["input", "w", "b"], ["conv"],
                         kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        helper.make_node("HardSigmoid", ["conv"], ["logits"], alpha=0.2, beta=0.5),  # Myriad 2 lacks HardSigmoid
    ]
    graph = helper.make_graph(
        nodes, "conv_hardsigmoid",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 8, 8])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 8, 8, 8])],
        initializer=[numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 6
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def main() -> int:
    ie = IECore()
    with tempfile.TemporaryDirectory(prefix="cap-demo-") as tmp:
        onnx_path = Path(tmp) / "conv_hardsigmoid.onnx"
        build_conv_hardsigmoid(onnx_path)
        ir = to_ir(onnx_path, cache_dir=Path(tmp))
        net = ie.read_network(model=str(ir), weights=str(ir.with_suffix(".bin")))
        in_name = next(iter(net.inputs)); out_name = next(iter(net.outputs))
        inp = np.random.default_rng(42).standard_normal(net.inputs[in_name].shape).astype(np.float32)

        print("=== pure MYRIAD (expect rejection: HardSigmoid unsupported) ===", flush=True)
        try:
            ex = ie.load_network(network=net, device_name="MYRIAD", config={}, num_requests=1)
            ex.infer({in_name: inp})
            print("  MYRIAD ran it (HardSigmoid apparently supported)", flush=True)
        except Exception as e:
            print(f"  MYRIAD REJECTED: {type(e).__name__}: {str(e)[:160]}", flush=True)

        print("=== HETERO:MYRIAD,CPU (expect success: HardSigmoid -> CPU) ===", flush=True)
        try:
            t0 = time.perf_counter_ns()
            ex = ie.load_network(network=net, device_name="HETERO:MYRIAD,CPU", config={}, num_requests=1)
            load = (time.perf_counter_ns() - t0) / 1e6
            res = ex.infer({in_name: inp})
            out = np.asarray(res[out_name])
            print(f"  HETERO RAN IT: load {load:.0f}ms, out shape {out.shape}, "
                  f"sample {np.round(out.reshape(-1)[:3], 4).tolist()}", flush=True)
        except Exception as e:
            print(f"  HETERO FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
