"""NCS1 WS0 CPU smoke test - runs INSIDE the OpenVINO 2020.3.2 venv (Py3.7).

Proves the toolchain end to end without the dongle: build a small ONNX model,
convert to FP16 IR via the 2020.3 Model Optimizer, run on the CPU plugin
(IECore), and compare logits to an onnxruntime reference (rtol 1e-2 - FP16).
MYRIAD absence is logged, not fatal. opset 11 / IR 6 keeps the graph readable
by onnx 1.7 (the cp37 venv) and MO 2020.3 alike.

Usage:  python tools/ncs1/smoke_cpu.py   (exit 0 = pass)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import ncs1_paths  # noqa: E402

ncs1_paths.setup()
import numpy as np  # noqa: E402


def _build_onnx(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(7)
    w = rng.standard_normal((8, 1, 3, 3)).astype(np.float32) * 0.05
    b = rng.standard_normal(8).astype(np.float32) * 0.01
    fc_w = rng.standard_normal((4, 8)).astype(np.float32) * 0.05
    fc_b = rng.standard_normal(4).astype(np.float32) * 0.01
    nodes = [
        helper.make_node("Conv", ["input", "conv_w", "conv_b"], ["conv"],
                         kernel_shape=[3, 3], strides=[1, 1], pads=[1, 1, 1, 1]),
        helper.make_node("Relu", ["conv"], ["relu"]),
        helper.make_node("GlobalAveragePool", ["relu"], ["gap"]),
        helper.make_node("Reshape", ["gap", "shape"], ["flat"]),
        helper.make_node("Gemm", ["flat", "fc_w", "fc_b"], ["logits"], transB=1),
    ]
    graph = helper.make_graph(
        nodes, "ncs1_smoke",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 8, 8])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4])],
        initializer=[
            numpy_helper.from_array(w, "conv_w"), numpy_helper.from_array(b, "conv_b"),
            numpy_helper.from_array(fc_w, "fc_w"), numpy_helper.from_array(fc_b, "fc_b"),
            numpy_helper.from_array(np.array([1, 8], dtype=np.int64), "shape"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 6
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def main() -> int:
    from openvino.inference_engine import IECore
    from apps.common.ov import to_ir

    ie = IECore()
    devices = sorted(ie.available_devices)
    print(f"[smoke] OpenVINO 2020.3.2 devices: {devices}")
    if not any(d.startswith("CPU") for d in devices):
        print("[smoke] FAIL: no CPU plugin")
        return 1
    if not any(d.startswith("MYRIAD") for d in devices):
        print("[smoke] INFO: no MYRIAD device yet (driver contact is a separate step)")

    with tempfile.TemporaryDirectory(prefix="ncs1-smoke-") as tmp:
        onnx_path = Path(tmp) / "smoke.onnx"
        _build_onnx(onnx_path)

        import onnxruntime as ort
        rng = np.random.default_rng(42)
        inp = rng.standard_normal((1, 1, 8, 8)).astype(np.float32)
        ref = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ref_logits = ref.run(None, {"input": inp})[0]

        ir = to_ir(onnx_path, cache_dir=Path(tmp))
        net = ie.read_network(model=str(ir), weights=str(ir.with_suffix(".bin")))
        exec_net = ie.load_network(network=net, device_name="CPU", config={})
        in_name = next(iter(net.inputs))
        out_name = next(iter(net.outputs))
        ov_logits = exec_net.infer({in_name: inp})[out_name]

        max_diff = float(np.max(np.abs(ref_logits - ov_logits)))
        ok = np.allclose(ref_logits, ov_logits, rtol=1e-2, atol=1e-3)
        print(f"[smoke] reference (onnxruntime): {np.round(ref_logits[0], 4).tolist()}")
        print(f"[smoke] candidate (FP16 IR, IECore CPU): {np.round(ov_logits[0], 4).tolist()}")
        print(f"[smoke] max |diff| = {max_diff:.3e} (rtol=1e-2, atol=1e-3)")

    if ok:
        print("[smoke] PASS - ONNX -> FP16 IR -> IECore CPU matches onnxruntime")
        return 0
    print("[smoke] FAIL - logits diverge beyond FP16 tolerance")
    return 1


if __name__ == "__main__":
    sys.exit(main())
