"""WS0 CPU smoke test - runs INSIDE the isolated OpenVINO 2022.3.1 venv.

Proves the toolchain end to end without the dongle: build a small ONNX
model, convert to FP16 IR via Model Optimizer, run on the CPU plugin, and
compare logits to an onnxruntime reference (rtol 1e-2 - FP16 tolerance).
Env sanity: ov.Core() lists CPU/GPU; absence of MYRIAD is expected and
logged, not fatal (driver contact is a separate step).

Usage (from the ncs2 venv, any cwd):
  python tools/ncs2/smoke_cpu.py
Exit 0 = pass, nonzero = fail.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


def _build_onnx(path: Path) -> None:
    """Tiny deterministic ONNX graph: Conv(3x3) -> Relu -> GlobalAvgPool -> Gemm."""
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
        nodes, "ncs2_smoke",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 8, 8])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4])],
        initializer=[
            numpy_helper.from_array(w, "conv_w"),
            numpy_helper.from_array(b, "conv_b"),
            numpy_helper.from_array(fc_w, "fc_w"),
            numpy_helper.from_array(fc_b, "fc_b"),
            numpy_helper.from_array(np.array([1, 8], dtype=np.int64), "shape"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])
    model.ir_version = 8  # 2022.3 MO-era opset/IR combo
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _to_fp16_ir(onnx_path: Path, out_dir: Path) -> Path:
    """ONNX -> FP16 IR via the 2022.3 Model Optimizer API."""
    from openvino.runtime import serialize
    from openvino.tools.mo import convert_model

    import inspect

    kwargs = {}
    params = inspect.signature(convert_model).parameters
    if "compress_to_fp16" in params:
        kwargs["compress_to_fp16"] = True
    elif "data_type" in params:
        kwargs["data_type"] = "FP16"

    ov_model = convert_model(str(onnx_path), **kwargs)
    ir_path = out_dir / (onnx_path.stem + ".xml")
    serialize(ov_model, str(ir_path))
    return ir_path


def main() -> int:
    from openvino.runtime import Core

    core = Core()
    devices = sorted(core.available_devices)
    print(f"[smoke] OpenVINO devices: {devices}")
    if not any(d.startswith("CPU") for d in devices):
        print("[smoke] FAIL: no CPU plugin")
        return 1
    if not any(d.startswith("MYRIAD") for d in devices):
        # Expected until the dongle driver is installed - informational only.
        print("[smoke] INFO: no MYRIAD device yet (driver contact is a separate step)")

    with tempfile.TemporaryDirectory(prefix="ncs2-smoke-") as tmp:
        onnx_path = Path(tmp) / "smoke.onnx"
        _build_onnx(onnx_path)

        # Reference: onnxruntime on the original ONNX.
        import onnxruntime as ort

        rng = np.random.default_rng(42)
        inp = rng.standard_normal((1, 1, 8, 8)).astype(np.float32)
        ref = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ref_logits = ref.run(None, {"input": inp})[0]

        # Candidate: FP16 IR on the OpenVINO CPU plugin.
        ir_path = _to_fp16_ir(onnx_path, Path(tmp))
        compiled = core.compile_model(str(ir_path), "CPU")
        res = compiled.create_infer_request().infer({0: inp})
        out_sig = next(iter(compiled.outputs))
        ov_logits = res[out_sig]

        max_diff = float(np.max(np.abs(ref_logits - ov_logits)))
        ok = np.allclose(ref_logits, ov_logits, rtol=1e-2, atol=1e-3)
        print(f"[smoke] reference (onnxruntime): {np.round(ref_logits[0], 4).tolist()}")
        print(f"[smoke] candidate (FP16 IR, CPU plugin): {np.round(ov_logits[0], 4).tolist()}")
        print(f"[smoke] max |diff| = {max_diff:.3e} (rtol=1e-2, atol=1e-3)")

    if ok:
        print("[smoke] PASS - ONNX -> FP16 IR -> CPU plugin matches onnxruntime")
        return 0
    print("[smoke] FAIL - logits diverge beyond FP16 tolerance")
    return 1


if __name__ == "__main__":
    sys.exit(main())
