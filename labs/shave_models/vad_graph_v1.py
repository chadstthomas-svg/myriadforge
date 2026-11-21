"""VAD graph transplant (vad latency iteration): trained weights -> handoff-pattern graph.

The torch opset-9 export of the vad model measures p95 ~9.9 ms on the stick
while the builder-built handoff graph (same input, MORE compute) measures
p50 5.0 ms - the export path, not the math, costs the difference. This script
rebuilds the SAME trained weights into the handoff-style opset-11 graph
(Conv/Relu/MaxPool x2 -> GlobalAveragePool -> Reshape -> Gemm) with onnx
1.7 helpers, then the standard eval_vad flow measures it.

Usage (NCS1 Python 3.7 venv)::
    python vad_graph_v1.py            # build bundles/vad-v1graph from bundles/vad
    eval_vad.py bundles/vad-v1graph   # device eval vs the 6.0 ms gate
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

LAB = Path(__file__).resolve().parent
SRC = LAB / "bundles" / "vad"
DEST = LAB / "bundles" / "vad-v1graph"


def build():
    src = onnx.load(str(SRC / "model.onnx"))
    inits = {i.name: numpy_helper.to_array(i) for i in src.graph.initializer}
    w = {k: np.asarray(v, dtype=np.float32) for k, v in inits.items()}
    print("source initializers:", {k: list(v.shape) for k, v in w.items()})

    nodes = [
        helper.make_node("Conv", ["features", "stem.w", "stem.b"], ["stem"],
                         name="stem", pads=[0, 4, 0, 4]),
        helper.make_node("Relu", ["stem"], ["stem.r"]),
        helper.make_node("MaxPool", ["stem.r"], ["stem.p"], kernel_shape=[1, 2], strides=[1, 2]),
        helper.make_node("Conv", ["stem.p", "block.w", "block.b"], ["block"],
                         name="block", pads=[0, 4, 0, 4]),
        helper.make_node("Relu", ["block"], ["block.r"]),
        helper.make_node("MaxPool", ["block.r"], ["block.p"], kernel_shape=[1, 2], strides=[1, 2]),
        helper.make_node("GlobalAveragePool", ["block.p"], ["gap"]),
        helper.make_node("Reshape", ["gap", "shape16"], ["flat"], name="flat"),
        helper.make_node("Gemm", ["flat", "head.w", "head.b"], ["logits"], name="head"),
    ]
    graph = helper.make_graph(
        nodes, "vad-v1graph",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 1, 40, 200])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 3])],
        initializer=[
            numpy_helper.from_array(w["stem.weight"], "stem.w"),
            numpy_helper.from_array(w["stem.bias"], "stem.b"),
            numpy_helper.from_array(w["block1.weight"], "block.w"),
            numpy_helper.from_array(w["block1.bias"], "block.b"),
            numpy_helper.from_array(np.array([1, 16], dtype=np.int64), "shape16"),
            numpy_helper.from_array(w["head.weight"].T.copy(), "head.w"),
            numpy_helper.from_array(w["head.bias"], "head.b"),
        ])
    model = helper.make_model(graph, producer_name="myriadforge-vad-v1graph",
                              opset_imports=[helper.make_opsetid("", 11)])
    onnx.checker.check_model(model)

    DEST.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(DEST / "model.onnx"))
    info = {
        "model": "vad-v1graph",
        "classes": list(("speech", "noise", "silence")),
        "input_shape": [1, 1, 40, 200],
        "source_digest": hashlib.sha256((SRC / "model.onnx").read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256((DEST / "model.onnx").read_bytes()).hexdigest(),
        "note": "trained weights transplanted into the builder-built opset-11 graph "
                "(handoff pattern); latency lane experiment",
    }
    (DEST / "model.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {DEST / 'model.onnx'}")


if __name__ == "__main__":
    build()
