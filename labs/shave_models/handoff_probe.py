"""Handoff experiment (research plan step 4): can MO 2020.3 + IECore return TWO outputs?

One 2 s audio graph with a tiny tool-score head AND a frame/window value head,
through ONNX (opset 11) -> MO FP16 -> MYRIAD. If both outputs come back with
the right shapes and finite values, the shared multi-head model is trainable;
if not, the heads split into separate graphs. Also measures warm latency of
the [1,1,40,200] fit-class, which sizes the voice-to-tool budget.

Steps (both inside the NCS1 Python 3.7 venv):
  handoff_probe.py export --out bundles/handoff          build ONNX (onnx 1.7 helpers,
                                                         seeded numpy weights) + fixture NPZ
  handoff_probe.py probe --out bundles/handoff           MO FP16 -> MYRIAD -> verdict
                          [--toolchain D:/NTVStudy/tools/ncs1]

The graph is deliberately tiny (~4k params): Conv(1->16, kernel 1x9) -> ReLU ->
MaxPool -> Conv(16->24, kernel 1x9) -> ReLU -> MaxPool -> GlobalAvgPool ->
Gemm scores [1,13] (12 tools + reject) and Gemm value [1,1]. The 1-D temporal
conv is expressed as 2-D kernel [1,9] because Conv1d wedges the device
(docs/NCS1_OPS.md).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
SEED = 132


def build_graph():
    import onnx
    from onnx import helper, TensorProto

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
        helper.make_node("Reshape", ["gap", "shape24"], ["flat"], name="flat"),
        helper.make_node("Gemm", ["flat", "scores.w", "scores.b"], ["scores"], name="scores"),
        helper.make_node("Gemm", ["flat", "value.w", "value.b"], ["value"], name="value"),
    ]
    rng = np.random.RandomState(SEED)
    # Xavier-ish scaling: trained-regime activations stay well inside FP16
    # (raw N(0,1) kernels overflow the stick's half-precision accumulator).
    def scaled(shape, fan_in):
        return (rng.randn(*shape) * (1.0 / np.sqrt(fan_in))).astype(np.float32)
    inits = [
        helper.make_tensor("stem.w", TensorProto.FLOAT, [16, 1, 1, 9],
                           scaled([16, 1, 1, 9], 9).ravel().tolist()),
        helper.make_tensor("stem.b", TensorProto.FLOAT, [16],
                           (rng.randn(16) * 0.01).astype(np.float32).ravel().tolist()),
        helper.make_tensor("block.w", TensorProto.FLOAT, [24, 16, 1, 9],
                           scaled([24, 16, 1, 9], 144).ravel().tolist()),
        helper.make_tensor("block.b", TensorProto.FLOAT, [24],
                           (rng.randn(24) * 0.01).astype(np.float32).ravel().tolist()),
        helper.make_tensor("shape24", TensorProto.INT64, [2], [1, 24]),
        helper.make_tensor("scores.w", TensorProto.FLOAT, [24, 13],
                           scaled([24, 13], 24).ravel().tolist()),
        helper.make_tensor("scores.b", TensorProto.FLOAT, [13],
                           (rng.randn(13) * 0.01).astype(np.float32).ravel().tolist()),
        helper.make_tensor("value.w", TensorProto.FLOAT, [24, 1],
                           scaled([24, 1], 24).ravel().tolist()),
        helper.make_tensor("value.b", TensorProto.FLOAT, [1],
                           (rng.randn(1) * 0.01).astype(np.float32).ravel().tolist()),
    ]
    graph = helper.make_graph(
        nodes, "voicetools-handoff",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 1, 40, 200])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 13]),
         helper.make_tensor_value_info("value", TensorProto.FLOAT, [1, 1])],
        initializer=inits)
    return helper.make_model(graph, producer_name="myriadforge-handoff",
                             opset_imports=[helper.make_opsetid("", 11)])


def cmd_export(out):
    import onnx
    out.mkdir(parents=True, exist_ok=True)
    model = build_graph()
    onnx_path = out / "model.onnx"
    onnx.save(model, str(onnx_path))
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    rng = np.random.RandomState(SEED + 1)
    fixtures = rng.randn(8, 1, 40, 200).astype(np.float32)
    np.savez(str(Path("D:/models/ncs-shave-research/voicetools-handoff-fixtures.npz")),
             x=fixtures)
    print(f"exported {onnx_path} ({onnx_path.stat().st_size} bytes, opset 11, "
          f"outputs scores[1,13] + value[1,1]), fixtures.npz 8x[1,1,40,200]")


def cmd_probe(out, toolchain, fixtures_path):
    sys.path.insert(0, str(toolchain.resolve()))
    import ncs1_paths
    ncs1_paths.setup()
    from openvino.inference_engine import IECore

    source = out / "model.onnx"
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    ir = out / "ir" / sha
    ir.mkdir(parents=True, exist_ok=True)
    xml, binary = ir / "model.xml", ir / "model.bin"
    if not (xml.is_file() and binary.is_file()):
        env = dict(os.environ, PYTHONPATH=str(ncs1_paths.MO))
        subprocess.run([sys.executable, str(ncs1_paths.MO / "mo_onnx.py"),
                        "--input_model", str(source), "--data_type", "FP16",
                        "--output_dir", str(ir)], env=env, check=True, timeout=120)

    core = IECore()
    devices = [d for d in core.available_devices]
    net = core.read_network(model=str(xml), weights=str(binary))
    input_name = list(net.input_info if hasattr(net, "input_info") else net.inputs)[0]
    output_names = [str(o) for o in net.outputs]
    start = time.perf_counter()
    executable = core.load_network(network=net, device_name="MYRIAD", num_requests=1)
    load_ms = (time.perf_counter() - start) * 1000

    fixtures = np.load(str(fixtures_path))["x"]
    for _ in range(5):
        executable.infer({input_name: fixtures[0]})
    elapsed, shapes, stats = [], {}, {}
    for sample in fixtures:
        start = time.perf_counter()
        outs = executable.infer({input_name: sample})
        elapsed.append((time.perf_counter() - start) * 1000)
        for name, tensor in outs.items():
            shapes[name] = list(tensor.shape)
            acc = stats.setdefault(name, {"max_abs": 0.0, "nonfinite_samples": 0})
            if np.isfinite(tensor).all():
                acc["max_abs"] = max(acc["max_abs"], float(np.abs(tensor).max()))
            else:
                acc["nonfinite_samples"] += 1
    finite = all(s["nonfinite_samples"] == 0 for s in stats.values())

    verdict = {
        "onnx_sha256": sha,
        "available_devices": devices,
        "requested_outputs": ["scores", "value"],
        "returned_outputs": output_names,
        "two_output_supported": len(output_names) == 2,
        "output_shapes": shapes,
        "output_stats": stats,
        "all_finite": finite,
        "load_ms": round(load_ms, 1),
        "infer_p50_ms": round(float(np.percentile(elapsed, 50)), 2),
        "infer_p95_ms": round(float(np.percentile(elapsed, 95)), 2),
        "ncs1_runtime": "OpenVINO 2020.3.2",
    }
    (out / "probe.json").write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    if not verdict["two_output_supported"]:
        print("VERDICT: single output - the heads must split into separate graphs")
    else:
        print("VERDICT: two outputs returned - shared multi-head graph is trainable")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step", choices=["export", "probe"])
    ap.add_argument("--out", type=Path, default=LAB / "bundles" / "handoff")
    ap.add_argument("--toolchain", type=Path,
                    default=Path("D:/NTVStudy/tools/ncs1"))
    ap.add_argument("--fixtures", type=Path,
                    default=Path("D:/models/ncs-shave-research/voicetools-handoff-fixtures.npz"))
    args = ap.parse_args()
    import onnx  # noqa: F401  (both steps live in the NCS1 venv; fail fast if missing)
    {"export": lambda: cmd_export(args.out),
     "probe": lambda: cmd_probe(args.out, args.toolchain, args.fixtures)}[args.step]()


if __name__ == "__main__":
    main()
