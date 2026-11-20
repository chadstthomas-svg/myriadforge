"""Systematic Myriad 2 op-support matrix (NCS1_ROADMAP WS1, done empirically).

Builds a minimal ONNX graph per op, converts it with the 2020.3 Model Optimizer,
and tries a pure MYRIAD load+infer on the real ma2450. Each op runs in its own
SUBPROCESS with a timeout, so a firmware hang (like the wide-GEMM one) is recorded
as HANG instead of stalling the sweep.

Usage (ncs1 venv):
  python op_matrix.py                 # run the whole sweep, write op_support.json + docs/NCS1_OPS.md
  python op_matrix.py --op Softplus   # test a single op, print JSON

Output: op_support.json (committed) + docs/NCS1_OPS.md (generated table).
"""
from __future__ import annotations

import argparse
import json
import subprocess
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

TIMEOUT_S = 75
RNG = np.random.default_rng(7)


def _f32(name, shape):
    return numpy_helper.from_array(RNG.standard_normal(shape).astype(np.float32) * 0.1, name)


def _i64(name, values):
    return numpy_helper.from_array(np.array(values, dtype=np.int64), name)


def _i32(name, values):
    return numpy_helper.from_array(np.array(values, dtype=np.int32), name)


def _g(name, ins, outs, nodes, inits, opset=11):
    graph = helper.make_graph(nodes, name,
                              [helper.make_tensor_value_info(i, TensorProto.FLOAT, s) for i, s in ins],
                              [helper.make_tensor_value_info(o, t, s) for o, t, s in outs],
                              initializer=inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    m.ir_version = 6
    onnx.checker.check_model(m)
    return m


V1 = ("input", [1, 8])          # vector input
V4 = ("input", [1, 1, 8, 8])    # spatial input
F1 = ("output", TensorProto.FLOAT, [1, 8])

# name -> (family, builder(tmp_path) -> onnx.ModelProto)
OPS: dict[str, tuple[str, object]] = {
    # --- sanity anchors (known truth) ---
    "Conv": ("conv/pool", lambda p: _g("conv", [V4], [("output", TensorProto.FLOAT, [1, 8, 8, 8])],
                                       [helper.make_node("Conv", ["input", "w", "b"], ["output"],
                                                          kernel_shape=[3, 3], pads=[1, 1, 1, 1])],
                                       [_f32("w", (8, 1, 3, 3)), _f32("b", (8,))])),
    "Relu": ("activation", lambda p: _g("relu", [V1], [F1], [helper.make_node("Relu", ["input"], ["output"])], [])),
    # --- activations ---
    "Sigmoid": ("activation", lambda p: _g("sigmoid", [V1], [F1], [helper.make_node("Sigmoid", ["input"], ["output"])], [])),
    "Tanh": ("activation", lambda p: _g("tanh", [V1], [F1], [helper.make_node("Tanh", ["input"], ["output"])], [])),
    "LeakyRelu": ("activation", lambda p: _g("lrelu", [V1], [F1], [helper.make_node("LeakyRelu", ["input"], ["output"], alpha=0.01)], [])),
    "Prelu": ("activation", lambda p: _g("prelu", [V1], [F1], [helper.make_node("Prelu", ["input", "s"], ["output"])], [_f32("s", (1,))], opset=9)),
    "Elu": ("activation", lambda p: _g("elu", [V1], [F1], [helper.make_node("Elu", ["input"], ["output"], alpha=1.0)], [])),
    "Selu": ("activation", lambda p: _g("selu", [V1], [F1], [helper.make_node("Selu", ["input"], ["output"])], [])),
    "HardSigmoid": ("activation", lambda p: _g("hsig", [V1], [F1], [helper.make_node("HardSigmoid", ["input"], ["output"], alpha=0.2, beta=0.5)], [])),
    "Softplus": ("activation", lambda p: _g("softplus", [V1], [F1], [helper.make_node("Softplus", ["input"], ["output"])], [])),
    "Softsign": ("activation", lambda p: _g("softsign", [V1], [F1], [helper.make_node("Softsign", ["input"], ["output"])], [])),
    "Softmax": ("activation", lambda p: _g("softmax", [V1], [F1], [helper.make_node("Softmax", ["input"], ["output"], axis=1)], [])),
    "LogSoftmax": ("activation", lambda p: _g("logsoftmax", [V1], [F1], [helper.make_node("LogSoftmax", ["input"], ["output"], axis=1)], [])),
    "Erf": ("activation", lambda p: _g("erf", [V1], [F1], [helper.make_node("Erf", ["input"], ["output"])], [])),
    "Sign": ("activation", lambda p: _g("sign", [V1], [F1], [helper.make_node("Sign", ["input"], ["output"])], [])),
    # --- elementwise math ---
    "Add": ("math", lambda p: _g("add", [V1], [F1], [helper.make_node("Add", ["input", "b"], ["output"])], [_f32("b", (1, 8))])),
    "Mul": ("math", lambda p: _g("mul", [V1], [F1], [helper.make_node("Mul", ["input", "b"], ["output"])], [_f32("b", (1, 8))])),
    "Sub": ("math", lambda p: _g("sub", [V1], [F1], [helper.make_node("Sub", ["input", "b"], ["output"])], [_f32("b", (1, 8))])),
    "Div": ("math", lambda p: _g("div", [V1], [F1], [helper.make_node("Div", ["input", "b"], ["output"])], [_f32("b", (1, 8))])),
    "Pow": ("math", lambda p: _g("pow", [V1], [F1], [helper.make_node("Pow", ["input", "b"], ["output"])], [_f32("b", (1, 8))])),
    "Exp": ("math", lambda p: _g("exp", [V1], [F1], [helper.make_node("Exp", ["input"], ["output"])], [])),
    "Log": ("math", lambda p: _g("log", [V1], [F1], [helper.make_node("Log", ["input"], ["output"])], [])),
    "Sqrt": ("math", lambda p: _g("sqrt", [V1], [F1], [helper.make_node("Sqrt", ["input"], ["output"])], [])),
    "Abs": ("math", lambda p: _g("abs", [V1], [F1], [helper.make_node("Abs", ["input"], ["output"])], [])),
    "Neg": ("math", lambda p: _g("neg", [V1], [F1], [helper.make_node("Neg", ["input"], ["output"])], [])),
    "Floor": ("math", lambda p: _g("floor", [V1], [F1], [helper.make_node("Floor", ["input"], ["output"])], [])),
    "Round": ("math", lambda p: _g("round", [V1], [F1], [helper.make_node("Round", ["input"], ["output"])], [])),
    "MatMul": ("math", lambda p: _g("matmul", [V1], [F1], [helper.make_node("MatMul", ["input", "w"], ["output"])], [_f32("w", (8, 8))])),
    "Gemm": ("math", lambda p: _g("gemm", [V1], [F1], [helper.make_node("Gemm", ["input", "w", "b"], ["output"], transB=1)],
                                   [_f32("w", (4, 8)), _f32("b", (4,))])),
    # --- tensor manipulation ---
    "Transpose": ("tensor", lambda p: _g("transpose", [V1], [F1], [helper.make_node("Transpose", ["input"], ["output"], perm=[1, 0])], [])),
    "Reshape": ("tensor", lambda p: _g("reshape", [V1], [("output", TensorProto.FLOAT, [2, 4])],
                                       [helper.make_node("Reshape", ["input", "sh"], ["output"])], [_i64("sh", [2, 4])])),
    "Flatten": ("tensor", lambda p: _g("flatten", [V1], [("output", TensorProto.FLOAT, [1, 8])],
                                       [helper.make_node("Flatten", ["input"], ["output"], axis=1)], [])),
    "Squeeze": ("tensor", lambda p: _g("squeeze", [("input", [1, 1, 8])], [F1],
                                       [helper.make_node("Squeeze", ["input"], ["output"], axes=[1])], [])),
    "Unsqueeze": ("tensor", lambda p: _g("unsqueeze", [V1], [("output", TensorProto.FLOAT, [1, 1, 8])],
                                         [helper.make_node("Unsqueeze", ["input"], ["output"], axes=[1])], [])),
    "Concat": ("tensor", lambda p: _g("concat", [V1], [F1], [helper.make_node("Concat", ["input", "b"], ["output"], axis=1)], [_f32("b", (1, 8))])),
    "Slice11": ("tensor", lambda p: _g("slice11", [V1], [("output", TensorProto.FLOAT, [1, 4])],
                                       [helper.make_node("Slice", ["input", "st", "en", "ax"], ["output"])],
                                       [_i64("st", [0]), _i64("en", [4]), _i64("ax", [1])])),
    "Cast": ("tensor", lambda p: _g("cast", [V1], [("output", TensorProto.INT32, [1, 8])],
                                    [helper.make_node("Cast", ["input"], ["output"], to=TensorProto.INT32)], [])),
    "Gather": ("tensor", lambda p: _g("gather", [V1], [("output", TensorProto.FLOAT, [2])],
                                      [helper.make_node("Gather", ["input", "idx"], ["output"], axis=1)], [_i64("idx", [0, 3])])),
    "Tile11": ("tensor", lambda p: _g("tile11", [V1], [("output", TensorProto.FLOAT, [1, 16])],
                                      [helper.make_node("Tile", ["input", "rep"], ["output"])], [_i64("rep", [1, 2])])),
    "Expand8": ("tensor", lambda p: _g("expand8", [V1], [("output", TensorProto.FLOAT, [1, 16])],
                                       [helper.make_node("Expand", ["input", "sh"], ["output"])], [_i64("sh", [1, 16])])),
    "Shape": ("tensor", lambda p: _g("shape", [V1], [("output", TensorProto.INT64, [2])],
                                     [helper.make_node("Shape", ["input"], ["output"])], [])),
    "Pad11": ("tensor", lambda p: _g("pad11", [("input", [1, 1, 8, 8])],
                                     [("output", TensorProto.FLOAT, [1, 1, 10, 10])],
                                     [helper.make_node("Pad", ["input", "pads", "val"], ["output"])],
                                     [_i64("pads", [0, 0, 1, 1, 0, 0, 1, 1]), numpy_helper.from_array(np.array(0.0, dtype=np.float32), "val")])),
    "Transpose4d": ("tensor", lambda p: _g("transpose4d", [V4], [("output", TensorProto.FLOAT, [8, 1, 1, 8])],
                                           [helper.make_node("Transpose", ["input"], ["output"], perm=[2, 0, 1, 3])], [])),
    # --- conv / pool (spatial [1,1,8,8] unless noted) ---
    "MaxPool": ("conv/pool", lambda p: _g("maxpool", [V4], [("output", TensorProto.FLOAT, [1, 1, 8, 8])],
                                          [helper.make_node("MaxPool", ["input"], ["output"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])], [])),
    "AveragePool": ("conv/pool", lambda p: _g("avgpool", [V4], [("output", TensorProto.FLOAT, [1, 1, 8, 8])],
                                              [helper.make_node("AveragePool", ["input"], ["output"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])], [])),
    "GlobalAveragePool": ("conv/pool", lambda p: _g("gap", [V4], [("output", TensorProto.FLOAT, [1, 1, 1, 1])],
                                                    [helper.make_node("GlobalAveragePool", ["input"], ["output"])], [])),
    "GlobalMaxPool": ("conv/pool", lambda p: _g("gmp", [V4], [("output", TensorProto.FLOAT, [1, 1, 1, 1])],
                                                [helper.make_node("GlobalMaxPool", ["input"], ["output"])], [])),
    "ConvTranspose": ("conv/pool", lambda p: _g("convT", [V4], [("output", TensorProto.FLOAT, [1, 1, 10, 10])],
                                                [helper.make_node("ConvTranspose", ["input", "w"], ["output"], kernel_shape=[3, 3])],
                                                [_f32("w", (1, 1, 3, 3))])),
    "DepthToSpace": ("conv/pool", lambda p: _g("d2s", [("input", [1, 4, 2, 2])],
                                               [("output", TensorProto.FLOAT, [1, 1, 4, 4])],
                                               [helper.make_node("DepthToSpace", ["input"], ["output"], blocksize=2)], [])),
    "SpaceToDepth": ("conv/pool", lambda p: _g("s2d", [("input", [1, 1, 4, 4])],
                                               [("output", TensorProto.FLOAT, [1, 4, 2, 2])],
                                               [helper.make_node("SpaceToDepth", ["input"], ["output"], blocksize=2)], [])),
    "LpNormalization": ("conv/pool", lambda p: _g("lpnorm", [V1], [F1], [helper.make_node("LpNormalization", ["input"], ["output"], axis=1)], [])),
    "LRN": ("conv/pool", lambda p: _g("lrn", [V4], [("output", TensorProto.FLOAT, [1, 1, 8, 8])],
                                      [helper.make_node("LRN", ["input"], ["output"], size=5, alpha=0.0001, beta=0.75, bias=1.0)], [])),
    "BatchNormalization": ("conv/pool", lambda p: _g("bn", [V4], [("output", TensorProto.FLOAT, [1, 1, 8, 8])],
                                                     [helper.make_node("BatchNormalization", ["input", "s", "b", "m", "v"], ["output"])],
                                                     [_f32("s", (1,)), _f32("b", (1,)),
                                                      numpy_helper.from_array(np.ones((1,), dtype=np.float32), "m"),
                                                      numpy_helper.from_array(np.ones((1,), dtype=np.float32), "v")])),
    "Resize11": ("conv/pool", lambda p: _g("resize11", [V4], [("output", TensorProto.FLOAT, [1, 1, 8, 16])],
                                           [helper.make_node("Resize", ["input", "roi", "scales"], ["output"])],
                                           [numpy_helper.from_array(np.array([], dtype=np.float32), "roi"),
                                            numpy_helper.from_array(np.array([1, 1, 1, 2], dtype=np.float32), "scales")])),
    # --- reductions ---
    "ReduceMean": ("reduction", lambda p: _g("reducemean", [V1], [("output", TensorProto.FLOAT, [1, 1])],
                                             [helper.make_node("ReduceMean", ["input"], ["output"], axes=[1], keepdims=1)], [])),
    "ReduceSum": ("reduction", lambda p: _g("reducesum", [V1], [("output", TensorProto.FLOAT, [1, 1])],
                                            [helper.make_node("ReduceSum", ["input"], ["output"], axes=[1], keepdims=1)], [])),
    "ReduceMax": ("reduction", lambda p: _g("reducemax", [V1], [("output", TensorProto.FLOAT, [1, 1])],
                                            [helper.make_node("ReduceMax", ["input"], ["output"], axes=[1], keepdims=1)], [])),
    "ArgMax": ("reduction", lambda p: _g("argmax", [V1], [("output", TensorProto.INT64, [1, 1])],
                                         [helper.make_node("ArgMax", ["input"], ["output"], axis=1, keepdims=1)], [])),
    "ArgMin": ("reduction", lambda p: _g("argmin", [V1], [("output", TensorProto.INT64, [1, 1])],
                                         [helper.make_node("ArgMin", ["input"], ["output"], axis=1, keepdims=1)], [])),
    # --- data-dependent (the ones that most likely fall back) ---
    "TopK10": ("data-dependent", lambda p: _g("topk10", [V1],
                                              [("output", TensorProto.FLOAT, [1, 3]),
                                               ("idx", TensorProto.INT64, [1, 3])],
                                              [helper.make_node("TopK", ["input", "k"], ["output", "idx"], axis=1, largest=1)],
                                              [_i64("k", [3])])),
    "CumSum": ("data-dependent", lambda p: _g("cumsum", [V1], [F1],
                                              [helper.make_node("CumSum", ["input", "ax"], ["output"])],
                                              [_i32("ax", [1])])),
    "NonZero": ("data-dependent", lambda p: _g("nonzero", [V1],
                                               [("output", TensorProto.INT64, [1, None])],
                                               [helper.make_node("NonZero", ["input"], ["output"])], [])),
    "Where": ("data-dependent", lambda p: _g("where", [V1], [F1],
                                             [helper.make_node("Where", ["c", "input", "b"], ["output"])],
                                             [numpy_helper.from_array(RNG.standard_normal((1, 8)).astype(np.float32) > 0, "c"),
                                              _f32("b", (1, 8))])),
    "ScatterND": ("data-dependent", lambda p: _g("scatternd", [V1], [F1],
                                                 [helper.make_node("ScatterND", ["input", "idx", "upd"], ["output"])],
                                                 [_i64("idx", [[0]]), _f32("upd", (1,))])),
    "Clip_afterConv": ("data-dependent", lambda p: _g("clipac", [("input", [1, 1, 8, 8])], [F1],
        [helper.make_node("Conv", ["input", "w", "b"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
         helper.make_node("Clip", ["c", "mn", "mx"], ["output"])],
        [_f32("w", (1, 1, 3, 3)), _f32("b", (1,)),
         numpy_helper.from_array(np.array(-1.0, dtype=np.float32), "mn"),
         numpy_helper.from_array(np.array(1.0, dtype=np.float32), "mx")])),
    "Expand_afterConv": ("tensor", lambda p: _g("expandac", [("input", [1, 1, 8, 8])], [F1],
        [helper.make_node("Conv", ["input", "w", "b"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
         helper.make_node("Reshape", ["c", "sh1"], ["r"]),
         helper.make_node("Expand", ["r", "sh2"], ["output"])],
        [_f32("w", (1, 1, 3, 3)), _f32("b", (1,)), _i64("sh1", [1, 8]), _i64("sh2", [1, 16])])),
    "Shape_afterConv": ("tensor", lambda p: _g("shapeac", [("input", [1, 1, 8, 8])], [("out", TensorProto.INT64, [4])],
        [helper.make_node("Conv", ["input", "w", "b"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
         helper.make_node("Shape", ["c"], ["out"])],
        [_f32("w", (1, 1, 3, 3)), _f32("b", (1,))])),
    "Clip11": ("data-dependent", lambda p: _g("clip11", [V1], [F1],
                                              [helper.make_node("Clip", ["input", "mn", "mx"], ["output"])],
                                              [numpy_helper.from_array(np.array(-1.0, dtype=np.float32), "mn"),
                                               numpy_helper.from_array(np.array(1.0, dtype=np.float32), "mx")])),
}


def test_one(name: str) -> dict:
    """Child: build, convert, load+infer one op on MYRIAD. Prints a JSON line."""
    family, builder = OPS[name]
    with tempfile.TemporaryDirectory(prefix=f"op-{name}-") as tmp:
        tmp = Path(tmp)
        try:
            model = builder(tmp / f"{name}.onnx")
            p = tmp / f"{name}.onnx"
            onnx.save(model, str(p))
        except Exception as exc:  # noqa: BLE001
            return {"op": name, "family": family, "status": "CHECK_FAIL", "error": str(exc)[:140]}

        from apps.common.ov import to_ir

        try:
            ir = to_ir(p, cache_dir=tmp)
        except Exception as exc:  # noqa: BLE001
            return {"op": name, "family": family, "status": "MO_FAIL", "error": str(exc)[:140]}

        try:
            from openvino.inference_engine import IECore

            ie = IECore()
            net = ie.read_network(model=str(ir), weights=str(ir.with_suffix(".bin")))
            feeds = {}
            for in_name, sig in net.inputs.items():
                feeds[in_name] = np.random.default_rng(0).standard_normal(list(sig.shape)).astype(np.float32)
            # FORCE_RESET: give every op a freshly reset device so a crashed
            # predecessor cannot poison the next measurement.
            ex = ie.load_network(network=net, device_name="MYRIAD",
                                 config={"VPU_MYRIAD_FORCE_RESET": "YES"}, num_requests=1)
            ex.infer(feeds)
            return {"op": name, "family": family, "status": "OK", "error": ""}
        except Exception as exc:  # noqa: BLE001
            return {"op": name, "family": family, "status": "REJECT", "error": str(exc)[:140]}


def run_child(name: str) -> dict:
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--op", name],
            capture_output=True, text=True, timeout=TIMEOUT_S,
            cwd=str(HERE.parent.parent), errors="replace",
        )
        dt = time.perf_counter() - t0
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("{")), None)
        if line is None:
            return {"op": name, "family": OPS[name][0], "status": "CRASH",
                    "error": (proc.stderr or proc.stdout)[-140:]}
        res = json.loads(line)
        res["_dt"] = dt
        return res
    except subprocess.TimeoutExpired:
        return {"op": name, "family": OPS[name][0], "status": "HANG",
                "error": f"no result within {TIMEOUT_S}s"}


def is_device_transient(res: dict) -> bool:
    """NC_ERROR at device init is a transient host-USB wedge, not an op verdict."""
    return res["status"] == "REJECT" and "NC_ERROR" in res.get("error", "")


def write_outputs(ordered: list[dict]) -> None:
    counts: dict[str, int] = {}
    for r in ordered:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    payload = {
        "schema": "myriad2.op-support/1",
        "chip": "ma2450",
        "runtime": "OpenVINO 2020.3.2 LTS",
        "opset": 11,
        "date": time.strftime("%Y-%m-%d"),
        "method": "minimal per-op ONNX graph -> MO -> MYRIAD load+infer, one subprocess per op, "
                  "chunked with settle pauses; transient NC_ERROR retried once",
        "summary": counts,
        "results": ordered,
    }
    (HERE / "op_support.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Myriad 2 op-support matrix (empirical)",
        "",
        f"Generated {payload['date']} by `tools/ncs1/op_matrix.py` on real ma2450 silicon —",
        "OpenVINO 2020.3.2, ONNX opset 11, one minimal graph per op, subprocess-per-op with a",
        f"{TIMEOUT_S}s hang timeout. `MO_FAIL` = Model Optimizer could not convert the graph itself;",
        "`REJECT` = converted fine but the MYRIAD plugin refused to compile/infer it; `HANG` = never returned;",
        "`DEVICE_WEDGE` = the host USB path wedged before a verdict could be taken (transient, re-probe later).",
        "",
        f"**Summary:** {json.dumps(counts)} of {len(ordered)} ops.",
        "",
        "| Op | Family | Status | Detail |",
        "|---|---|---|---|",
    ]
    for r in ordered:
        err = (r.get("error", "") or "").replace("|", "/")[:80]
        lines.append(f"| `{r['op']}` | {r['family']} | {r['status']} | {err} |")
    lines += ["", "Unknown ops in a graph should be assumed HETERO-fallback until probed.",
              "", "_This table is generated - do not edit by hand._"]
    sep = chr(10)
    (HERE.parent.parent / "docs" / "NCS1_OPS.md").write_text(sep.join(lines) + sep, encoding="utf-8")
    print(f"{sep}wrote op_support.json + docs/NCS1_OPS.md — {json.dumps(counts)}", flush=True)


def run_sweep() -> int:
    results: dict[str, dict] = {}
    requeue: list[str] = []
    names = list(OPS)

    print(f"pass 1: {len(names)} ops (chunks of 8, 10 s settle between)", flush=True)
    for i, name in enumerate(names):
        if i > 0 and i % 8 == 0:
            print("  ... settling 10 s (rapid firmware boots wedge the host USB path)", flush=True)
            time.sleep(10)
        res = run_child(name)
        dt = res.pop("_dt", 0.0)
        if is_device_transient(res):
            requeue.append(name)
            print(f"[{i + 1}/{len(names)}] {name:<20} DEVICE-TRANSIENT - re-queued", flush=True)
            continue
        results[name] = res
        print(f"[{i + 1}/{len(names)}] {name:<20} {res['status']:<10} ({dt:.1f}s) "
              f"{res.get('error', '')[:70]}", flush=True)

    if requeue:
        print(f"\nsettle 20 s, then re-run {len(requeue)} device-transient ops", flush=True)
        time.sleep(20)
        for i, name in enumerate(requeue):
            res = run_child(name)
            dt = res.pop("_dt", 0.0)
            if is_device_transient(res):
                res["status"] = "DEVICE_WEDGE"
                res["error"] = "unmeasurable this session (host USB transient)"
            results[name] = res
            print(f"[re {i + 1}/{len(requeue)}] {name:<20} {res['status']:<10} "
                  f"{res.get('error', '')[:70]}", flush=True)
            if i < len(requeue) - 1:
                time.sleep(8)

    ordered = [results[name] for name in names if name in results]
    counts: dict[str, int] = {}
    for r in ordered:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    write_outputs(ordered)

    # generated docs table
    lines = [
        "# Myriad 2 op-support matrix (empirical)",
        "",
        f"Generated {payload['date']} by `tools/ncs1/op_matrix.py` on real ma2450 silicon —",
        f"OpenVINO 2020.3.2, ONNX opset 11, one minimal graph per op, subprocess-per-op with a",
        f"{TIMEOUT_S}s hang timeout. `MO_FAIL` = Model Optimizer could not convert the graph itself;",
        "`REJECT` = converted fine but the MYRIAD plugin refused to compile/infer it; `HANG` = never returned;",
        "`DEVICE_WEDGE` = the host USB path wedged before a verdict could be taken (transient, re-probe later).",
        "",
        f"**Summary:** {json.dumps(counts)} of {len(results)} ops.",
        "",
        "| Op | Family | Status | Detail |",
        "|---|---|---|---|",
    ]
    for r in ordered:
        err = (r.get("error", "") or "").replace("|", "/")[:80]
        lines.append(f"| `{r['op']}` | {r['family']} | {r['status']} | {err} |")
    lines += ["", "Unknown ops in a graph should be assumed HETERO-fallback until probed.",
              "", "_This table is generated - do not edit by hand._"]
    (HERE.parent.parent / "docs" / "NCS1_OPS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote op_support.json + docs/NCS1_OPS.md — {json.dumps(counts)}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", help="test a single op and print JSON")
    ap.add_argument("--merge", help="merge a JSON file of results into op_support.json and regenerate docs")
    args = ap.parse_args()
    if args.merge:
        incoming = json.loads(Path(args.merge).read_text(encoding="utf-8"))
        existing = json.loads((HERE / "op_support.json").read_text(encoding="utf-8"))
        by_op = {r["op"]: r for r in existing["results"]}
        for r in incoming if isinstance(incoming, list) else incoming.get("results", []):
            by_op[r["op"]] = r
        ordered = [by_op[name] for name in OPS if name in by_op]
        ordered += [r for op, r in by_op.items() if op not in OPS]
        write_outputs(ordered)
        return 0
    if args.op:
        if args.op not in OPS:
            print(json.dumps({"error": f"unknown op {args.op}"}))
            return 2
        print(json.dumps(test_one(args.op)))
        return 0
    return run_sweep()


if __name__ == "__main__":
    sys.exit(main())
