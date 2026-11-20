"""Predictive HETERO partitioner: what will fall back to CPU before touching the stick?

Walks an ONNX model's op types, looks each up in the measured op-support table
(tools/ncs1/op_support.json from op_matrix.py), and predicts the partition:
which ops run on the Myriad, which fall back to the host CPU, and whether the
model needs HETERO at all.

Usage (ncs1 venv):
  python hetero_plan.py model.onnx              # predict partition
  python hetero_plan.py --demo                  # self-test on Conv->HardSigmoid
Exit 0; prints JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import ncs1_paths  # noqa: E402
ncs1_paths.setup()

TABLE = HERE / "op_support.json"

# Ops that appear in graphs but are never real compute decisions.
IGNORE = {"Constant", "Dropout", "Identity"}


def load_table(path: Path = TABLE) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {r["op"]: r["status"] for r in data["results"]}


def predict_partition(op_types: list[str], table: dict[str, str]) -> dict:
    """Pure function: op types + measured table -> the predicted device partition.

    OK          -> runs on the Myriad
    REJECT/HANG -> falls back to CPU under HETERO
    MO_FAIL/CHECK_FAIL/absent -> UNKNOWN; HETERO will try it and the runtime decides,
                                 but flag it so the user knows it was never measured.
    """
    per_op = []
    fallback: list[str] = []
    unknown: list[str] = []
    myriad_ops: list[str] = []
    seen: set[str] = set()
    for op in op_types:
        if op in IGNORE or op in seen:
            continue
        seen.add(op)
        status = table.get(op)
        if status == "OK":
            per_op.append({"op": op, "device": "MYRIAD", "status": "OK"})
            myriad_ops.append(op)
        elif status in ("REJECT", "HANG"):
            per_op.append({"op": op, "device": "CPU", "status": status})
            fallback.append(op)
        else:
            label = status if status else "UNMEASURED"
            per_op.append({"op": op, "device": "UNKNOWN", "status": label})
            unknown.append(op)
    if unknown:
        verdict = "HETERO required (contains unmeasured ops - runtime decides at load)"
    elif fallback:
        verdict = "HETERO required (" + ", ".join(fallback) + " on CPU)"
    else:
        verdict = "MYRIAD (every measured op is supported)"
    return {
        "verdict": verdict,
        "myriad_ops": myriad_ops,
        "fallback_ops": fallback,
        "unknown_ops": unknown,
        "per_op": per_op,
    }


def collect_op_types(model_path: Path) -> list[str]:
    import onnx

    model = onnx.load(str(model_path))
    return [n.op_type for n in model.graph.node]


def main() -> int:
    ap = argparse.ArgumentParser(description="Predict the HETERO partition for an ONNX model")
    ap.add_argument("model", nargs="?", help="path to an ONNX model")
    ap.add_argument("--demo", action="store_true",
                    help="self-test on a synthesized Conv->HardSigmoid graph")
    args = ap.parse_args()

    table = load_table()
    if not table:
        print(json.dumps({"error": f"empty/missing support table at {TABLE} - run op_matrix.py first"}))
        return 2

    if args.demo:
        import capability_demo  # noqa: PLC0415 - builds the Conv->HardSigmoid graph

        with tempfile.TemporaryDirectory(prefix="hetero-plan-") as tmp:
            path = Path(tmp) / "conv_hardsigmoid.onnx"
            capability_demo.build_conv_hardsigmoid(path)
            op_types = collect_op_types(path)
    else:
        if not args.model:
            print(json.dumps({"error": "pass a model path or --demo"}))
            return 2
        op_types = collect_op_types(Path(args.model))

    prediction = predict_partition(op_types, table)
    print(json.dumps({"model": args.model or "(demo: Conv->HardSigmoid)",
                      "op_types": op_types, **prediction}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
