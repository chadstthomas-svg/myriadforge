"""kNN-as-MatMul engine - promoted from the demo layer to a real app.

Builds an ONNX graph whose only weights are the (transposed) vector table,
converts to FP16 IR, compiles on the chosen device, and benches batched
queries: real queries/sec/stick when the dongle is present, OpenVINO CPU as
the functional baseline. TopK stays on the host (the roadmap contract).

Usage (in the ncs2 venv, from repo root)::

    python -m apps.knn_engine.engine --n 100000 --dim 256 --batch 32 \\
        --queries 100 --device myriad
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import apps.common  # noqa: F401 - sys.path bootstrap, must precede myriadforge imports
from apps.common.ov import compile_model, lane_prefix, ov_device
from apps.knn_engine.capacity import check
from myriadforge.bench.schema import make_demo_record, save_demo


def build_table_onnx(table: np.ndarray, out_path: Path, batch: int = 32) -> Path:
    """ONNX with the table as constant MatMul weights; static input [batch, dim].

    MYRIAD pins shapes at convert time (roadmap envelope: static shapes only),
    so the query batch is baked into the graph rather than left dynamic.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    n, dim = table.shape
    table_t = np.ascontiguousarray(table.T.astype(np.float16))  # [dim, N]
    # Host interface stays FP32; the GEMM runs FP16 (the stick's native math).
    cast = helper.make_node("Cast", ["query"], ["query_f16"], to=TensorProto.FLOAT16)
    node = helper.make_node("MatMul", ["query_f16", "table_t"], ["scores"])
    graph = helper.make_graph(
        [cast, node], "knn_as_matmul",
        [helper.make_tensor_value_info("query", TensorProto.FLOAT, [batch, dim])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT16, [batch, n])],
        initializer=[numpy_helper.from_array(table_t, "table_t")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 6  # portable across onnx 1.7 (ncs1) and 1.12 (ncs2)
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    return out_path


def bench(
    model_path: Path,
    *,
    dim: int,
    batch: int = 32,
    n_queries: int = 100,
    device: str = "myriad",
    shaves: int = 16,
) -> dict:
    """Batched-query bench: (warmup + measured) timed batches -> qps."""
    # Fresh IR cache beside the ONNX: the table graph is parameterized, so a
    # persistent stem-keyed cache could serve a stale IR (wrong n/batch).
    compiled = compile_model(model_path, device, shaves=shaves, cache_dir=model_path.parent)
    sig = next(iter(compiled.inputs))
    out_sig = next(iter(compiled.outputs))

    rng = np.random.default_rng(42)
    inp = rng.standard_normal((batch, dim)).astype(np.float32)

    req = compiled.create_infer_request()

    def one_batch() -> np.ndarray:
        return np.asarray(req.infer({sig: inp})[out_sig])

    one_batch()  # first contact (may include weight streaming)
    t0 = time.perf_counter_ns()
    for _ in range(max(1, n_queries // batch)):
        one_batch()
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    queries_done = max(1, n_queries // batch) * batch
    return {
        "device": ov_device(device),
        "batch": batch,
        "queries": queries_done,
        "total_ms": round(elapsed_ms, 2),
        "ms_per_batch": round(elapsed_ms / max(1, queries_done // batch), 3),
        "queries_per_sec": round(queries_done / max(elapsed_ms / 1000.0, 1e-9), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="kNN-as-MatMul engine on the NCS2")
    ap.add_argument("--n", type=int, default=100_000, help="vectors in the table")
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--device", choices=["myriad", "cpu", "hetero"], default="myriad")
    ap.add_argument("--shaves", type=int, default=16)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    cap = check(args.n, args.dim)
    if not cap.fits:
        print(json.dumps({"error": "table does not fit the stick budget",
                          "capacity": cap.as_dict()}))
        return 1

    rng = np.random.default_rng(42)
    table = rng.standard_normal((args.n, args.dim)).astype(np.float16).astype(np.float32)

    with tempfile.TemporaryDirectory(prefix="knn-engine-") as tmp:
        onnx_path = build_table_onnx(table, Path(tmp) / "knn_table.onnx", batch=args.batch)
        result = bench(onnx_path, dim=args.dim, batch=args.batch, n_queries=args.queries,
                       device=args.device, shaves=args.shaves)

    payload = {"capacity": cap.as_dict(), "bench": result}
    record = make_demo_record(
        demo="knn-engine",
        model={"name": f"knn-{args.n // 1000}k-x{args.dim}", "file": "knn_table.onnx",
               "path": "-", "format": "onnx"},
        device={"id": f"{lane_prefix()}-{args.device}" if args.device == "myriad" else "openvino-cpu",
                "lane": lane_prefix() if args.device == "myriad" else "host",
                "kind": "usb-accel" if args.device == "myriad" else "host",
                "label": f"kNN GEMM on {result['device']}"},
        backend={"name": {"myriad": "openvino-myriad", "cpu": "openvino-cpu", "hetero": "openvino-hetero"}[args.device],
                 "runtime": "OpenVINO 2022.3.1 LTS",
                 "accelerator": "myriad" if args.device == "myriad" else "cpu",
                 "shave_count": args.shaves if args.device == "myriad" else None,
                 "simulated": False},
        config={"n_vectors": args.n, "dim": args.dim, "batch": args.batch,
                "queries": args.queries},
        metrics={"queries_per_sec": result["queries_per_sec"],
                 "ms_per_batch": result["ms_per_batch"],
                 "table_mb_fp16": cap.as_dict()["table_mb_fp16"]},
        outputs={"summary": payload},
    )
    out = save_demo(record, Path(args.out_dir))
    print(json.dumps({"summary": payload, "record": str(out)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
