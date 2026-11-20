"""Cascade orchestrator - gate on the NCS2, brain on the host.

Usage (in the ncs2 venv, from repo root)::

    python -m apps.cascade.run data/samples/gate_input.jpg --device myriad \\
        --watch "golden retriever" --threshold 0.3

Stage 1 (gate): GateRunner on the chosen OpenVINO device (MYRIAD on the
stick). Stage 2 (brain): Mirage caption via cactus-compute when installed,
otherwise the chain honestly reports the skip. The full latency chain
prints end-to-end and lands in results/ as a ``myriadforge.demo/1`` record.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import apps.common  # noqa: F401 - sys.path bootstrap, must precede myriadforge imports
from apps.gate.runner import DEFAULT_MODEL, GateRunner
from apps.common.ov import lane_prefix
from myriadforge.bench.schema import make_demo_record, save_demo


def run_mirage(image_path: Path) -> tuple[str | None, float | None, str | None]:
    """Returns (caption, mirage_ms, error_note) - honest skip when absent."""
    try:
        import cactus  # type: ignore[import-not-found]
    except ImportError:
        return None, None, ("cactus-compute not installed - caption stage skipped "
                            "(pip install cactus-compute); gate ran for real")
    t0 = time.perf_counter_ns()
    try:
        caption = cactus.Mirage().describe(str(image_path))
    except Exception as exc:  # noqa: BLE001 - the brain must fail soft
        return None, None, f"Mirage call failed: {exc}"
    return str(caption), (time.perf_counter_ns() - t0) / 1e6, None


def main() -> int:
    ap = argparse.ArgumentParser(description="WS5 cascade: NCS2 gate -> host Mirage")
    ap.add_argument("image")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--device", choices=["myriad", "cpu"], default="myriad")
    ap.add_argument("--shaves", type=int, default=16)
    ap.add_argument("--watch", default="", help="comma-separated label substrings or class indices")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--labels", default="")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    watch = [w.strip() for w in args.watch.split(",") if w.strip()] or None
    gate = GateRunner(
        args.model, device=args.device, shaves=args.shaves,
        watch=watch, threshold=args.threshold,
        labels_path=args.labels or None,
    )
    image_path = Path(args.image)
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(json.dumps({"error": f"cannot read image {args.image}"}))
        return 1

    event = gate.evaluate(frame, ts=0.0)
    fired = event is not None
    probs, gate_ms = gate.infer(frame)
    labels = gate.labels or [str(i) for i in range(len(probs))]
    if event is None:
        top = int(np.argmax(probs))
        event = {
            "schema": "ncs2.event/1", "ts": 0.0, "label": labels[top],
            "class_index": top, "confidence": round(float(probs[top]), 4),
            "device": gate.ov_name, "latency_ms": round(gate_ms, 3),
            "fired": False,
        }

    chain: dict = {"gate": round(gate_ms, 1)}
    outputs: dict = {"event": event, "top": None}
    top_idx = np.argsort(probs)[::-1][:5]
    outputs["top"] = [{"class": labels[int(i)], "score": float(probs[int(i)])} for i in top_idx]

    if fired:
        caption, mirage_ms, note = run_mirage(image_path)
        if mirage_ms is not None:
            chain["mirage"] = round(mirage_ms, 1)
        if caption:
            outputs["text"] = caption
        if note:
            outputs["note"] = note
        chain["triggered"] = "yes"
    else:
        chain["triggered"] = "no - gate below threshold, brain not invoked"
        outputs.setdefault("note", f"no watched class above {args.threshold}")

    chain["total"] = round(sum(v for v in chain.values() if isinstance(v, (int, float))), 1)
    metrics = {"gate_ms": chain["gate"], "triggered": fired, "total_ms": chain["total"]}

    record = make_demo_record(
        demo="cascade",
        model={"name": gate.model_path.stem, "file": gate.model_path.name,
               "path": str(gate.model_path), "format": gate.model_path.suffix.lstrip(".")},
        device={"id": f"{lane_prefix()}-{args.device}" if args.device == "myriad" else "openvino-cpu",
                "lane": lane_prefix() if args.device == "myriad" else "host",
                "kind": "usb-accel" if args.device == "myriad" else "host",
                "label": f"Intel {lane_prefix().upper()} ({gate.ov_name})"},
        backend={"name": "openvino-myriad" if args.device == "myriad" else "openvino-cpu",
                 "runtime": "OpenVINO 2022.3.1 LTS",
                 "accelerator": "myriad" if args.device == "myriad" else "cpu",
                 "simulated": False},
        config={"image": image_path.name, "watched": watch,
                "threshold": args.threshold, "gate_device": gate.ov_name},
        metrics=metrics,
        outputs={**outputs, "latency_chain": chain},
    )
    out = save_demo(record, Path(args.out_dir))
    print(json.dumps({"chain": chain, "outputs": outputs, "record": str(out)},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
