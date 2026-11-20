"""Footage triage - the WS5 pipeline pilot, driven by real media.

Usage (in the ncs2 venv, from repo root)::

    python -m apps.footage.triage data/samples/clip.mp4 --device myriad \\
        --fps 2 --watch "golden retriever" --threshold 0.3

Reads frames with OpenCV, gates each sampled frame through the decision
model (GateRunner), and writes an ``ncs2.index/1`` JSONL next to the video
(``<video stem>.index.jsonl``) plus a ``myriadforge.demo/1`` record in
``results/`` so the run shows up in ``myriadforge report``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import apps.common  # noqa: F401 - sys.path bootstrap, must precede myriadforge imports
from apps.common.events import IndexWriter
from apps.common.ov import lane_prefix, ov_device
from apps.gate.runner import DEFAULT_MODEL, GateRunner
from myriadforge.bench.schema import make_demo_record, save_demo


def triage(
    video_path: Path,
    gate: GateRunner,
    *,
    sample_fps: float = 2.0,
    out_dir: Path = Path("results"),
) -> dict:
    """One triage pass; returns the summary dict (also saved as a demo record).

    Each sampled frame is gated once; with gate.pipeline>1 the gating runs with
    continuous async scheduling (the next frame is submitted the instant any
    request completes, overlapping USB transfer with on-stick compute) for higher
    pass throughput. ``gated_fps`` is the wall-clock pass rate (the speedup metric).
    """
    import time

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video {video_path}")
    video_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(video_fps / max(sample_fps, 1e-6))))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    stamps: list[float] = []
    frames: list = []
    frame_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            stamps.append(frame_idx / max(video_fps, 1e-6))
            frames.append(frame)
        frame_idx += 1
    capture.release()

    t0 = time.perf_counter_ns()
    results = gate.infer_batch(frames)
    pass_ms = (time.perf_counter_ns() - t0) / 1e6

    index_path = video_path.with_suffix(".index.jsonl")
    latencies: list[float] = []
    events: list[dict] = []
    with IndexWriter(index_path, source=video_path.name,
                     model=gate.model_path.stem, device=gate.ov_name) as writer:
        for ts, (probs, latency_ms) in zip(stamps, results):
            latencies.append(latency_ms)
            event = gate.event_from_probs(probs, ts, latency_ms)
            if event["fired"]:
                events.append(event)
            writer.write(event)

    return {
        "video": str(video_path),
        "video_fps": round(video_fps, 2),
        "sample_fps": sample_fps,
        "frames_total": total_frames,
        "frames_gated": len(frames),
        "events_fired": len(events),
        "gate_p50_ms": round(float(np.percentile(latencies, 50)), 2) if latencies else None,
        "pipeline_requests": gate.pipeline,
        "pass_ms": round(pass_ms, 1),
        "gated_fps": round(len(frames) / max(pass_ms / 1000.0, 1e-9), 2) if frames else None,
        "index": str(index_path),
        "events": events[:10],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="WS5 footage triage (video -> gate -> index)")
    ap.add_argument("video")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--device", choices=["myriad", "cpu"], default="myriad")
    ap.add_argument("--shaves", type=int, default=16)
    ap.add_argument("--fps", type=float, default=2.0, help="gate sampling rate (frames/sec)")
    ap.add_argument("--watch", default="", help="comma-separated label substrings or class indices")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--labels", default="")
    ap.add_argument("--pipeline", type=int, default=2,
                    help="in-flight async gate requests (ma2450 sweet spot=2, ~+48%% throughput; 1=sync)")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    watch = [w.strip() for w in args.watch.split(",") if w.strip()] or None
    gate = GateRunner(
        args.model, device=args.device, shaves=args.shaves,
        watch=watch, threshold=args.threshold,
        labels_path=args.labels or None, pipeline=args.pipeline,
    )
    summary = triage(Path(args.video), gate, sample_fps=args.fps, out_dir=Path(args.out_dir))

    record = make_demo_record(
        demo="footage-triage",
        model={"name": gate.model_path.stem, "file": gate.model_path.name,
               "path": str(gate.model_path), "format": gate.model_path.suffix.lstrip(".")},
        device={"id": f"{lane_prefix()}-{args.device}" if args.device == "myriad" else "openvino-cpu",
                "lane": lane_prefix() if args.device == "myriad" else "host",
                "kind": "usb-accel" if args.device == "myriad" else "host",
                "label": f"Intel {lane_prefix().upper()} ({gate.ov_name})"},
        backend={"name": "openvino-myriad" if args.device == "myriad" else "openvino-cpu",
                 "runtime": "OpenVINO 2020.3.2 LTS" if lane_prefix() == "ncs1" else "OpenVINO 2022.3.1 LTS",
                 "accelerator": "myriad" if args.device == "myriad" else "cpu",
                 "simulated": False, "pipelined": args.pipeline > 1,
                 "pipeline_requests": args.pipeline},
        config={"video": summary["video"], "sample_fps": args.fps,
                "watched": watch, "threshold": args.threshold},
        metrics={"frames_gated": summary["frames_gated"],
                 "events_fired": summary["events_fired"],
                 "gate_p50_ms": summary["gate_p50_ms"]},
        outputs={"summary": summary},
    )
    out = save_demo(record, Path(args.out_dir))

    import json

    print(json.dumps({"summary": summary, "record": str(out)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
