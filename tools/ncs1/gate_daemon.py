"""Persistent gate daemon: loads the gate model ONCE, serves frames over stdin/stdout.

Amortizes the ~1.9 s Myriad firmware boot across the whole frame stream - the
production shape for an always-on gate. Reads one JSON line per frame
({"frame": "<path>", "ts": seconds?}), writes one JSON event line per frame
(an ncs2.event/1 event with a "fired" flag, via GateRunner.event_from_probs).

Usage (ncs1 venv):
  python gate_daemon.py --device myriad --threshold 0.3 [--model path] [--labels path]
Protocol: lines in -> lines out; a {"ready": true, "device": ...} line is printed
once the gate is loaded.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import ncs1_paths  # noqa: E402
ncs1_paths.setup()

import cv2  # noqa: E402
from apps.gate.runner import DEFAULT_MODEL, GateRunner  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="persistent NCS1 gate daemon (JSON lines)")
    ap.add_argument("--device", choices=["myriad", "cpu", "hetero"], default="myriad")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--labels", default="")
    args = ap.parse_args()

    gate = GateRunner(
        args.model, device=args.device, watch=None, threshold=args.threshold,
        labels_path=args.labels or None,
    )
    print(json.dumps({"ready": True, "device": gate.ov_name, "model": gate.model_path.stem,
                      "threshold": args.threshold}), flush=True)

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            print(json.dumps({"error": f"bad json: {exc}"}), flush=True)
            continue
        frame_path = msg.get("frame", "")
        frame = cv2.imread(frame_path)
        if frame is None:
            print(json.dumps({"frame": frame_path, "error": "unreadable image"}), flush=True)
            continue
        probs, latency_ms = gate.infer(frame)
        event = gate.event_from_probs(probs, ts=msg.get("ts", 0.0), latency_ms=latency_ms)
        event["frame"] = frame_path
        print(json.dumps(event), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
