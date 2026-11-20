"""Persistent gate daemon: loads the gate model ONCE, serves frames over stdin/stdout.

Amortizes the ~1.9 s Myriad firmware boot across the whole frame stream - the
production shape for an always-on gate. Reads one JSON line per frame
({"frame": "<path>", "ts": seconds?}), writes one JSON event line per frame
(an ncs2.event/1 event with a "fired" flag, via GateRunner.event_from_probs).

cv2-free by design: the daemon's preprocessing uses PIL + numpy, because the
OpenVINO DLL dirs on PATH break cv2's import in this venv combination (see
docs/RESEARCH_MYRIAD2_LIMITS_AND_MIRAGE.md). GateRunner.infer_nchw takes the
preprocessed NCHW array directly.

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

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from apps.gate.runner import DEFAULT_MODEL, GateRunner  # noqa: E402

SCALE = 256  # short-side resize, matching apps/common/vision.py


def load_png_as_nchw(path: str, channels: int, size: int) -> np.ndarray:
    """PNG -> the exact preprocessing frame_to_nchw applies (PIL instead of cv2)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    short = min(h, w)
    if short != SCALE:
        if h < w:
            nh, nw = SCALE, max(SCALE, round(w * SCALE / h))
        else:
            nh, nw = max(SCALE, round(h * SCALE / w)), SCALE
        img = img.resize((nw, nh), Image.BILINEAR)
    w, h = img.size
    y0 = max(0, (h - size) // 2)
    x0 = max(0, (w - size) // 2)
    crop = img.crop((x0, y0, x0 + size, y0 + size))
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    chw = arr.transpose(2, 0, 1)
    if channels != 3:
        chw = chw[:channels]
    return chw[None].astype(np.float32)


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
        try:
            nchw = load_png_as_nchw(frame_path, gate.channels, gate.h)
        except Exception as exc:  # noqa: BLE001
            print(json.dumps({"frame": frame_path, "error": f"unreadable: {exc}"}), flush=True)
            continue
        probs, latency_ms = gate.infer_nchw(nchw)
        event = gate.event_from_probs(probs, ts=msg.get("ts", 0.0), latency_ms=latency_ms)
        event["frame"] = frame_path
        print(json.dumps(event), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
