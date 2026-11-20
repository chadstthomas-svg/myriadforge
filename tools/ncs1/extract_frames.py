"""Extract sample frames from a video for the cascade benchmark (ncs1 venv, has cv2).

Usage:
  python extract_frames.py <video> --fps 6 --out data/samples/frames [--max 60]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--fps", type=float, default=6.0, help="frames per second of video to sample")
    ap.add_argument("--out", default="data/samples/frames")
    ap.add_argument("--max", type=int, default=60, help="cap on extracted frames")
    args = ap.parse_args()

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        print(f"error: cannot open {args.video}")
        return 2
    video_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(video_fps / max(args.fps, 1e-6))))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    frame_idx = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_idx % stride == 0 and n < args.max:
            ts = frame_idx / max(video_fps, 1e-6)
            path = out_dir / f"frame_{n:04d}_t{ts:07.3f}.png"
            cv2.imwrite(str(path), frame)
            n += 1
        frame_idx += 1
    capture.release()
    print(f"extracted {n} frames (video {video_fps:.1f} fps, stride {stride}) -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
