"""Podcast triage - the WS5 audio pipeline skeleton, driven by real media.

Usage (in the ncs2 venv, from repo root)::

    python -m apps.podcast.triage data/samples/podcast.wav --window-s 1.0 --hop-s 0.5

Windows the WAV, gates each window (energy/RMS VAD by default; pass
``--model`` to run a 1D-CNN gate through GateRunner), and writes an
``ncs2.index/1`` JSONL (timestamps + labels + confidences) next to the audio
plus a ``myriadforge.demo/1`` record in ``results/``.
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

import apps.common  # noqa: F401 - sys.path bootstrap, must precede myriadforge imports
from apps.common.events import IndexWriter
from myriadforge.bench.schema import make_demo_record, save_demo

RMS_THRESHOLD = 500.0  # int16 RMS above which a window counts as speech


def load_wav(path: Path) -> tuple[int, np.ndarray]:
    """Mono float32 samples + sample rate (stdlib wave; int16 PCM assumed)."""
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        n_channels = wav.getnchannels()
        width = wav.getsampwidth()
        if width != 2:
            raise ValueError(f"expected 16-bit PCM WAV, got sampwidth={width}")
        raw = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    if n_channels > 1:
        raw = raw.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
    return rate, raw.astype(np.float32)


def windows(rate: int, samples: np.ndarray, window_s: float, hop_s: float):
    """Yield (ts_seconds, window) at a fixed hop."""
    win = int(rate * window_s)
    hop = int(rate * hop_s)
    for start in range(0, max(1, len(samples) - win + 1), hop):
        yield start / rate, samples[start : start + win]


def rms_gate(window: np.ndarray, threshold: float) -> tuple[bool, float]:
    """Energy gate: (fired, rms). Honest stand-in for a neural VAD."""
    rms = float(np.sqrt(np.mean(window * window)))
    return rms >= threshold, rms


def triage(
    audio_path: Path,
    *,
    window_s: float = 1.0,
    hop_s: float = 0.5,
    threshold: float = RMS_THRESHOLD,
    out_dir: Path = Path("results"),
) -> dict:
    rate, samples = load_wav(audio_path)
    index_path = audio_path.with_suffix(".index.jsonl")
    events: list[dict] = []
    n_windows = 0

    with IndexWriter(index_path, source=audio_path.name,
                     model="rms-vad", device="host-cpu") as writer:
        for ts, win in windows(rate, samples, window_s, hop_s):
            n_windows += 1
            fired, rms = rms_gate(win, threshold)
            event = {
                "schema": "ncs2.event/1",
                "ts": round(ts, 3),
                "label": "speech" if fired else "silence",
                "class_index": 1 if fired else 0,
                "confidence": round(min(rms / max(threshold, 1e-6), 9.99), 4),
                "device": "host-cpu",
                "latency_ms": 0.0,
                "rms": round(rms, 1),
                "fired": fired,
            }
            if fired:
                events.append(event)
            writer.write(event)

    return {
        "audio": str(audio_path),
        "sample_rate": rate,
        "duration_s": round(len(samples) / rate, 2),
        "window_s": window_s,
        "hop_s": hop_s,
        "windows_gated": n_windows,
        "events_fired": len(events),
        "index": str(index_path),
        "note": "energy/RMS gate (neural 1D-CNN slot via --model pending)",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="WS5 podcast triage (WAV -> gate -> index)")
    ap.add_argument("audio")
    ap.add_argument("--window-s", type=float, default=1.0)
    ap.add_argument("--hop-s", type=float, default=0.5)
    ap.add_argument("--threshold", type=float, default=RMS_THRESHOLD)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    summary = triage(Path(args.audio), window_s=args.window_s,
                     hop_s=args.hop_s, threshold=args.threshold,
                     out_dir=Path(args.out_dir))

    record = make_demo_record(
        demo="podcast-triage",
        model={"name": "rms-vad", "file": "-", "path": "-", "format": "heuristic"},
        device={"id": "host-cpu", "lane": "host", "kind": "host",
                "label": "Energy/RMS VAD (ncs2 neural slot pending)"},
        backend={"name": "rms-vad", "runtime": "stdlib+numpy", "accelerator": "cpu",
                 "simulated": False},
        config={"window_s": args.window_s, "hop_s": args.hop_s,
                "threshold": args.threshold},
        metrics={"windows_gated": summary["windows_gated"],
                 "events_fired": summary["events_fired"],
                 "duration_s": summary["duration_s"]},
        outputs={"summary": summary},
    )
    out = save_demo(record, Path(args.out_dir))

    import json

    print(json.dumps({"summary": summary, "record": str(out)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
