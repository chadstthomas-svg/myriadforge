"""GateRunner - the WS5 gate as a small, reusable object.

Usage (in the ncs2 venv, from repo root)::

    python -m apps.gate.runner photo.jpg --model models/ncs2/mobilenetv2-12.onnx \\
        --device myriad --watch "golden retriever, Labrador retriever" --threshold 0.3

``--watch`` accepts comma-separated label substrings (matched against the
synset sidecar when present) or raw class indices. With no ``--watch``, the
gate fires on top-1 confidence >= threshold.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from apps.common.ov import compile_model, ov_device
from apps.common.vision import frame_to_nchw, input_hw, softmax
from apps.gate.synset import load_labels

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "models" / "ncs2" / "mobilenetv2-12.onnx"


class GateRunner:
    """Classifier gate emitting ``ncs2.event/1`` triggers."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL,
        *,
        device: str = "myriad",
        shaves: int = 16,
        watch: list[int | str] | None = None,
        threshold: float = 0.5,
        labels_path: Path | str | None = None,
        pipeline: int = 1,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self.ov_name = ov_device(device)
        self.threshold = float(threshold)
        self.pipeline = max(1, int(pipeline))
        self.compiled: Any = compile_model(model_path, device, shaves=shaves,
                                           num_requests=self.pipeline)
        self.channels, self.h, self.w = input_hw(self.compiled)
        self.labels = load_labels(labels_path or self.model_path.with_suffix(".labels.txt"))
        self.watch = self._resolve_watch(watch)

        self._sig = next(iter(self.compiled.inputs))
        self._out = next(iter(self.compiled.outputs))
        self._req = self.compiled.create_infer_request()

    def _resolve_watch(self, watch: list[int | str] | None) -> list[int]:
        """Label substrings -> class indices via the synset sidecar (best match)."""
        if not watch:
            return []
        idx: list[int] = []
        for item in watch:
            if isinstance(item, int) or str(item).isdigit():
                idx.append(int(item))
                continue
            needle = str(item).lower()
            if not self.labels:
                raise ValueError(
                    f"watch {item!r} is a label substring but no label file found "
                    f"(pass labels_path or use class indices)"
                )
            matches = [i for i, name in enumerate(self.labels) if needle in name.lower()]
            if not matches:
                raise ValueError(f"no class label matches {item!r}")
            idx.append(matches[0])
        return idx

    def infer(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """One gate inference: returns (probabilities, latency_ms)."""
        inp = frame_to_nchw(frame_bgr, channels=self.channels, size=self.h)
        t0 = time.perf_counter_ns()
        res = self._req.infer({self._sig: inp})
        latency_ms = (time.perf_counter_ns() - t0) / 1e6
        logits = np.asarray(res[self._out]).reshape(-1)
        return softmax(logits), latency_ms

    def infer_batch(self, frames_bgr: list) -> list:
        """Gate a batch of frames -> list[(probs, latency_ms)].

        With pipeline>1 on the IECore/ncs1 path, frames are gated with continuous
        async scheduling: N requests stay in flight and the next frame is submitted
        the instant any request completes (no wave barrier), overlapping USB
        transfer with on-stick compute (~+48% throughput on ma2450 at N=2). Falls
        back to sequential infer on the 2022.3 runtime path or when pipeline=1.
        """
        inputs = [frame_to_nchw(f, channels=self.channels, size=self.h) for f in frames_bgr]
        n = len(inputs)
        reqs = getattr(self.compiled, "requests", None)
        if self.pipeline > 1 and reqs and n:
            in_key = self._sig.name if hasattr(self._sig, "name") else self._sig
            out_key = self._out.name if hasattr(self._out, "name") else self._out
            nr = len(reqs)
            results: list = [None] * n
            slot_frame: dict = {}
            nxt = 0
            for slot in range(min(nr, n)):  # prime the pipeline
                reqs[slot].async_infer({in_key: inputs[nxt]})
                slot_frame[slot] = nxt
                nxt += 1
            completed = 0
            k = 0
            while completed < n:
                slot = k % nr
                k += 1
                if slot not in slot_frame:
                    continue
                reqs[slot].wait()
                fi = slot_frame.pop(slot)
                logits = np.asarray(reqs[slot].outputs[out_key]).reshape(-1)
                lat = float(reqs[slot].latency or 0.0)
                results[fi] = (softmax(logits), lat)
                completed += 1
                if nxt < n:  # immediately refill this slot with the next frame
                    reqs[slot].async_infer({in_key: inputs[nxt]})
                    slot_frame[slot] = nxt
                    nxt += 1
            return results
        return [self.infer(f) for f in frames_bgr]

    def event_from_probs(self, probs: np.ndarray, ts: float, latency_ms: float) -> dict:
        """Build a gate event (with a ``fired`` flag) from probabilities."""
        from apps.common.events import make_event

        if self.watch:
            best = max(self.watch, key=lambda i: probs[i])
        else:
            best = int(np.argmax(probs))
        fired = float(probs[best]) >= self.threshold
        event = make_event(
            ts=ts,
            label=self.labels[best] if self.labels else str(best),
            class_index=best,
            confidence=float(probs[best]),
            device=self.ov_name,
            latency_ms=latency_ms,
        )
        event["fired"] = fired
        return event

    def evaluate(
        self, frame_bgr: np.ndarray, ts: float = 0.0
    ) -> dict[str, Any] | None:
        """Gate one frame -> event dict when a watched class fires, else None.

        With no watch list, fires on top-1 >= threshold.
        """
        probs, latency_ms = self.infer(frame_bgr)
        event = self.event_from_probs(probs, ts, latency_ms)
        return event if event["fired"] else None


def main() -> int:
    ap = argparse.ArgumentParser(description="WS5 decision-model gate (single image)")
    ap.add_argument("image", help="image to gate")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--device", choices=["myriad", "cpu", "hetero"], default="myriad")
    ap.add_argument("--shaves", type=int, default=16)
    ap.add_argument("--watch", default="", help="comma-separated label substrings or class indices")
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--labels", default="", help="synset sidecar (default: <model>.labels.txt)")
    args = ap.parse_args()

    import cv2

    watch = [w.strip() for w in args.watch.split(",") if w.strip()] or None
    gate = GateRunner(
        args.model, device=args.device, shaves=args.shaves,
        watch=watch, threshold=args.threshold,
        labels_path=args.labels or None,
    )
    frame = cv2.imread(args.image)
    if frame is None:
        print(json.dumps({"error": f"cannot read image {args.image}"}))
        return 1

    event = gate.evaluate(frame, ts=0.0)
    if event is None:
        top = int(np.argmax(gate.infer(frame)[0]))
        labels = gate.labels or [str(top)]
        print(json.dumps({
            "fired": False,
            "note": f"no watched class above {args.threshold}; top-1: {labels[top]}",
            "device": gate.ov_name,
        }))
        return 0
    event["fired"] = True
    print(json.dumps(event, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
