"""Gate trigger contract (`ncs2.event/1`) and index writer (`ncs2.index/1`).

Every WS5 pilot speaks the same two contracts. The gate emits events; the
footage/podcast pipelines (and cascade) append them to a JSONL index of
timestamps + labels + confidences that downstream search/editing consumes.
Keeping the contract versioned and JSON-only is what lets the gate device
swap between MYRIAD and the CPU plugin without touching consumers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EVENT_SCHEMA = "ncs2.event/1"
INDEX_SCHEMA = "ncs2.index/1"


def make_event(
    *,
    ts: float,
    label: str,
    class_index: int,
    confidence: float,
    device: str,
    latency_ms: float,
) -> dict[str, Any]:
    """One gate trigger event (the WS5 trigger contract)."""
    return {
        "schema": EVENT_SCHEMA,
        "ts": round(float(ts), 3),
        "label": str(label),
        "class_index": int(class_index),
        "confidence": round(float(confidence), 4),
        "device": str(device),
        "latency_ms": round(float(latency_ms), 3),
    }


@dataclass
class IndexWriter:
    """Append-only JSONL index: header line, then one event per line."""

    path: Path
    source: str = ""
    model: str = ""
    device: str = ""

    def __enter__(self) -> "IndexWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        header = {
            "schema": INDEX_SCHEMA,
            "source": self.source,
            "model": self.model,
            "device": self.device,
        }
        self._fh.write(json.dumps(header) + "\n")
        return self

    def write(self, event: dict[str, Any]) -> None:
        self._fh.write(json.dumps(event) + "\n")

    def __exit__(self, exc_type, exc, tb) -> None:
        self._fh.close()
