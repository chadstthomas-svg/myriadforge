"""Schema extension tests: demo records, save paths, additive fields."""

from __future__ import annotations

import json
from pathlib import Path

from myriadforge.bench.schema import (
    SCHEMA_DEMO_ID,
    make_demo_record,
    make_result,
    save_demo,
    save_result,
)


def _record() -> dict:
    return make_result(
        model={"name": "m", "file": "m.tflite", "path": "m.tflite", "format": "tflite"},
        device={"id": "host-cpu", "lane": "host", "kind": "host"},
        backend={"name": "litert", "accelerator": "cpu", "simulated": True},
        config={},
        metrics={"init_ms": 1.0},
    )


def test_make_result_accepts_lane_and_sim_fields() -> None:
    r = _record()
    assert r["backend"]["simulated"] is True
    assert r["device"]["lane"] == "host"
    assert r["schema"] == "myriadforge.result/1"


def test_demo_record_round_trip(tmp_path: Path) -> None:
    rec = make_demo_record(
        demo="knn",
        model={"name": "knn-100k", "file": "-", "path": "-", "format": "-"},
        device={"id": "myriad-sim", "lane": "sim-myriad", "kind": "sim"},
        backend={"name": "sim-myriad", "simulated": True},
        config={"n_vectors": 100_000},
        metrics={"search_ms": 12.5},
        outputs={"latency_chain": {"search": 12.5}, "top": [{"id": 1, "score": 0.9}]},
    )
    assert rec["schema"] == SCHEMA_DEMO_ID
    assert rec["demo"]["name"] == "knn"

    out = save_demo(rec, tmp_path)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == rec
    # filename: ts_slug_demo_device_runid
    assert "_knn_myriad-sim_" in out.name


def test_save_result_name_unchanged(tmp_path: Path) -> None:
    rec = _record()
    out = save_result(rec, tmp_path)
    assert "_cpu_" in out.name
    assert out.name.startswith("20")


def test_error_record_has_error_field(tmp_path: Path) -> None:
    rec = make_result(
        model={"name": "m", "file": "m", "path": "m", "format": "tflite"},
        device={"id": "d"},
        backend={"name": "b", "accelerator": "cpu"},
        config={},
        metrics={},
        status="error",
        error="gate rejection",
    )
    out = save_result(rec, tmp_path)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["status"] == "error"
    assert loaded["error"] == "gate rejection"
