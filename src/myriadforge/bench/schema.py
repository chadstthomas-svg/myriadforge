"""myriadforge result record schema (versioned) and persistence."""

from __future__ import annotations

import hashlib
import json
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_ID = "myriadforge.result/1"
SCHEMA_DEMO_ID = "myriadforge.demo/1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_result(
    *,
    model: dict[str, Any],
    device: dict[str, Any],
    backend: dict[str, Any],
    config: dict[str, Any],
    metrics: dict[str, Any],
    status: str = "ok",
    error: str | None = None,
) -> dict[str, Any]:
    """Assemble one versioned result record."""
    record: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "run_id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host_platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "model": model,
        "device": device,
        "backend": backend,
        "config": config,
        "metrics": metrics,
        "status": status,
    }
    if error:
        record["error"] = error
    return record


def save_result(record: dict[str, Any], out_dir: Path) -> Path:
    """Write the record to results/<timestamp>_<model>_<accel>.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = record["model"]["name"].replace("/", "_").replace(".", "_")[:40]
    accel = record["backend"]["accelerator"]
    ts = record["timestamp"].replace(":", "").replace("-", "")[:15]
    path = out_dir / f"{ts}_{slug}_{accel}_{record['run_id']}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def make_demo_record(
    *,
    demo: str,
    model: dict[str, Any],
    device: dict[str, Any],
    backend: dict[str, Any],
    config: dict[str, Any],
    metrics: dict[str, Any],
    outputs: dict[str, Any] | None = None,
    status: str = "ok",
    error: str | None = None,
) -> dict[str, Any]:
    """Assemble one demo-run record (`myriadforge.demo/1`).

    Same envelope as result/1 plus a `demo` payload: `outputs` holds the
    human-visible artifacts (labels, captions, top-k, latency chain) that
    the report renders as demo views.
    """
    record = make_result(
        model=model,
        device=device,
        backend=backend,
        config=config,
        metrics=metrics,
        status=status,
        error=error,
    )
    record["schema"] = SCHEMA_DEMO_ID
    record["demo"] = {"name": demo, "outputs": outputs or {}}
    return record


def save_demo(record: dict[str, Any], out_dir: Path) -> Path:
    """Write a demo record as results/<timestamp>_<model>_<demo>_<device>.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = record["model"]["name"].replace("/", "_").replace(".", "_")[:40]
    name = record["demo"]["name"]
    device = record["device"].get("id", "device")
    ts = record["timestamp"].replace(":", "").replace("-", "")[:15]
    path = out_dir / f"{ts}_{slug}_{name}_{device}_{record['run_id']}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path
