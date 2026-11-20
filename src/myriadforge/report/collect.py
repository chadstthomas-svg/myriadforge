"""Collect and normalize result records from results/ for the report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BENCH_SCHEMA = "myriadforge.result/1"
DEMO_SCHEMA = "myriadforge.demo/1"


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load every valid record (bench + demo); unparseable files are skipped."""
    records: list[dict[str, Any]] = []
    if not results_dir.exists():
        return records
    for path in sorted(results_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if record.get("schema") in {BENCH_SCHEMA, DEMO_SCHEMA}:
            record["_file"] = path.name
            records.append(record)
    return records


def group_bench(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group bench records by model name, newest run per device kept last."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        if r.get("schema") != BENCH_SCHEMA:
            continue
        grouped.setdefault(r["model"]["name"], []).append(r)
    return grouped


def group_demos(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        if r.get("schema") != DEMO_SCHEMA:
            continue
        name = r.get("demo", {}).get("name", "?")
        grouped.setdefault(name, []).append(r)
    return grouped


def device_label(record: dict[str, Any]) -> str:
    device = record.get("device", {})
    return device.get("label") or device.get("id") or "unknown"


def device_id(record: dict[str, Any]) -> str:
    return record.get("device", {}).get("id", "unknown")


def is_simulated(record: dict[str, Any]) -> bool:
    return bool(record.get("backend", {}).get("simulated"))


def p50(record: dict[str, Any]) -> float | None:
    ss = record.get("metrics", {}).get("steady_state_ms", {})
    value = ss.get("p50")
    return float(value) if isinstance(value, (int, float)) else None


def baseline_map(grouped: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    """Per model: the host-cpu p50 to use as the speedup baseline (best-effort)."""
    baselines: dict[str, float] = {}
    for model, runs in grouped.items():
        cpu_p50s = [
            p50(r) for r in runs
            if device_id(r).startswith("host-cpu")
            and r.get("status", "ok") == "ok" and p50(r) is not None
        ]
        if cpu_p50s:
            baselines[model] = min(cpu_p50s)
    return baselines


def speedup(record: dict[str, Any], baseline: float | None) -> float | None:
    if baseline is None or record.get("status", "ok") != "ok":
        return None
    value = p50(record)
    if not value:
        return None
    return baseline / value
