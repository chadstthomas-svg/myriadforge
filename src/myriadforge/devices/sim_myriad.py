"""MYRIAD.SIM - behavioral simulator of the Intel NCS sticks (Myriad X / Myriad 2).

Enforces the hardware envelope as gates (per docs/NCS2_ROADMAP.md WS3):
static-shape ONNX/OpenVINO-IR models only, FP16 weight budget. Metrics come
from a documented latency model, NOT from execution - the record is stamped
``backend.simulated: true`` and every constant lives in a chip CALIBRATION
profile for replacement with measured values.

Two chip profiles:
* ``ma2485`` (NCS2 / Myriad X, 16 SHAVEs, ~1 GB) - synthetic constants until
  an NCS2 lands; functional truth via the OpenVINO 2022.3.1 CPU plugin.
* ``ma2450`` (NCS1 / Myriad 2, 12 SHAVEs fixed, ~0.4 GB usable estimate) -
  constants calibrated on real hardware 2026-11-19 (tools/ncs1/calibration.json).
  The 2020.3 Myriad 2 plugin has no SHAVE-count key, so shaves are fixed at 12.

This provider adds the *behavioral* layer: what fits, what gets rejected,
modeled latency. Real functional truth comes from the OpenVINO CPU plugin in
the isolated venvs (devices/ncs1.py, devices/ncs2.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from myriadforge.bench.schema import make_result, save_result

if TYPE_CHECKING:
    from myriadforge.devices.registry import DeviceHandle

# Myriad X (NCS2): synthetic until measured on an NCS2.
CALIBRATION: dict[str, float] = {
    "usb_overhead_ms": 0.8,
    "ddr_read_gb_s": 3.4,
    "ms_per_mb_full_shaves": 2.0,
    "shave_scaling_exponent": 0.7,
    "usb3_load_mb_s": 300.0,
    "load_fixed_ms": 500.0,
    "p90_factor": 1.15,
    "p99_factor": 1.30,
    "stddev_frac": 0.08,
    "first_inference_factor": 1.5,
}

# Myriad 2 (NCS1): calibrated on real ma2450 silicon 2026-11-19 from a
# MobileNetV2-7 gate point (tools/ncs1/calibration.json). Single-point fit;
# rough for other model sizes. Myriad 2 latency is very consistent (low variance).
MA2450_CALIBRATION: dict[str, float] = {
    "usb_overhead_ms": 0.8,
    "ddr_read_gb_s": 3.4,
    "ms_per_mb_full_shaves": 7.466,
    "shave_scaling_exponent": 0.7,
    "usb3_load_mb_s": 15.9,
    "load_fixed_ms": 1500.0,
    "p90_factor": 1.02,
    "p99_factor": 1.04,
    "stddev_frac": 0.02,
    "first_inference_factor": 1.0,
}

CHIP_PROFILES: dict[str, dict[str, Any]] = {
    "ma2485": {"calibration": CALIBRATION, "usable_gb": 1.0,
               "default_shaves": 16, "id": "myriad-sim", "lane": "NCS2"},
    "ma2450": {"calibration": MA2450_CALIBRATION, "usable_gb": 0.4,
               "default_shaves": 12, "id": "myriad2-sim", "lane": "NCS1"},
}

ACCEPTED_SUFFIXES = {".onnx", ".xml"}


def _latency_model(size_mb: float, shaves: int, calibration: dict | None = None) -> dict[str, Any]:
    """Deterministic latency estimate in ms for one inference."""
    c = calibration if calibration is not None else CALIBRATION
    ddr_ms = size_mb / c["ddr_read_gb_s"]
    compute_ms = size_mb * c["ms_per_mb_full_shaves"] * (16.0 / shaves) ** c["shave_scaling_exponent"]
    p50 = c["usb_overhead_ms"] + ddr_ms + compute_ms
    return {
        "p50": p50,
        "p90": p50 * c["p90_factor"],
        "p99": p50 * c["p99_factor"],
    }


class SimMyriadProvider:
    """MYRIAD.SIM: one handle per chip; N SHAVEs configurable (ma2485 only)."""

    name = "sim-myriad"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        self.chip = cfg.get("chip", "ma2485")
        prof = CHIP_PROFILES[self.chip]
        self.calibration = prof["calibration"]
        self.usable_gb = prof["usable_gb"]
        self.shaves = int(cfg.get("shave_count", prof["default_shaves"]))
        self.id = prof["id"]
        self.lane_label = prof["lane"]

    def scan(self) -> list[DeviceHandle]:
        from myriadforge.devices.registry import DeviceHandle

        return [
            DeviceHandle(
                lane=self.name,
                id=self.id,
                label=f"MYRIAD.SIM ({self.chip}-emulated, {self.shaves} shaves)",
                kind="sim",
                fingerprint={
                    "chip": f"{self.chip}-emulated",
                    "shave_count": self.shaves,
                    "precision": "fp16",
                    "usable_model_gb": self.usable_gb,
                    "note": "behavioral simulator - latency modeled, not executed",
                },
                bench_mode="simulated",
            )
        ]

    def bench(
        self,
        handle: DeviceHandle,
        model_path: Path,
        *,
        warmup_runs: int,
        measured_runs: int,
        out_dir: Path,
    ) -> Path:
        size_mb = model_path.stat().st_size / 1024**2
        if model_path.suffix.lower() not in ACCEPTED_SUFFIXES:
            return self._error_record(
                handle, model_path, out_dir,
                f"MYRIAD.SIM gate: format '{model_path.suffix}' not accepted; "
                f"{self.lane_label} lane takes ONNX/OpenVINO IR (got {model_path.name})",
            )
        if size_mb > self.usable_gb * 1024:
            return self._error_record(
                handle, model_path, out_dir,
                f"MYRIAD.SIM gate: model {size_mb:.1f} MB exceeds the "
                f"{self.usable_gb:.1f} GB on-stick budget ({self.chip})",
            )

        est = _latency_model(size_mb, self.shaves, self.calibration)
        c = self.calibration
        load_ms = c["load_fixed_ms"] + (size_mb / c["usb3_load_mb_s"]) * 1000.0
        ss = {
            "mean": est["p50"] * 1.05,
            "p50": est["p50"],
            "p90": est["p90"],
            "p99": est["p99"],
            "stddev": est["p50"] * c["stddev_frac"],
            "min": est["p50"] * 0.92,
            "max": est["p99"],
            "runs": float(measured_runs),
        }
        metrics = {
            "init_ms": round(load_ms, 3),
            "first_inference_ms": round(est["p50"] * c["first_inference_factor"], 3),
            "warmup_runs": warmup_runs,
            "steady_state_ms": {k: round(v, 3) for k, v in ss.items()},
            "throughput_fps": round(1000.0 / ss["p50"], 2),
            "memory_mb": {"peak_mb": 0.0, "current_mb": 0.0},
        }
        record = make_result(
            model={
                "name": model_path.stem,
                "file": model_path.name,
                "path": str(model_path),
                "format": model_path.suffix.lstrip("."),
                "size_mb": round(size_mb, 2),
            },
            device=handle.result_device(),
            backend={
                "name": "openvino-myriad",
                "runtime": f"MYRIAD.SIM behavioral model ({self.chip})",
                "accelerator": "myriad",
                "simulated": True,
                "shave_count": self.shaves,
                "usb_transport": True,
                "latency_model": "docs/NCS2_ROADMAP.md WS3 (chip CALIBRATION constants)",
            },
            config={"warmup_runs": warmup_runs, "measured_runs": measured_runs},
            metrics=metrics,
        )
        return save_result(record, out_dir)

    def _error_record(
        self, handle: DeviceHandle, model_path: Path, out_dir: Path, reason: str
    ) -> Path:
        """Fail-soft gate rejection: a status=error record, not an exception."""
        record = make_result(
            model={"name": model_path.stem, "file": model_path.name,
                   "path": str(model_path), "format": model_path.suffix.lstrip(".")},
            device=handle.result_device(),
            backend={"name": "openvino-myriad", "runtime": "MYRIAD.SIM",
                     "accelerator": "myriad", "simulated": True},
            config={},
            metrics={},
            status="error",
            error=reason,
        )
        return save_result(record, out_dir)
