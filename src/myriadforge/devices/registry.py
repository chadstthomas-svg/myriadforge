"""Device provider registry: the "auto-seen" layer for the Myriad lanes.

Every benchmarkable (or at least discoverable) device lane implements
DeviceProvider and is registered here. ``discover()`` scans all providers
and returns DeviceHandles; a failing lane degrades to an error note, never
an exception, so one unplugged dongle can't break the sweep.

Providers (focused on the Movidius Myriad family):
  ncs1        - Intel NCS1 / Myriad 2 via the isolated OpenVINO 2020.3 venv bridge
  ncs2        - Intel NCS2 / Myriad X via the isolated OpenVINO 2022.3 venv bridge
  sim-myriad  - MYRIAD.SIM behavioral simulators (ma2485 + ma2450, always available)
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

# Where provider-specific config lives (ncs1/ncs2 venvs, sim knobs).
DEFAULT_CONFIG_PATH = Path("devices.toml")


@dataclass
class DeviceHandle:
    """One discovered device, normalized across lanes."""

    lane: str  # provider name: "ncs1" | "ncs2" | "sim-myriad"
    id: str  # stable selector: "ncs1-myriad", "myriad-sim", "myriad2-sim", ...
    label: str  # human-readable
    kind: str  # "host" | "usb-accel" | "sim"
    fingerprint: dict[str, Any] = field(default_factory=dict)
    available: bool = True
    bench_mode: str = "native"  # native | subprocess | simulated | none

    def result_device(self) -> dict[str, Any]:
        """Device dict for make_result / make_demo_record."""
        return {
            "id": self.id,
            "lane": self.lane,
            "kind": self.kind,
            "label": self.label,
            **self.fingerprint,
        }


class DeviceProvider(Protocol):
    """Contract every lane implements. scan() never raises."""

    name: str

    def scan(self) -> list[DeviceHandle]: ...

    def bench(
        self,
        handle: DeviceHandle,
        model_path: Path,
        *,
        warmup_runs: int,
        measured_runs: int,
        out_dir: Path,
    ) -> Path:
        """Run one benchmark; return the saved result JSON path."""
        ...


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load devices.toml (empty dict when absent)."""
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def default_providers(config: dict[str, Any] | None = None) -> list[DeviceProvider]:
    """Instantiate the Myriad provider set in display order.

    Simulators are always included; hardware lanes self-report availability
    (a missing venv or unplugged stick still yields an explanatory handle).
    """
    from myriadforge.devices.ncs1 import Ncs1Provider
    from myriadforge.devices.ncs2 import Ncs2Provider
    from myriadforge.devices.sim_myriad import SimMyriadProvider

    cfg = config if config is not None else load_config()
    return [
        Ncs1Provider(cfg.get("ncs1", {})),
        Ncs2Provider(cfg.get("ncs2", {})),
        SimMyriadProvider(cfg.get("sim", {}).get("myriad", {})),
        SimMyriadProvider({"chip": "ma2450", **cfg.get("sim", {}).get("myriad2", {})}),
    ]


def discover(
    providers: list[DeviceProvider] | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[list[DeviceHandle], list[str]]:
    """Scan all providers; return (handles, warnings).

    A provider whose scan() raises is skipped with a warning rather than
    aborting discovery - the "auto-seen" contract.
    """
    provs = providers if providers is not None else default_providers(config)
    handles: list[DeviceHandle] = []
    warnings: list[str] = []
    for prov in provs:
        try:
            handles.extend(prov.scan())
        except Exception as exc:  # noqa: BLE001 - fail-soft by design
            warnings.append(f"{prov.name}: scan failed ({exc})")
    return handles, warnings


def find_handle(handles: list[DeviceHandle], device_id: str) -> DeviceHandle | None:
    return next((h for h in handles if h.id == device_id), None)


def benchable(handles: list[DeviceHandle]) -> list[DeviceHandle]:
    """Handles a workflow can actually run on right now."""
    return [h for h in handles if h.available and h.bench_mode != "none"]
