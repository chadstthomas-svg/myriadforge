"""NCS2 lane: Intel Neural Compute Stick via the isolated OpenVINO 2022.3 venv.

The main myriadforge env never imports OpenVINO (the MYRIAD plugin is frozen
at 2022.3.1 LTS and Python <=3.10); this provider shells out to the venv's
python running tools/ncs2/bridge_runner.py. Graceful no-op when the venv
or the dongle is absent. Configuration (devices.toml):

    [ncs2]
    python = "tools/ncs2/.venv/Scripts/python.exe"   # venv python
    shaves = 16
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from myriadforge.bench.schema import make_result, save_result

if TYPE_CHECKING:
    from myriadforge.devices.registry import DeviceHandle

_BRIDGE = Path(__file__).resolve().parent.parent.parent.parent / "tools" / "ncs2" / "bridge_runner.py"
_DEFAULT_PY = _BRIDGE.parent / ".venv" / "Scripts" / "python.exe"


class Ncs2Provider:
    name = "ncs2"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        self.python = Path(cfg.get("python", _DEFAULT_PY))
        self.shaves = int(cfg.get("shaves", 16))

    def _venv_state(self) -> tuple[bool, str]:
        """(venv_ok, openvino summary) - never raises."""
        if not self.python.exists():
            return False, f"venv python not found at {self.python}"
        try:
            proc = subprocess.run(
                [str(self.python), "-c",
                 "from openvino.runtime import Core; import json;"
                 "print(json.dumps(sorted(Core().available_devices)))"],
                capture_output=True, text=True, timeout=60, errors="replace",
            )
            if proc.returncode != 0:
                return False, f"openvino probe failed: {proc.stderr.strip()[-200:]}"
            return True, proc.stdout.strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"openvino probe error: {exc}"

    def scan(self) -> list[DeviceHandle]:
        from myriadforge.devices.registry import DeviceHandle

        ok, detail = self._venv_state()
        if not ok:
            return [DeviceHandle(
                lane=self.name, id="ncs2-myriad",
                label=f"Intel NCS2 - not ready ({detail})",
                kind="usb-accel",
                fingerprint={"detail": detail},
                available=False, bench_mode="subprocess",
            )]
        try:
            devices = json.loads(detail)
        except json.JSONDecodeError:
            devices = []
        has_myriad = any(d.startswith("MYRIAD") for d in devices)
        label = "Intel NCS2 (MYRIAD)" if has_myriad else "NCS2 venv ready - no dongle (OpenVINO CPU)"
        return [DeviceHandle(
            lane=self.name,
            id="ncs2-myriad" if has_myriad else "ncs2-ovcpu",
            label=label,
            kind="usb-accel" if has_myriad else "host",
            fingerprint={"openvino_devices": devices, "shave_count": self.shaves,
                         "runtime": "OpenVINO 2022.3.1 LTS (isolated venv)"},
            available=True,
            bench_mode="subprocess",
        )]

    def bench(
        self,
        handle: DeviceHandle,
        model_path: Path,
        *,
        warmup_runs: int,
        measured_runs: int,
        out_dir: Path,
    ) -> Path:
        device = "myriad" if handle.id == "ncs2-myriad" else "cpu"
        cmd = [
            str(self.python), str(_BRIDGE),
            "--model", str(model_path),
            "--device", device,
            "--shaves", str(self.shaves),
            "--warmup", str(warmup_runs),
            "--runs", str(measured_runs),
            "--out", str(out_dir),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=1800, errors="replace")
        except (OSError, subprocess.TimeoutExpired) as exc:
            return self._error(handle, model_path, out_dir, f"bridge error: {exc}")
        if proc.returncode != 0:
            return self._error(
                handle, model_path, out_dir,
                f"bridge_runner exited {proc.returncode}: "
                f"{(proc.stderr or proc.stdout).strip()[-300:]}",
            )
        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
            return Path(payload["result"])
        except (json.JSONDecodeError, KeyError, IndexError):
            return self._error(handle, model_path, out_dir,
                               f"bridge output unparseable: {proc.stdout[-200:]}")

    def _error(
        self, handle: DeviceHandle, model_path: Path, out_dir: Path, reason: str
    ) -> Path:
        record = make_result(
            model={"name": model_path.stem, "file": model_path.name,
                   "path": str(model_path), "format": model_path.suffix.lstrip(".")},
            device=handle.result_device(),
            backend={"name": "openvino-myriad", "runtime": "OpenVINO 2022.3.1",
                     "accelerator": "myriad"},
            config={},
            metrics={},
            status="error",
            error=reason,
        )
        return save_result(record, out_dir)
