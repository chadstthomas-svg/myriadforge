"""NCS1 lane: Intel Neural Compute Stick (Myriad 2) via OpenVINO 2020.3.2.

The original NCS1 (Myriad 2 MA2450, PID 0x2150) is only supported through
OpenVINO 2020.3.x - the MYRIAD boot table disabled Myriad 2 from 2020.4.
That runtime uses the older IECore API and Python 3.7, so it lives in its
own isolated venv (tools/ncs1/.venv); the main myriadforge env never imports
OpenVINO. This provider shells out to tools/ncs1/bridge_runner.py and is a
graceful no-op when the venv or dongle is absent. Configuration (devices.toml):

    [ncs1]
    python = "tools/ncs1/.venv/Scripts/python.exe"
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from myriadforge.bench.schema import make_result, save_result

if TYPE_CHECKING:
    from myriadforge.devices.registry import DeviceHandle

_BRIDGE = Path(__file__).resolve().parent.parent.parent.parent / "tools" / "ncs1" / "bridge_runner.py"
_DEFAULT_PY = _BRIDGE.parent / ".venv" / "Scripts" / "python.exe"


class Ncs1Provider:
    name = "ncs1"

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        cfg = cfg or {}
        self.python = Path(cfg.get("python", _DEFAULT_PY))
        # In-flight async requests: 2 is the measured ma2450 throughput sweet spot
        # (~+37% vs sync); 1 = single-shot latency bench. See docs/NCS1_ROADMAP.md.
        self.requests = int(cfg.get("requests", 2))

    def _venv_state(self) -> tuple[bool, str]:
        """(venv_ok, openvino summary) - never raises."""
        if not self.python.exists():
            return False, f"venv python not found at {self.python}"
        tools_dir = str(_BRIDGE.parent).replace("\\", "/")
        # Marker prefix: IECore may emit log noise (e.g. "[BSL] ...") to stdout
        # after the JSON, so parse by marker rather than "last line".
        snippet = (
            "import sys, json; "
            f"sys.path.insert(0, '{tools_dir}'); "
            "import ncs1_paths; ncs1_paths.setup(); "
            "from openvino.inference_engine import IECore; "
            "print('NCS1DEVS:' + json.dumps(sorted(IECore().available_devices)))"
        )
        try:
            proc = subprocess.run(
                [str(self.python), "-c", snippet],
                capture_output=True, text=True, timeout=120, errors="replace",
            )
            if proc.returncode != 0:
                return False, f"openvino probe failed: {proc.stderr.strip()[-200:]}"
            for line in proc.stdout.splitlines():
                if line.startswith("NCS1DEVS:"):
                    return True, line[len("NCS1DEVS:"):]
            return False, f"probe gave no device marker: {proc.stdout.strip()[-200:]}"
        except (OSError, subprocess.TimeoutExpired) as exc:
            return False, f"openvino probe error: {exc}"

    def scan(self) -> list[DeviceHandle]:
        from myriadforge.devices.registry import DeviceHandle

        ok, detail = self._venv_state()
        if not ok:
            return [DeviceHandle(
                lane=self.name, id="ncs1-myriad",
                label=f"Intel NCS1 - not ready ({detail})",
                kind="usb-accel",
                fingerprint={"detail": detail},
                available=False, bench_mode="subprocess",
            )]
        try:
            devices = json.loads(detail)
        except json.JSONDecodeError:
            devices = []
        has_myriad = any(d.startswith("MYRIAD") for d in devices)
        label = "Intel NCS1 (MYRIAD, Myriad 2)" if has_myriad else "NCS1 venv ready - no dongle (OpenVINO CPU)"
        return [DeviceHandle(
            lane=self.name,
            id="ncs1-myriad" if has_myriad else "ncs1-ovcpu",
            label=label,
            kind="usb-accel" if has_myriad else "host",
            fingerprint={"openvino_devices": devices, "myriad_platform": "ma2450",
                         "runtime": "OpenVINO 2020.3.2 LTS (isolated venv, Py3.7)"},
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
        device = "myriad" if handle.id == "ncs1-myriad" else "cpu"
        cmd = [
            str(self.python), str(_BRIDGE),
            "--model", str(model_path),
            "--device", device,
            "--warmup", str(warmup_runs),
            "--runs", str(measured_runs),
            "--requests", str(self.requests),
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
            line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("NCS1RESULT:"))
            payload = json.loads(line[len("NCS1RESULT:"):])
            return Path(payload["result"])
        except (StopIteration, json.JSONDecodeError, KeyError):
            return self._error(handle, model_path, out_dir,
                               f"bridge output unparseable: {proc.stdout[-200:]}")

    def _error(
        self, handle: DeviceHandle, model_path: Path, out_dir: Path, reason: str
    ) -> Path:
        record = make_result(
            model={"name": model_path.stem, "file": model_path.name,
                   "path": str(model_path), "format": model_path.suffix.lstrip(".")},
            device=handle.result_device(),
            backend={"name": "openvino-myriad", "runtime": "OpenVINO 2020.3.2",
                     "accelerator": "myriad"},
            config={},
            metrics={},
            status="error",
            error=reason,
        )
        return save_result(record, out_dir)
