"""OpenVINO runtime helpers for the apps/ pilots - dual-backend compat layer.

Two OpenVINO generations are supported behind one surface so the same apps
run on either stick:

* **NCS2 / Myriad X** - OpenVINO 2022.3.1 (``openvino.runtime.Core``), the
  ncs2 venv (Python 3.10). MYRIAD plugin from the LTS archive.
* **NCS1 / Myriad 2** - OpenVINO 2020.3.2 (``openvino.inference_engine.IECore``),
  the ncs1 venv (Python 3.7). 2020.3 is the *last* release whose MYRIAD boot
  table accepts the Myriad 2 PID 0x2150 (disabled from 2020.4). Its runtime
  is not pip-installed; it lives in tools/ncs1/ov2020 and needs TBB + ngraph
  + the Intel C++ runtime (libmmd/svml_dispmd) on PATH.

``compile_model`` returns either a real 2022.3 CompiledModel or a thin
``_Compiled`` wrapper that mimics the small 2022.3 surface the apps use
(``inputs``/``outputs`` ports with ``get_partial_shape()``,
``create_infer_request().infer({port: arr})``). Backend detection is automatic:
whichever OpenVINO imports in the current venv wins.

Note: the 2020.3 Myriad 2 plugin exposes **no SHAVE-count config key** (fixed
12-SHAVE allocation), so ``shaves`` is honored only on the 2022.3 Myriad X path.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from myriadforge.myriad_x import compile_myriad_x

REPO_ROOT = Path(__file__).resolve().parents[2]
IR_CACHE_DIR = REPO_ROOT / "models" / "ncs2" / "ir"
NCS1_IR_CACHE_DIR = REPO_ROOT / "models" / "ncs1" / "ir"
NCS1_OV2020 = REPO_ROOT / "tools" / "ncs1" / "ov2020"
NCS1_REDIST = REPO_ROOT / "tools" / "ncs1" / "redist"

_BACKEND: tuple[str, Any] | None = None


def _wire_iecore() -> None:
    """Put the extracted 2020.3.2 runtime on PATH/sys.path (mirrors setupvars.bat)."""
    ie = NCS1_OV2020 / "deployment_tools" / "inference_engine"
    release = ie / "bin" / "intel64" / "Release"
    dirs = [
        NCS1_REDIST,
        ie / "external" / "tbb" / "bin",
        release,
        ie / "bin" / "intel64" / "Debug",
        ie / "external" / "hddl" / "bin",
        NCS1_OV2020 / "deployment_tools" / "ngraph" / "lib",
    ]
    os.environ["PATH"] = os.pathsep.join(str(d) for d in dirs) + os.pathsep + os.environ.get("PATH", "")
    for p in (NCS1_OV2020 / "python" / "python3.7",
              NCS1_OV2020 / "python" / "python3",
              NCS1_OV2020 / "deployment_tools" / "model_optimizer"):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def _detect() -> tuple[str, Any]:
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND
    try:
        from openvino.runtime import Core  # 2022.3+ (ncs2 venv)
        _BACKEND = ("runtime", Core)
    except ImportError:
        _wire_iecore()
        from openvino.inference_engine import IECore  # 2020.3 (ncs1 venv)
        _BACKEND = ("iecore", IECore)
    return _BACKEND


def backend_name() -> str:
    return _detect()[0]


def lane_prefix() -> str:
    """Device-id lane prefix matching the active venv: ncs1 (IECore) / ncs2 (runtime)."""
    return "ncs1" if backend_name() == "iecore" else "ncs2"


def core() -> Any:
    """A fresh Core (2022.3) or IECore (2020.3) instance."""
    _, cls = _detect()
    return cls()


def available_devices() -> list[str]:
    c = core()
    return list(c.available_devices)


def ov_device(device: str) -> str:
    """Map app-level device names to OpenVINO plugin names.

    ``hetero`` -> HETERO:MYRIAD,CPU — the capability shim: ops the Myriad 2
    can't compile (e.g. HardSigmoid) fall back to the host CPU automatically,
    widening the runnable model set while keeping supported ops on the stick.
    """
    d = device.lower().strip()
    if d == "myriad":
        return "MYRIAD"
    if d == "cpu":
        return "CPU"
    if d == "hetero":
        return "HETERO:MYRIAD,CPU"
    raise ValueError(f"unknown device {device!r}; expected 'myriad', 'cpu', or 'hetero'")


# --- IECore compat wrapper: mimic the 2022.3 surface the apps use ---------

class _Dim:
    def __init__(self, n: int) -> None:
        self._n = int(n)

    @property
    def is_static(self) -> bool:
        return True

    def get_length(self) -> int:
        return self._n


class _Port:
    """Hashable input/output port keyed by blob name (2022.3 ports are hashable)."""

    def __init__(self, name: str, shape: Any) -> None:
        self.name = name
        self._shape = list(shape)

    def get_partial_shape(self) -> list[_Dim]:
        return [_Dim(d) for d in self._shape]

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Port) and other.name == self.name


class _Request:
    def __init__(self, exec_net: Any, out_ports: list[_Port]) -> None:
        self._net = exec_net
        self._out = out_ports

    def infer(self, feeds: dict) -> dict:
        named = {(k.name if isinstance(k, _Port) else k): v for k, v in feeds.items()}
        res = self._net.infer(named)
        return {p: res[p.name] for p in self._out}


class _Compiled:
    def __init__(self, exec_net: Any, net: Any) -> None:
        self.inputs = [_Port(n, net.inputs[n].shape) for n in net.inputs]
        self.outputs = [_Port(n, net.outputs[n].shape) for n in net.outputs]
        self._net = exec_net

    def create_infer_request(self) -> _Request:
        return _Request(self._net, self.outputs)

    @property
    def requests(self) -> list:
        """Raw IECore InferRequests for async pipelining (async_infer/wait).

        With compile_model(num_requests=N>1), keep N in flight to overlap USB
        transfer with on-stick compute (~+48% throughput on ma2450 at N=2).
        """
        return self._net.requests


# --- conversion + compile -------------------------------------------------

def _to_ir_runtime(src: Path, cache_dir: Path) -> Path:
    from openvino.runtime import serialize
    from openvino.tools.mo import convert_model

    import inspect

    kwargs: dict[str, Any] = {}
    params = inspect.signature(convert_model).parameters
    if "compress_to_fp16" in params:
        kwargs["compress_to_fp16"] = True
    elif "data_type" in params:
        kwargs["data_type"] = "FP16"
    ov_model = convert_model(str(src), **kwargs)
    ir = cache_dir / (src.stem + ".xml")
    serialize(ov_model, str(ir))
    return ir


def _to_ir_iecore(src: Path, cache_dir: Path) -> Path:
    """ONNX -> FP16 IR via the 2020.3 mo_onnx.py CLI (no python convert API)."""
    mo_dir = NCS1_OV2020 / "deployment_tools" / "model_optimizer"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(mo_dir) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, str(mo_dir / "mo_onnx.py"),
           "--input_model", str(src), "--data_type", "FP16",
           "--output_dir", str(cache_dir), "--model_name", src.stem]
    proc = subprocess.run(cmd, cwd=str(mo_dir), env=env,
                          capture_output=True, text=True, errors="replace")
    ir = cache_dir / (src.stem + ".xml")
    if proc.returncode != 0 or not ir.exists():
        raise RuntimeError(f"mo_onnx conversion failed for {src.name}: "
                           f"{(proc.stderr or proc.stdout)[-400:]}")
    return ir


def to_ir(model_path: str | Path, cache_dir: Path | None = None) -> Path:
    """ONNX -> FP16 IR with a disk cache; OpenVINO IR (.xml) passes through."""
    src = Path(model_path)
    if src.suffix.lower() == ".xml":
        return src
    kind, _ = _detect()
    cache = Path(cache_dir) if cache_dir else (IR_CACHE_DIR if kind == "runtime" else NCS1_IR_CACHE_DIR)
    cache.mkdir(parents=True, exist_ok=True)
    ir = cache / (src.stem + ".xml")
    if ir.exists():
        return ir
    return _to_ir_runtime(src, cache) if kind == "runtime" else _to_ir_iecore(src, cache)


def compile_model(model_path: str | Path, device: str = "myriad", *, shaves: int = 16,
                  cache_dir: Path | None = None, num_requests: int = 1) -> Any:
    """Load + compile on the chosen device (converted + cached if ONNX).

    cache_dir overrides the IR cache location; parameterized graphs (e.g. the
    kNN table) pass a fresh temp dir so a stem-keyed cache never serves a
    stale IR built for different shapes/weights.

    num_requests>1 (IECore/ncs1) allocates N in-flight async InferRequests for
    pipelined throughput; use ``compiled.requests`` with async_infer/wait. The
    ma2450 sweet spot is 2 (~+48% vs sync). Ignored on the 2022.3 runtime path.
    """
    kind, _ = _detect()
    ir = to_ir(model_path, cache_dir)
    target = ov_device(device)
    if kind == "runtime":
        session = core()
        if target == "MYRIAD":
            return compile_myriad_x(session, ir, int(shaves))
        return session.compile_model(str(ir), target)
    # IECore: config values must be strings; Myriad 2 has no SHAVE-count key.
    net = core().read_network(model=str(ir), weights=str(ir.with_suffix(".bin")))
    config: dict[str, str] = {}
    # HETERO partitioning uses a single request (async pipelining is a MYRIAD path).
    nr = 1 if target.startswith("HETERO") else max(1, num_requests)
    exec_net = core().load_network(network=net, device_name=target, config=config,
                                   num_requests=nr)
    return _Compiled(exec_net, net)
