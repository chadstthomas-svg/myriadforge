"""NCS1 hardware-contact probe (OpenVINO 2020.3.2 IECore, Python 3.7).

Wires the extracted 2020.3.2 runtime onto PATH/PYTHONPATH and asks the
MYRIAD plugin what it sees. The Myriad 2 (NCS1, PID 2150) boot path is
enabled in 2020.3.x and disabled from 2020.4 onward.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "ov2020"
IE = ROOT / "deployment_tools" / "inference_engine"
RELEASE = IE / "bin" / "intel64" / "Release"
PYAPI = ROOT / "python" / "python3.7"

# Mirror bin/setupvars.bat: TBB + Release + Debug + hddl + ngraph on PATH.
_path_dirs = [
    Path(__file__).resolve().parent / "redist",  # Intel C++ runtime (libmmd/svml_dispmd)
    IE / "external" / "tbb" / "bin",
    RELEASE,
    IE / "bin" / "intel64" / "Debug",
    IE / "external" / "hddl" / "bin",
    ROOT / "deployment_tools" / "ngraph" / "lib",
]
os.environ["PATH"] = os.pathsep.join(str(p) for p in _path_dirs) + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, str(PYAPI))
sys.path.insert(0, str(ROOT / "python" / "python3"))

from openvino.inference_engine import IECore  # noqa: E402

ie = IECore()
print("available_devices:", ie.available_devices)
for metric in ("AVAILABLE_DEVICES", "DEVICE_THERMAL_LEVEL", "MYRIAD_DEVICE_TOTAL_MEM_SIZE",
               "MYRIAD_DEVICE_FREE_MEM_SIZE", "OPTIMIZATION_CAPABILITIES"):
    try:
        print(f"MYRIAD {metric}:", ie.get_metric("MYRIAD", metric))
    except Exception as exc:  # noqa: BLE001
        print(f"MYRIAD {metric}: <{type(exc).__name__}: {str(exc)[:120]}>")
