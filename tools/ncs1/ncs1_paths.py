"""Canonical OpenVINO 2020.3.2 path wiring for the NCS1 lane (Python 3.7).

The 2020.3 Windows runtime is not pip-installed; it lives in the extracted
archive under tools/ncs1/ov2020. IECore's ie_api.pyd (cp37) needs the IE
DLLs + TBB + ngraph + the Intel C++ runtime (libmmd/svml_dispmd, from the
intel-cmplr-lib-rt wheel) on PATH, and the python3.7 API + Model Optimizer
on sys.path. This mirrors bin/setupvars.bat. Call setup() before importing
openvino.inference_engine.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

NCS1_DIR = Path(__file__).resolve().parent
OV2020 = NCS1_DIR / "ov2020"
IE = OV2020 / "deployment_tools" / "inference_engine"
RELEASE = IE / "bin" / "intel64" / "Release"
MO = OV2020 / "deployment_tools" / "model_optimizer"
PYAPI = OV2020 / "python" / "python3.7"
REDIST = NCS1_DIR / "redist"


def setup() -> None:
    dirs = [
        REDIST,
        IE / "external" / "tbb" / "bin",
        RELEASE,
        IE / "bin" / "intel64" / "Debug",
        IE / "external" / "hddl" / "bin",
        OV2020 / "deployment_tools" / "ngraph" / "lib",
    ]
    os.environ["PATH"] = os.pathsep.join(str(d) for d in dirs) + os.pathsep + os.environ.get("PATH", "")
    for p in (PYAPI, OV2020 / "python" / "python3", MO):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
