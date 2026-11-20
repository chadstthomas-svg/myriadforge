"""Shared runtime for the apps/ pilots (sys.path bootstrap, OV helpers, contracts).

Importing this package puts repo ``src/`` on sys.path so the stdlib-only
myriadforge modules (bench.schema) are importable inside the isolated OpenVINO
venv - the same trick tools/ncs2/bridge_runner.py uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from apps.common.events import IndexWriter, make_event  # noqa: E402,F401
from apps.common.ov import compile_model, core, ov_device, to_ir  # noqa: E402,F401
