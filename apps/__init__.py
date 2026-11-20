"""NCS2 application pilots (NCS2_ROADMAP WS5).

The 2026 applications of the Neural Compute Stick: decision-model gate,
footage triage, podcast automation, kNN-as-MatMul engine, and the
always-on-gate -> host-brain cascade.

Environment boundary: apps run INSIDE the isolated OpenVINO 2022.3.1 venv
(``tools/ncs2/.venv``, Python 3.10) - the myriadforge main env never imports
OpenVINO. Run them with::

    tools\\ncs2\\.venv\\Scripts\\python.exe -m apps.<app>.<module> ...

from the repo root. myriadforge stdlib-only modules (bench.schema) are importable
via the sys.path bootstrap in ``apps.common``.
"""
