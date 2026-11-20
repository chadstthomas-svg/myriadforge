"""NCS2_ROADMAP WS5 decision-model gate: classifier + threshold + trigger contract.

The gate is the always-on first stage of every NCS2 application: cheap
static-shape CNN inference (MYRIAD on the stick, ~1 W, or the OpenVINO CPU
plugin when the dongle is absent) that fires an ``ncs2.event/1`` trigger
when a watched class crosses threshold. Downstream pilots
(footage/podcast/cascade) consume the same contract.

GateRunner lives in apps.gate.runner (kept out of __init__ so the module can
double as a `-m` CLI entry point without import warnings).
"""
