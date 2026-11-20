"""Async-pipelined shim tests (hardware-free).

The ncs1 bridge's pipelined cadence logic and the compat layer's num_requests
surface are testable without the stick: _timed_pipelined/_stats are pure
Python (IECore is imported only inside bridge main()), and compile_model's
signature is inspectable. Real throughput gains are measured on hardware
(docs/NCS1_ROADMAP.md: sync ~16 inf/s -> 2-request pipeline ~23 inf/s).
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_bridge():
    path = REPO / "tools" / "ncs1" / "bridge_runner.py"
    spec = importlib.util.spec_from_file_location("ncs1_bridge_runner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeRequest:
    """Instant async request stand-in: exercises the pipeline loop logic."""

    def __init__(self) -> None:
        self.submitted = 0
        self.waited = 0

    def async_infer(self, inputs) -> None:
        self.submitted += 1

    def wait(self) -> None:
        self.waited += 1


def test_stats_shape() -> None:
    bridge = _load_bridge()
    s = bridge._stats([10.0, 12.0, 11.0, 13.0, 9.0])
    assert s["runs"] == 5.0
    assert s["min"] == 9.0 and s["max"] == 13.0
    assert s["p50"] > 0 and s["p90"] >= s["p50"]


def test_timed_pipelined_counts_and_cadence() -> None:
    bridge = _load_bridge()
    reqs = [_FakeRequest(), _FakeRequest()]
    measured = 12
    s = bridge._timed_pipelined(reqs, {"input": None}, warmup_runs=2, measured_runs=measured)
    assert s["runs"] == float(measured)
    assert s["p50"] >= 0.0
    # every measured completion plus warmup/prime/drain submits and waits
    assert all(r.submitted == r.waited for r in reqs)
    assert sum(r.submitted for r in reqs) >= measured


def test_compile_model_exposes_num_requests() -> None:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from apps.common.ov import compile_model

    params = inspect.signature(compile_model).parameters
    assert "num_requests" in params
    assert params["num_requests"].default == 1
