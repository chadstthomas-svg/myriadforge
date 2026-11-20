"""Provider tests: Myriad discovery, MYRIAD.SIM gates, latency model, lane no-ops."""

from __future__ import annotations

from pathlib import Path

from myriadforge.devices.registry import benchable, default_providers, discover
from myriadforge.devices.sim_myriad import SimMyriadProvider, _latency_model


def test_discover_includes_myriad_sims() -> None:
    handles, warnings = discover()
    ids = {h.id for h in handles}
    assert "myriad-sim" in ids
    assert "myriad2-sim" in ids
    assert warnings == []  # no lane should blow up on a standard machine


def test_discover_fail_soft_with_broken_provider() -> None:
    class Boom:
        name = "boom"

        def scan(self):  # noqa: ANN201
            raise RuntimeError("exploded")

        def bench(self, *a, **k):  # noqa: ANN002, ANN003
            raise NotImplementedError

    handles, warnings = discover([Boom(), *default_providers()])  # type: ignore[list-item]
    assert any("boom" in w for w in warnings)
    assert handles  # the other lanes still came through


def test_myriad_sim_latency_model_scales_with_shaves() -> None:
    fast = _latency_model(10.0, 16)
    slow = _latency_model(10.0, 4)
    assert fast["p50"] < slow["p50"]
    assert fast["p90"] > fast["p50"]


def test_myriad_sim_rejects_tflite(tmp_path: Path) -> None:
    model = tmp_path / "m.tflite"
    model.write_bytes(b"\x00" * 1024)
    prov = SimMyriadProvider()
    handle = prov.scan()[0]
    out = prov.bench(handle, model, warmup_runs=1, measured_runs=2,
                     out_dir=tmp_path / "res")
    import json

    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["status"] == "error"
    assert "gate" in rec["error"]
    assert rec["backend"]["simulated"] is True


def test_myriad_sim_accepts_onnx(tmp_path: Path) -> None:
    model = tmp_path / "m.onnx"
    model.write_bytes(b"\x00" * (2 * 1024 * 1024))
    prov = SimMyriadProvider({"shave_count": 8})
    handle = prov.scan()[0]
    out = prov.bench(handle, model, warmup_runs=1, measured_runs=2,
                     out_dir=tmp_path / "res")
    import json

    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["status"] == "ok"
    assert rec["backend"]["shave_count"] == 8
    ss = rec["metrics"]["steady_state_ms"]
    assert ss["p50"] > 0 and ss["p90"] >= ss["p50"]


def test_ncs2_provider_graceful_without_venv(tmp_path: Path) -> None:
    prov = SimMyriadProvider  # keep import-time cost honest
    from myriadforge.devices.ncs2 import Ncs2Provider

    ncs2 = Ncs2Provider({"python": str(tmp_path / "missing" / "python.exe")})
    handles = ncs2.scan()
    assert len(handles) == 1
    assert handles[0].available is False
    assert "not found" in handles[0].label


def test_benchable_filter() -> None:
    handles, _ = discover()
    bench = benchable(handles)
    assert all(h.available and h.bench_mode != "none" for h in bench)
    ids = {h.id for h in bench}
    assert "myriad-sim" in ids and "myriad2-sim" in ids


def test_ncs1_provider_graceful_without_venv(tmp_path: Path) -> None:
    from myriadforge.devices.ncs1 import Ncs1Provider

    ncs1 = Ncs1Provider({"python": str(tmp_path / "missing" / "python.exe")})
    handles = ncs1.scan()
    assert len(handles) == 1
    assert handles[0].available is False
    assert "not found" in handles[0].label


def test_myriad2_sim_profile(tmp_path: Path) -> None:
    prov = SimMyriadProvider({"chip": "ma2450"})
    handle = prov.scan()[0]
    assert handle.id == "myriad2-sim"
    assert handle.fingerprint["chip"] == "ma2450-emulated"
    assert handle.fingerprint["shave_count"] == 12  # Myriad 2: fixed, no sweep

    model = tmp_path / "m.onnx"
    model.write_bytes(b"\x00" * (2 * 1024 * 1024))
    out = prov.bench(handle, model, warmup_runs=1, measured_runs=2,
                     out_dir=tmp_path / "res")
    import json

    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["status"] == "ok"
    assert rec["backend"]["simulated"] is True
    assert rec["backend"]["shave_count"] == 12


def test_myriad2_sim_budget_gate(tmp_path: Path) -> None:
    prov = SimMyriadProvider({"chip": "ma2450"})  # ~0.4 GB usable
    handle = prov.scan()[0]
    model = tmp_path / "big.onnx"
    with model.open("wb") as fh:  # sparse 500 MB > 0.4 GB; sim reads size only
        fh.truncate(500 * 1024 * 1024)
    out = prov.bench(handle, model, warmup_runs=1, measured_runs=2,
                     out_dir=tmp_path / "res")
    import json

    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["status"] == "error"
    assert "budget" in rec["error"]


def test_discover_includes_myriad2_sim() -> None:
    handles, warnings = discover()
    ids = {h.id for h in handles}
    assert "myriad2-sim" in ids
    assert warnings == []
