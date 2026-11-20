"""Report tests: collect + render smoke over Myriad sim + demo records."""

from __future__ import annotations

from pathlib import Path

from myriadforge.bench.schema import make_demo_record, save_demo
from myriadforge.devices.sim_myriad import SimMyriadProvider
from myriadforge.report import collect, render


def _mini_onnx(tmp_path: Path) -> Path:
    p = tmp_path / "mini.onnx"
    p.write_bytes(b"\x00" * (512 * 1024))
    return p


def test_sim_bench_and_report(tmp_path: Path) -> None:
    model = _mini_onnx(tmp_path)
    prov = SimMyriadProvider()
    handle = prov.scan()[0]
    prov.bench(handle, model, warmup_runs=1, measured_runs=2, out_dir=tmp_path / "results")

    rec = make_demo_record(
        demo="knn",
        model={"name": "knn-test", "file": "-", "path": "-", "format": "onnx"},
        device=handle.result_device(),
        backend={"name": "openvino-myriad", "runtime": "MYRIAD.SIM",
                 "accelerator": "myriad", "simulated": True},
        config={}, metrics={"search_ms": 10.0},
        outputs={"latency_chain": {"search": 10.0}, "top": [{"id": 3, "score": 0.9}]},
    )
    save_demo(rec, tmp_path / "results")

    records = collect.load_results(tmp_path / "results")
    assert len(records) >= 2
    assert collect.group_bench(records)
    assert collect.group_demos(records)["knn"]

    html = render.render_html(records)
    assert "MyriadForge report" in html
    assert "SIM" in html              # simulated tag present
    assert "<svg" in html             # charts rendered
    assert "knn" in html              # demo view present
    assert "http://" not in html and "https://" not in html  # zero external assets
