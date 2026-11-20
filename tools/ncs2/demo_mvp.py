"""NCS2/NCS1 roadmap MVP: one script, every pilot, one flag.

Runs the WS5 application pilots end to end and writes one summary report
(tools/ncs2/mvp_summary.json + printed table). The same script runs unchanged
on simulators and on real silicon, for either stick:

    --lane ncs2   Intel NCS2 / Myriad X  (OpenVINO 2022.3.1, Py3.10 venv)
    --lane ncs1   Intel NCS1 / Myriad 2  (OpenVINO 2020.3.2, Py3.7 venv)

Usage (from repo root, any shell)::

    tools\\ncs2\\.venv\\Scripts\\python.exe tools\\ncs2\\demo_mvp.py --lane ncs1 --device myriad --full
    tools\\ncs2\\.venv\\Scripts\\python.exe tools\\ncs2\\demo_mvp.py --lane ncs2 --device myriad
    tools\\ncs2\\.venv\\Scripts\\python.exe tools\\ncs2\\demo_mvp.py --device myriad-sim

--device myriad      : the stick (hardware mode)
--device cpu         : OpenVINO CPU plugin (functional truth, no dongle)
--device myriad-sim  : MYRIAD.SIM via the main myriadforge env (uv run myriadforge)

Components: bench (gate latency), knn (queries/sec), cascade (gate->Mirage),
and with --full: footage + podcast pipelines. Missing sample media are skipped
gracefully. The kNN table size defaults per lane (Myriad 2 hangs on very wide
GEMM outputs >=100k, so ncs1 uses a 10k table; ncs2 uses 100k).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

LANES = {
    "ncs2": {
        "venv": REPO_ROOT / "tools" / "ncs2" / ".venv" / "Scripts" / "python.exe",
        "bridge": REPO_ROOT / "tools" / "ncs2" / "bridge_runner.py",
        "model": REPO_ROOT / "models" / "ncs2" / "mobilenetv2-12.onnx",
        "model_key": "onnx-models/mobilenetv2-12",
        "knn_n": 100_000,
    },
    "ncs1": {
        "venv": REPO_ROOT / "tools" / "ncs1" / ".venv" / "Scripts" / "python.exe",
        "bridge": REPO_ROOT / "tools" / "ncs1" / "bridge_runner.py",
        "model": REPO_ROOT / "models" / "ncs1" / "mobilenetv2-7.onnx",
        "model_key": "onnx-models/mobilenetv2-7",
        "knn_n": 10_000,
    },
}

SAMPLE_IMAGE = REPO_ROOT / "results" / "sample_input.jpg"
SAMPLE_VIDEO = REPO_ROOT / "data" / "samples" / "bigbuckbunny_720_10s.mp4"
SAMPLE_WAV = REPO_ROOT / "data" / "samples" / "osr_us_000_0010_8k.wav"
SIM_DEVICES = {"myriad-sim", "myriad2-sim"}


def _run_native(python: Path, args: list[str]) -> tuple[int, str]:
    proc = subprocess.run([str(python), *args], capture_output=True, text=True,
                          cwd=str(REPO_ROOT), errors="replace", timeout=1800)
    return proc.returncode, (proc.stdout + proc.stderr)


def _run_main_env(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(["uv", "run", "myriadforge", *args], capture_output=True,
                          text=True, cwd=str(REPO_ROOT), errors="replace", timeout=1800)
    return proc.returncode, (proc.stdout + proc.stderr)


def _module_payload(raw: str) -> dict:
    """Extract the first JSON object a pilot prints.

    raw_decode tolerates both marker prefixes (NCS1RESULT:) and trailing
    IECore shutdown noise (BSL/XLink) after the JSON object.
    """
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[raw.index("{"):])
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Run every WS5 pilot, one summary report")
    ap.add_argument("--lane", choices=["ncs1", "ncs2"], default="ncs2")
    ap.add_argument("--device", default="myriad", help="myriad | cpu | myriad-sim | myriad2-sim")
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--knn-n", type=int, default=None, help="override kNN table size")
    ap.add_argument("--knn-batch", type=int, default=32)
    ap.add_argument("--full", action="store_true", help="also run footage + podcast pilots")
    ap.add_argument("--out", default=str(Path(__file__).parent / "mvp_summary.json"))
    cli = ap.parse_args()

    lane = LANES[cli.lane]
    venv_py, bridge, gate_model = lane["venv"], lane["bridge"], lane["model"]
    model_key = lane["model_key"]
    knn_n = cli.knn_n if cli.knn_n is not None else lane["knn_n"]
    device = cli.device
    sim = device in SIM_DEVICES

    report: dict = {"schema": "myriadforge.mvp-summary/1", "lane": cli.lane,
                    "device": device, "components": {}}

    # --- component 1: bench (gate model latency on the chosen lane) ---
    if sim:
        code, raw = _run_main_env(["bench", model_key, "--device", device])
    else:
        code, raw = _run_native(venv_py, [str(bridge), "--model", str(gate_model),
                                          "--device", device, "--warmup", "5",
                                          "--runs", "50", "--out", "results"])
    report["components"]["bench"] = {"status": "ok" if code == 0 else "error",
                                     "payload": _module_payload(raw) or {"raw": raw[-400:]}}

    # --- component 2: kNN engine ---
    if sim:
        report["components"]["knn"] = {"status": "skipped",
                                       "note": "kNN app needs a real OpenVINO device (sim is behavioral)"}
    else:
        code, raw = _run_native(venv_py, ["-m", "apps.knn_engine.engine", "--n", str(knn_n),
                                          "--batch", str(cli.knn_batch), "--queries", "128",
                                          "--device", device])
        report["components"]["knn"] = {"status": "ok" if code == 0 else "error",
                                       "payload": _module_payload(raw) or {"raw": raw[-400:]}}

    # --- component 3: cascade (gate -> Mirage) ---
    if SAMPLE_IMAGE.exists():
        if sim:
            report["components"]["cascade"] = {"status": "skipped",
                                               "note": "cascade app needs a real OpenVINO device"}
        else:
            code, raw = _run_native(venv_py, ["-m", "apps.cascade.run", str(SAMPLE_IMAGE),
                                              "--model", str(gate_model), "--device", device,
                                              "--threshold", str(cli.threshold)])
        report["components"]["cascade"] = {"status": "ok" if code == 0 else "error",
                                           "payload": _module_payload(raw) or {"raw": raw[-400:]}}
    else:
        report["components"]["cascade"] = {"status": "skipped", "note": f"no image at {SAMPLE_IMAGE}"}

    # --- components 4/5: footage + podcast pipelines (optional, native only) ---
    if cli.full and SAMPLE_VIDEO.exists() and not sim:
        code, raw = _run_native(venv_py, ["-m", "apps.footage.triage", str(SAMPLE_VIDEO),
                                          "--model", str(gate_model), "--device", device,
                                          "--fps", "2", "--threshold", str(min(cli.threshold, 0.10))])
        report["components"]["footage"] = {"status": "ok" if code == 0 else "error",
                                           "payload": _module_payload(raw) or {"raw": raw[-400:]}}
    elif cli.full:
        report["components"]["footage"] = {"status": "skipped", "note": "no sample video / sim lane"}

    if cli.full and SAMPLE_WAV.exists() and not sim:
        code, raw = _run_native(venv_py, ["-m", "apps.podcast.triage", str(SAMPLE_WAV)])
        report["components"]["podcast"] = {"status": "ok" if code == 0 else "error",
                                           "payload": _module_payload(raw) or {"raw": raw[-400:]}}
    elif cli.full:
        report["components"]["podcast"] = {"status": "skipped", "note": "no sample wav / sim lane"}

    out_path = Path(cli.out)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # --- printed summary table ---
    print(f"\n=== NCS MVP summary (lane={cli.lane}, device={device}) ===")
    for name, comp in report["components"].items():
        status = comp["status"].upper()
        payload = comp.get("payload", {})
        detail = ""
        if name == "bench" and payload.get("result"):
            try:
                rec = json.loads(Path(REPO_ROOT / payload["result"]).read_text(encoding="utf-8"))
                ss = rec["metrics"]["steady_state_ms"]
                detail = f"p50 {ss['p50']} ms, {rec['metrics']['throughput_fps']} inf/s"
            except Exception:
                pass
        elif name == "knn":
            bench = payload.get("summary", {}).get("bench", {})
            cap = payload.get("summary", {}).get("capacity", {})
            if bench:
                detail = (f"{bench.get('queries_per_sec')} qps "
                          f"({cap.get('table_mb_fp16')} MB table, batch {bench.get('batch')})")
        elif name == "cascade":
            chain = payload.get("chain", {})
            if chain:
                detail = (f"gate {chain.get('gate')} ms, total {chain.get('total')} ms, "
                          f"triggered: {chain.get('triggered')}")
        elif name in ("footage", "podcast"):
            summary = payload.get("summary", {})
            if summary:
                detail = (f"{summary.get('events_fired', '?')} events / "
                          f"{summary.get('frames_gated', summary.get('windows_gated', '?'))} gated")
        print(f"  {name:<10} {status:<7} {detail}")
    print(f"  report -> {out_path}")

    failed = [n for n, c in report["components"].items() if c["status"] == "error"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
