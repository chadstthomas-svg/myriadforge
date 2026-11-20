"""The product-defining cascade benchmark: gate-on-stick vs gate-on-CPU vs no-gate.

Same 60 frames, same Mirage brain, three ways of deciding when to invoke it:
  no-gate   Mirage.describe() on every frame
  gate-stick  MobileNetV2 gate on the NCS1 (persistent daemon); Mirage on hits only
  gate-cpu    the same gate on the OpenVINO CPU plugin; Mirage on hits only

Runs in the MAIN myriadforge env (Mirage lives there); spawns tools/ncs1/gate_daemon.py
inside the isolated OpenVINO venv for the gate. Threshold sweep produces the
invocation-rate curve. Energy is not measured (no power meter) - wall time and
host CPU-time are the proxies.

Usage (main env, repo root):
  python tools/ncs1/cascade_bench.py --frames data/samples/frames
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
VENV_PY = HERE.parent / "ncs1" / ".venv" / "Scripts" / "python.exe"
DAEMON = HERE / "gate_daemon.py"
THRESHOLDS = (0.1, 0.2, 0.3, 0.5)
GATE_MODEL = REPO / "models" / "ncs1" / "mobilenetv2-7.onnx"
GATE_LABELS = REPO / "models" / "ncs1" / "mobilenetv2-7.labels.txt"

sys.path.insert(0, str(REPO))


def _daemon_env() -> dict:
    """Clean PATH for the daemon: the orchestrator's uv-run PATH carries the main
    venv's DLLs, which break the ncs1 venv's cv2 import. The daemon wires its own
    OpenVINO dirs via ncs1_paths.setup()."""
    env = os.environ.copy()
    win = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    clean = [
        str(VENV_PY.parent),                    # ncs1 venv Scripts
        str(HERE.parent / "ncs1" / "python37"),  # base python37 dir
        str(win / "System32"),
        str(win),
    ]
    env["PATH"] = os.pathsep.join(clean)
    return env


def _read_json(proc, key: str) -> dict:
    """Read lines until one parses as JSON containing `key` (skips OpenVINO log noise)."""
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("gate daemon exited before replying")
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if key in obj:
            return obj


def run_gate(frames: list[Path], device: str, threshold: float, *, model: str) -> tuple[dict, list[dict]]:
    """Persistent daemon pass: one boot, one JSON reply per frame. Returns (ready, replies)."""
    proc = subprocess.Popen(
        [str(VENV_PY), str(DAEMON), "--device", device, "--threshold", str(threshold),
         "--model", model, "--labels", str(Path(model).with_suffix(".labels.txt"))],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        cwd=str(REPO), errors="replace", bufsize=1, env=_daemon_env(),
    )
    try:
        ready = _read_json(proc, "ready")
        replies: list[dict] = []
        t0 = time.perf_counter_ns()
        for frame in frames:
            proc.stdin.write(json.dumps({"frame": str(frame)}) + "\n")
            proc.stdin.flush()
            replies.append(_read_json(proc, "fired"))
        gate_wall_ms = (time.perf_counter_ns() - t0) / 1e6
    finally:
        try:
            proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        proc.wait(timeout=60)
    ready["gate_wall_ms"] = round(gate_wall_ms, 1)
    return ready, replies


def main() -> int:
    ap = argparse.ArgumentParser(description="cascade benchmark: gate vs Mirage brain")
    ap.add_argument("--frames", default="data/samples/frames")
    ap.add_argument("--thresholds", type=float, nargs="*", default=list(THRESHOLDS))
    ap.add_argument("--model", default=str(GATE_MODEL))
    ap.add_argument("--out", default="results/cascade_bench.json")
    args = ap.parse_args()

    frames = sorted(Path(args.frames).glob("frame_*.png"))
    if not frames:
        print(f"error: no frames under {args.frames} - run tools/ncs1/extract_frames.py first")
        return 2
    print(f"frames: {len(frames)} from {args.frames}", flush=True)

    import cactus  # noqa: PLC0415 - main env; first use downloads engine + weights

    # Warmup outside timing: first describe() may fetch/cache the engine.
    warm_t0 = time.perf_counter()
    brain = cactus.Mirage()
    brain.describe(str(frames[0]))
    print(f"mirage warmup (incl. any first-use download): {time.perf_counter() - warm_t0:.1f} s", flush=True)

    def brain_pass(paths: list[Path]) -> tuple[float, list[str]]:
        t0 = time.perf_counter()
        outs = []
        for p in paths:
            outs.append(brain.describe(str(p)))
        return (time.perf_counter() - t0) * 1000, outs

    configs = []
    total_frames = len(frames)

    # --- no gate: Mirage on every frame ---
    print("\n=== no-gate: Mirage on every frame ===", flush=True)
    t0 = time.perf_counter()
    mirage_ms, outs = brain_pass(frames)
    wall = (time.perf_counter() - t0) * 1000
    configs.append({"config": "no-gate", "threshold": None, "frames": total_frames,
                    "invocations": total_frames, "gate_ms": 0.0,
                    "mirage_ms": round(mirage_ms, 1), "wall_ms": round(wall, 1),
                    "per_frame_ms": round(wall / total_frames, 1)})
    print(f"  wall {wall/1000:.1f} s | mirage {mirage_ms/1000:.1f} s | {total_frames} invocations", flush=True)
    sample_text = outs[0] if outs else ""

    # --- gated configs ---
    for device in ("myriad", "cpu"):
        for thr in args.thresholds:
            label = f"gate-{device}@{thr}"
            print(f"\n=== {label} ===", flush=True)
            ready, replies = run_gate(frames, device, thr, model=args.model)
            gate_ms = ready.get("gate_wall_ms", 0.0)
            hits = [Path(r["frame"]) for r in replies if r.get("fired")]
            print(f"  gate: {ready.get('device')} | fired {len(hits)}/{total_frames} | gate wall {gate_ms:.0f} ms", flush=True)
            mouts = []
            if hits:
                mirage_ms, mouts = brain_pass(hits)
            else:
                mirage_ms = 0.0
            wall = gate_ms + mirage_ms
            configs.append({"config": label, "threshold": thr, "device": device,
                            "frames": total_frames, "invocations": len(hits),
                            "gate_ms": round(gate_ms, 1), "mirage_ms": round(mirage_ms, 1),
                            "wall_ms": round(wall, 1),
                            "per_frame_ms": round(wall / total_frames, 1),
                            "mirage_desc_sample": (json.dumps(mouts[0], default=str)[:200] if mouts else "")})
            print(f"  wall {wall/1000:.1f} s | mirage {mirage_ms/1000:.1f} s | {len(hits)} invocations", flush=True)

    out = {
        "schema": "myriad2.cascade-bench/1",
        "date": time.strftime("%Y-%m-%d"),
        "frames": total_frames,
        "gate_model": "mobilenetv2-7",
        "brain": "cactus.Mirage (22 MB, 2-bit)",
        "note": "wall = gate + Mirage, sequential; energy not measured (no power meter)",
        "configs": configs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== summary (per-frame wall cost, ms) ===", flush=True)
    for c in configs:
        thr = f"@{c['threshold']}" if c["threshold"] is not None else ""
        print(f"  {c['config']:<18} inv {c['invocations']:>3}/{total_frames}  wall/frame {c['per_frame_ms']:8.1f} ms", flush=True)
    print(f"written -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
