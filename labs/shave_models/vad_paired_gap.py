"""VAD paired-gap latency check (vad/1 acceptance): the compute_probe method.

Loads TWO graphs in ONE IECore session - a zero-compute gap model with the
identical [1,1,40,200] input (GlobalAveragePool -> 1-value output: it pays the
same 32 KB input transfer + dispatch and no math) and the vad bundle - and
infers the SAME window set back-to-back. The session-relative gates
(vad_spec.ACCEPTANCE): vad p95 <= 12 ms absolute AND vad p95 <= 2x gap p95.
Cancels host-USB state entirely.

Usage (NCS1 Python 3.7 venv)::
    python vad_paired_gap.py bundles/vad --fixtures D:/models/ncs-shave-research/vad
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))

from vad_spec import ACCEPTANCE, CLASS_IDX, SAMPLE_RATE, WINDOW_STRIDE  # noqa: E402

RATE = SAMPLE_RATE
WIN_FRAMES = 200
HOP = 160
NFFT = 512
MAX_WINDOWS = 600  # paired sample: enough percentiles, one session


def build_gap_onnx(path: Path):
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    nodes = [
        helper.make_node("GlobalAveragePool", ["features"], ["gap"]),
        helper.make_node("Reshape", ["gap", "shape1"], ["flat"], name="flat"),
        helper.make_node("Gemm", ["flat", "w", "b"], ["logits"], name="head"),
    ]
    graph = helper.make_graph(
        nodes, "vad-gap",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 1, 40, 200])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 1])],
        initializer=[
            numpy_helper.from_array(np.array([1, 1], dtype=np.int64), "shape1"),
            numpy_helper.from_array(np.ones((1, 1), dtype=np.float32), "w"),
            numpy_helper.from_array(np.zeros(1, dtype=np.float32), "b"),
        ])
    model = helper.make_model(graph, producer_name="vad-paired-gap",
                              opset_imports=[helper.make_opsetid("", 11)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def window_starts(seconds):
    if seconds <= 2.0:
        return [0.0]
    aligned = [w * WINDOW_STRIDE for w in range(int((seconds - 2.0) / 1.0) + 1)]
    aligned.append(max(0.0, seconds - 2.0))
    return sorted(set(round(s, 3) for s in aligned))


def mel_filters():
    from features import mel_filters as shared
    return shared()


def logmel(samples):
    need = (WIN_FRAMES - 1) * HOP + 400
    if len(samples) < need:
        samples = np.pad(samples, (0, need - len(samples)))
    offsets = np.arange(WIN_FRAMES)[:, None] * HOP + np.arange(400)
    frames = samples[offsets] * np.hanning(400)
    power = np.abs(np.fft.rfft(frames, n=NFFT, axis=1)) ** 2
    mel = np.log(np.maximum(power @ mel_filters(), 1e-8))
    return ((mel.T - mel.mean()) / max(mel.std(), 0.1)).astype(np.float32)[None]


def read_wav(path):
    with wave.open(str(path), "rb") as src:
        return np.frombuffer(src.readframes(src.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--fixtures", type=Path, default=Path("D:/models/ncs-shave-research/vad"))
    ap.add_argument("--toolchain", type=Path, default=Path("D:/NTVStudy/tools/ncs1"))
    args = ap.parse_args()
    args.bundle = args.bundle.resolve()

    sys.path.insert(0, str(args.toolchain.resolve()))
    import ncs1_paths
    ncs1_paths.setup()
    from openvino.inference_engine import IECore

    # gap graph: build once per session (zero-compute, same input shape)
    gap_onnx = args.bundle / "gap-model.onnx"
    build_gap_onnx(gap_onnx)

    core = IECore()
    # 2020.3 read_network needs IR, not raw ONNX - convert the tiny gap graph via MO
    gap_ir = args.bundle / "gap-ir"
    gap_ir.mkdir(parents=True, exist_ok=True)
    gap_xml, gap_bin = gap_ir / "gap-model.xml", gap_ir / "gap-model.bin"
    if not (gap_xml.is_file() and gap_bin.is_file()):
        env = dict(os.environ, PYTHONPATH=str(ncs1_paths.MO))
        subprocess.run([sys.executable, str(ncs1_paths.MO / "mo_onnx.py"),
                        "--input_model", str(gap_onnx), "--data_type", "FP16",
                        "--output_dir", str(gap_ir)], env=env, check=True, timeout=120)
    gap_net = core.read_network(model=str(gap_xml), weights=str(gap_bin))
    gap_exe = core.load_network(network=gap_net, device_name="MYRIAD", num_requests=1)

    info = json.loads((args.bundle / "model.json").read_text(encoding="utf-8"))
    source = args.bundle / "model.onnx"
    if hashlib.sha256(source.read_bytes()).hexdigest() != info["onnx_sha256"]:
        raise ValueError("Model digest differs from the manifest")
    ir = args.bundle / "ir" / info["onnx_sha256"]
    ir.mkdir(parents=True, exist_ok=True)
    xml, binary = ir / "model.xml", ir / "model.bin"
    if not (xml.is_file() and binary.is_file()):
        env = dict(os.environ, PYTHONPATH=str(ncs1_paths.MO))
        subprocess.run([sys.executable, str(ncs1_paths.MO / "mo_onnx.py"),
                        "--input_model", str(source), "--data_type", "FP16",
                        "--output_dir", str(ir)], env=env, check=True, timeout=180)
    vad_net = core.read_network(model=str(xml), weights=str(binary))
    vad_in = list(vad_net.input_info if hasattr(vad_net, "input_info") else vad_net.inputs)[0]
    vad_exe = core.load_network(network=vad_net, device_name="MYRIAD", num_requests=1)

    # one window set, both graphs, same session
    rows = [json.loads(line) for line
            in (args.fixtures / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    windows = []
    for row in rows:
        wav = Path(row["wav"])
        if not wav.is_absolute():
            wav = args.fixtures / wav
        samples = read_wav(wav)
        for w_start in window_starts(len(samples) / RATE):
            seg = samples[int(w_start * RATE): int(w_start * RATE) + (WIN_FRAMES - 1) * HOP + 400]
            if len(seg) >= 400:
                windows.append(logmel(seg))
        if len(windows) >= MAX_WINDOWS:
            break
    windows = windows[:MAX_WINDOWS]
    print(f"paired sample: {len(windows)} windows, one session, both graphs resident", flush=True)

    gap_in = list(gap_net.input_info if hasattr(gap_net, "input_info") else gap_net.inputs)[0]
    for _ in range(5):
        gap_exe.infer({gap_in: windows[0]})
        vad_exe.infer({vad_in: windows[0]})

    gap_t, vad_t = [], []
    for w in windows:
        t0 = time.perf_counter_ns()
        gap_exe.infer({gap_in: w})
        gap_t.append((time.perf_counter_ns() - t0) / 1e6)
        t0 = time.perf_counter_ns()
        vad_exe.infer({vad_in: w})
        vad_t.append((time.perf_counter_ns() - t0) / 1e6)

    def pct(vals, q):
        return round(float(np.percentile(vals, q)), 2)

    ratio_p50 = round(vad_t and float(np.percentile(vad_t, 50)) / max(float(np.percentile(gap_t, 50)), 1e-9), 2)
    ratio_p95 = round(float(np.percentile(vad_t, 95)) / max(float(np.percentile(gap_t, 95)), 1e-9), 2)
    verdict = {
        "session": "single IECore session, both graphs resident, identical windows",
        "windows": len(windows),
        "gap_p50_ms": pct(gap_t, 50), "gap_p95_ms": pct(gap_t, 95),
        "vad_p50_ms": pct(vad_t, 50), "vad_p95_ms": pct(vad_t, 95),
        "ratio_p50": ratio_p50, "ratio_p95": ratio_p95,
        "gate_absolute": {"required": ACCEPTANCE["p95_window_ms_max"],
                          "measured": pct(vad_t, 95),
                          "pass": bool(pct(vad_t, 95) <= ACCEPTANCE["p95_window_ms_max"])},
        "gate_paired_gap": {"required": ACCEPTANCE["paired_gap_ratio_max"],
                            "measured": ratio_p95,
                            "pass": bool(ratio_p95 <= ACCEPTANCE["paired_gap_ratio_max"])},
        "interpretation": "the ratio isolates on-stick compute from transfer+dispatch: "
                          "ratio ~1 means the vad graph costs no more than moving the "
                          "input across USB; higher = real compute cost",
    }
    out = args.bundle / "paired-gap.json"
    out.write_text(json.dumps(verdict, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verdict, indent=2))
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
