"""Device evaluation for the VAD suite (vad/1): window classifier + event decode.

Single [1,3] head over the same [1,1,40,200] windows; per-window argmax with a
tunable speech-fire threshold (abstention calibration lane - the v0 model's
0.28% false-wake rate converts to ~10 wakes/hour at 1 s hop vs the <=1/hour
gate, so the threshold sweep here is the honest path to that gate). Reports
window accuracy, speech recall, false wakes/hour on the sealed non-speech set,
and per-window latency against vad_spec.ACCEPTANCE.

Usage (NCS1 Python 3.7 venv)::
    python eval_vad.py bundles/vad --fixtures D:/models/ncs-shave-research/vad \
        [--device cpu] [--threshold 0.9] [--sweep]
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

from vad_spec import ACCEPTANCE, CLASS_IDX, CLASSES, SAMPLE_RATE, TASK_SPEC_VERSION  # noqa: E402

RATE = SAMPLE_RATE
WIN_FRAMES = 200
HOP = 160
NFFT = 512


def window_starts(seconds):
    if seconds <= 2.0:
        return [0.0]
    aligned = [w * 1.0 for w in range(int((seconds - 2.0) / 1.0) + 1)]
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
    ap.add_argument("--device", choices=["myriad", "cpu"], default="myriad")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="min softmax speech confidence to call a window speech")
    ap.add_argument("--sweep", action="store_true",
                    help="threshold sweep 0.5..0.95 for the abstention-calibration lane")
    ap.add_argument("--run-dir", type=Path, default="")
    args = ap.parse_args()

    sys.path.insert(0, str(args.toolchain.resolve()))
    import ncs1_paths
    ncs1_paths.setup()
    from openvino.inference_engine import IECore

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

    core = IECore()
    net = core.read_network(model=str(xml), weights=str(binary))
    input_name = list(net.input_info if hasattr(net, "input_info") else net.inputs)[0]
    start = time.perf_counter()
    executable = core.load_network(network=net,
                                    device_name="MYRIAD" if args.device == "myriad" else "CPU",
                                    num_requests=1)
    load_ms = (time.perf_counter() - start) * 1000

    rows = [json.loads(line) for line
            in (args.fixtures / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    # Precompute window features + true labels once; thresholds rescore instantly.
    feats, labels, elapsed = [], [], []
    for row in rows:
        wav = Path(row["wav"])
        if not wav.is_absolute():
            wav = args.fixtures / wav
        samples = read_wav(wav)
        for w_start in window_starts(len(samples) / RATE):
            seg = samples[int(w_start * RATE): int(w_start * RATE) + (WIN_FRAMES - 1) * HOP + 400]
            if len(seg) < 400:
                continue
            t0 = time.perf_counter()
            out = executable.infer({input_name: logmel(seg)})
            elapsed.append((time.perf_counter() - t0) * 1000)
            feats.append(np.asarray(list(out.values())[0]).reshape(-1))
            labels.append(CLASS_IDX[row["label"]])
    feats = np.stack(feats)
    labels = np.asarray(labels)
    elapsed = np.asarray(elapsed)
    print(f"{len(feats)} windows, device {args.device}, load {load_ms:.0f} ms", flush=True)

    def score(threshold):
        exp = np.exp(feats - feats.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        pred = probs.argmax(1)
        speech_col = CLASS_IDX["speech"]
        called_speech = (pred == speech_col) & (probs[:, speech_col] >= threshold)
        acc = float((pred == labels).mean())
        speech = labels == speech_col
        recall = float(called_speech[speech].mean())
        nonspeech_rate = float(called_speech[~speech].mean())
        return {"threshold": threshold, "window_accuracy": round(acc, 4),
                "speech_recall": round(recall, 4),
                "false_wake_rate": round(nonspeech_rate, 5),
                "false_wakes_per_hour": round(nonspeech_rate * 3600, 1)}

    sweep = [score(t) for t in (0.5, 0.7, 0.8, 0.9, 0.95)] if args.sweep else [score(args.threshold)]
    metrics = {
        "device": "MYRIAD" if args.device == "myriad" else "CPU",
        "ncs1_runtime": "OpenVINO 2020.3.2",
        "spec_version": TASK_SPEC_VERSION,
        "windows": int(len(feats)),
        "model_digests": {"onnx_sha256": info["onnx_sha256"],
                          "ir_xml_sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
                          "ir_bin_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()},
        "timing_boundary": "per-window device inference only (host framing excluded here; "
                           "the voicetools eval covers the framed end-to-end path)",
        "p50_window_ms": round(float(np.percentile(elapsed, 50)), 2),
        "p95_window_ms": round(float(np.percentile(elapsed, 95)), 2),
        "load_ms": round(load_ms, 1),
        "sweep": sweep,
    }
    run_dir = (args.run_dir if args.run_dir else args.bundle).resolve()
    report = run_dir / f"device-evaluation-{args.device}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"metrics": metrics}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"written -> {report}")


if __name__ == "__main__":
    main()
