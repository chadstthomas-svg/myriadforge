"""Device evaluation for the voicetools bundle (voicetools/1 acceptance run).

Runs the trained multi-head model on the MA2450 (MO FP16 -> IECore MYRIAD,
single request) over the frozen fixtures with the v1 decision rule:
score every window (stride-aligned + end-aligned), fire a tool when its
argmax is not the reject class and its softmax confidence >= 0.5, take slot
values from the firing window, and read ordered calls by first occurrence in
window order. Reports full-call exact match, false-action rate on negatives,
and per-window latency (host framing + device infer) against the acceptance
gate in tools_spec.ACCEPTANCE.

Usage (NCS1 Python 3.7 venv)::
    python eval_voicetools.py bundles/voicetools --fixtures D:/models/ncs-shave-research/voicetools
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

from tools_spec import (ACCEPTANCE, TASK_SPEC_VERSION,  # noqa: E402
                        WINDOW_SECONDS, WINDOW_STRIDE)

RATE = 16000
WIN_FRAMES = 200
HOP = 160
NFFT = 512
FIRE_THRESHOLD = 0.5


def window_starts(seconds):
    """Mirror of train_voicetools.window_starts (torch-free copy for the 3.7 venv)."""
    if seconds <= WINDOW_SECONDS:
        return [0.0]
    aligned = [w * WINDOW_STRIDE for w in range(int((seconds - WINDOW_SECONDS) / WINDOW_STRIDE) + 1)]
    aligned.append(max(0.0, seconds - WINDOW_SECONDS))
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
        assert src.getframerate() == RATE and src.getnchannels() == 1 and src.getsampwidth() == 2
        return np.frombuffer(src.readframes(src.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def decide(window_out, info):
    """One window's head outputs -> (fired_tool_or_None, call_dict_or_None, confidence)."""
    labels = info["labels"]
    tool_idx = int(np.argmax(window_out["tool"]))
    conf = float(softmax(window_out["tool"])[tool_idx])
    if tool_idx == labels["none_idx"]["tool"] or conf < FIRE_THRESHOLD:
        return None, None, conf
    tool = labels["tools"][tool_idx]
    call = {"tool": tool}
    slot_of = {"room": "room", "device": "device", "scene": "scene",
               "level": "level", "seconds": "seconds"}
    for head, key in slot_of.items():
        idx = int(np.argmax(window_out[head]))
        if idx != labels["none_idx"][head]:
            value = labels["enums"][head][idx]
            call[key] = int(value) if head in ("level", "seconds") else str(value)
    return tool, call, conf


def normalize(call):
    args = {}
    for key, value in call.items():
        if key == "tool":
            continue
        args[key] = int(value) if key in ("level", "seconds") else str(value).casefold()
    return {"name": call["tool"], "args": args}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--fixtures", type=Path, default=Path("D:/models/ncs-shave-research/voicetools"))
    ap.add_argument("--toolchain", type=Path, default=Path("D:/NTVStudy/tools/ncs1"))
    ap.add_argument("--split", default="test")
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
    executable = core.load_network(network=net, device_name="MYRIAD", num_requests=1)
    load_ms = (time.perf_counter() - start) * 1000
    out_names = [str(o) for o in net.outputs]

    rows = [json.loads(line) for line
            in (args.fixtures / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row["split"] == args.split]
    print(f"{len(rows)} fixtures (split {args.split}), outputs {out_names}", flush=True)

    elapsed, results = [], []
    false_on_neg = hits = positives = 0
    for row in rows:
        samples = read_wav(args.fixtures / row["wav"])
        got_calls, fired_seen = [], set()
        t_start = time.perf_counter()
        for w_start in window_starts(row["seconds"]):
            seg = samples[int(w_start * RATE): int(w_start * RATE) + (WIN_FRAMES - 1) * HOP + 400]
            if len(seg) < 400:
                continue
            outs = executable.infer({input_name: logmel(seg)})
            # MO renames blobs to /<head>/Gemm - canonicalize by head substring
            heads = ("tool", "room", "device", "scene", "level", "seconds")
            window_out = {}
            for name, tensor in outs.items():
                match = next((h for h in heads if f"/{h}/" in name or name == h), None)
                if match is None:
                    raise KeyError(f"cannot map output blob {name!r} to a head")
                window_out[match] = np.asarray(tensor).reshape(-1)
            tool, call, _conf = decide(window_out, info)
            if tool is not None and tool not in fired_seen:
                fired_seen.add(tool)
                got_calls.append(call)
        elapsed.append((time.perf_counter() - t_start) * 1000)
        want = [normalize(c) for c in row["calls"]]
        got = [normalize(c) for c in got_calls]
        exact = got == want
        if row["kind"].startswith("neg"):
            if got:
                false_on_neg += 1
        else:
            positives += 1
            hits += int(exact)
        results.append({"id": row["id"], "kind": row["kind"], "got": got,
                        "want": want, "exact": exact})

    elapsed = np.asarray(elapsed)
    metrics = {
        "exact_match": round(hits / max(1, positives), 4),
        "exact_hits": hits, "positives": positives,
        "false_action_rate": round(false_on_neg / max(1, len(rows) - positives), 4),
        "negatives": len(rows) - positives, "false_actions": false_on_neg,
        "p50_end_to_end_ms": round(float(np.percentile(elapsed, 50)), 1),
        "p95_end_to_end_ms": round(float(np.percentile(elapsed, 95)), 1),
        "load_ms": round(load_ms, 1),
        "device": "MYRIAD", "ncs1_runtime": "OpenVINO 2020.3.2",
        "spec_version": TASK_SPEC_VERSION,
        "onnx_sha256": info["onnx_sha256"],
    }
    metrics["acceptance"] = {
        gate: {"required": required, "measured": metrics[measured_key],
               "pass": bool(metrics[measured_key] >= required if key == "exact_match_min"
                            else metrics[measured_key] <= required)}
        for gate, (required, key, measured_key) in {
            "exact_match_min": (ACCEPTANCE["exact_match_min"], "exact_match_min", "exact_match"),
            "false_action_max": (ACCEPTANCE["false_action_max"], "false_action_max", "false_action_rate"),
            "p95_ms_max": (ACCEPTANCE["p95_ms_max"], "p95_ms_max", "p95_end_to_end_ms"),
        }.items()
    }
    (args.bundle / "device-evaluation.json").write_text(
        json.dumps({"metrics": metrics, "results": results}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"written -> {args.bundle / 'device-evaluation.json'}")


if __name__ == "__main__":
    main()
