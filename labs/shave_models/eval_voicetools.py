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

from tools_spec import (ACCEPTANCE, FIRE_THRESHOLD, TASK_SPEC_VERSION,  # noqa: E402
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


SLOT_OF = {"room": "room", "device": "device", "scene": "scene",
           "level": "level", "seconds": "seconds"}


def call_from_window(window_out, info):
    """Slot heads of one window -> call dict (no tool decision here)."""
    labels = info["labels"]
    call = {}
    for head, key in SLOT_OF.items():
        idx = int(np.argmax(window_out[head]))
        if idx != labels["none_idx"][head]:
            value = labels["enums"][head][idx]
            call[key] = int(value) if head in ("level", "seconds") else str(value)
    return call


def decide(window_out, info):
    """Per-window rule (v1/v2 record compatibility): argmax tool, confidence gate."""
    labels = info["labels"]
    probs = softmax(window_out["tool"])
    tool_idx = int(np.argmax(probs))
    conf = float(probs[tool_idx])
    if tool_idx == labels["none_idx"]["tool"] or conf < FIRE_THRESHOLD:
        return None, None, conf
    tool = labels["tools"][tool_idx]
    call = {"tool": tool, **call_from_window(window_out, info)}
    return tool, call, conf


def decide_utterance(window_outs, info, threshold):
    """Utterance-level rule: a tool fires at its BEST window (max tool confidence
    across the utterance, above the frozen threshold); calls are ordered by the
    window position of that best confidence, capped at 2 (the spec's max)."""
    labels = info["labels"]
    best = {}
    for wi, out in enumerate(window_outs):
        probs = softmax(out["tool"])
        tool_idx = int(np.argmax(probs))
        if tool_idx == labels["none_idx"]["tool"]:
            continue
        tool = labels["tools"][tool_idx]
        conf = float(probs[tool_idx])
        if tool not in best or conf > best[tool][0]:
            best[tool] = (conf, wi, {"tool": tool, **call_from_window(out, info)})
    fired = [(wi, -conf, call) for tool, (conf, wi, call) in best.items() if conf >= threshold]
    fired.sort(key=lambda x: (x[0], x[1]))
    return [call for _, _, call in fired[:2]]


def _window_rule_calls(window_outs, info):
    """Per-window rule over stored outputs (record compatibility path)."""
    got, seen = [], set()
    for out in window_outs:
        tool, call, _conf = decide(out, info)
        if tool is not None and tool not in seen:
            seen.add(tool)
            got.append(call)
    return got


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
    ap.add_argument("--manifest", default="manifest.jsonl",
                    help="manifest file name (manifest-sealed.jsonl for the sealed set)")
    ap.add_argument("--device", choices=["myriad", "cpu"], default="myriad",
                    help="cpu = matched same-student control on the OpenVINO 2020.3 CPU plugin")
    ap.add_argument("--run-dir", type=Path, default="",
                    help="isolated report directory (default: the bundle dir; reports are "
                         "named device-evaluation-<device>.json so prior reports survive)")
    ap.add_argument("--rule", choices=["utterance", "window"], default="utterance",
                    help="utterance = best-window-per-tool aggregation (default); "
                         "window = the v1/v2 per-first-occurrence rule")
    ap.add_argument("--threshold", type=float, default=None,
                    help="speech-tool fire threshold; default = tools_spec FIRE_THRESHOLD")
    ap.add_argument("--sweep", action="store_true",
                    help="calibration sweep over thresholds on this split (no freeze)")
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
    executable = core.load_network(network=net, device_name=args.device.upper()
                                   if args.device == "myriad" else "CPU", num_requests=1)
    load_ms = (time.perf_counter() - start) * 1000
    out_names = [str(o) for o in net.outputs]

    rows = [json.loads(line) for line
            in (args.fixtures / args.manifest).read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row["split"] == args.split]
    print(f"{len(rows)} fixtures (split {args.split}), device {args.device}, "
          f"outputs {out_names}", flush=True)

    elapsed, results = [], []
    false_on_neg = hits = positives = 0
    finiteness_checked = False
    for row in rows:
        samples = read_wav(args.fixtures / row["wav"])
        seconds = row.get("seconds") or len(samples) / RATE  # sealed rows omit seconds
        window_outs = []
        t_start = time.perf_counter()
        for w_start in window_starts(seconds):
            seg = samples[int(w_start * RATE): int(w_start * RATE) + (WIN_FRAMES - 1) * HOP + 400]
            if len(seg) < 400:
                continue
            feats = logmel(seg)
            outs = executable.infer({input_name: feats})
            # MO renames blobs to /<head>/Gemm - canonicalize by head substring
            heads = ("tool", "room", "device", "scene", "level", "seconds")
            window_out = {}
            for name, tensor in outs.items():
                match = next((h for h in heads if f"/{h}/" in name or name == h), None)
                if match is None:
                    raise KeyError(f"cannot map output blob {name!r} to a head")
                window_out[match] = np.asarray(tensor).reshape(-1)
            if not finiteness_checked:
                finiteness_checked = True
                if not all(np.isfinite(v).all() for v in window_out.values()):
                    raise ValueError("non-finite head output on the first inference")
            window_outs.append(window_out)
        elapsed.append((time.perf_counter() - t_start) * 1000)
        results.append({"id": row["id"], "kind": row["kind"], "window_outs": window_outs,
                        "want": [normalize(c) for c in row["calls"]]})

    elapsed = np.asarray(elapsed)
    negatives = sum(1 for row, _ in zip(rows, results) if row["kind"].startswith("neg"))
    positives = len(rows) - negatives

    def score(threshold):
        hits = false_on_neg = 0
        scored = []
        for row, res in zip(rows, results):
            got_calls = (decide_utterance(res["window_outs"], info, threshold)
                         if args.rule == "utterance"
                         else [c for c in _window_rule_calls(res["window_outs"], info)])
            got = [normalize(c) for c in got_calls]
            exact = got == res["want"]
            if row["kind"].startswith("neg"):
                false_on_neg += int(bool(got))
            else:
                hits += int(exact)
            scored.append({"id": row["id"], "kind": row["kind"], "got": got,
                           "want": res["want"], "exact": exact})
        m = {"rule": args.rule, "threshold": threshold,
             "exact_match": round(hits / max(1, positives), 4),
             "exact_hits": hits, "positives": positives,
             "false_action_rate": round(false_on_neg / max(1, negatives), 4),
             "negatives": negatives, "false_actions": false_on_neg,
             "p50_end_to_end_ms": round(float(np.percentile(elapsed, 50)), 1),
             "p95_end_to_end_ms": round(float(np.percentile(elapsed, 95)), 1),
             "load_ms": round(load_ms, 1),
             "device": "MYRIAD" if args.device == "myriad" else "CPU",
             "ncs1_runtime": "OpenVINO 2020.3.2",
             "spec_version": TASK_SPEC_VERSION,
             "model_digests": {"onnx_sha256": info["onnx_sha256"],
                               "ir_xml_sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
                               "ir_bin_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()},
             "timing_boundary": ("end-to-end per fixture = host logmel framing over every window + "
                                 "USB/device inference + decision; model load and WAV decode excluded"),
             "fixture_ids": "manifest-order replay (identical IDs across device/CPU runs)"}
        m["acceptance"] = {
            gate: {"required": required, "measured": m[measured_key],
                   "pass": bool(m[measured_key] >= required if key == "exact_match_min"
                                else m[measured_key] <= required)}
            for gate, (required, key, measured_key) in {
                "exact_match_min": (ACCEPTANCE["exact_match_min"], "exact_match_min", "exact_match"),
                "false_action_max": (ACCEPTANCE["false_action_max"], "false_action_max", "false_action_rate"),
                "p95_ms_max": (ACCEPTANCE["p95_ms_max"], "p95_ms_max", "p95_end_to_end_ms"),
            }.items()}
        m["acceptance"]["min_negatives"] = {
            "required": ACCEPTANCE["min_negatives"], "measured": negatives,
            "pass": bool(negatives >= ACCEPTANCE["min_negatives"]),
            "note": "the sealed acceptance set needs >=300 held-out negatives; "
                    "the current test split is a diagnostic set"}
        return m, scored

    run_dir = (args.run_dir if args.run_dir else args.bundle).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.sweep:
        sweep = []
        for t in (0.5, 0.6, 0.7, 0.8, 0.9):
            m, _ = score(t)
            sweep.append({k: m[k] for k in ("threshold", "exact_match", "false_action_rate",
                                            "p95_end_to_end_ms")})
            print(sweep[-1], flush=True)
        report = run_dir / f"calibration-{args.device}-{args.split}.json"
        report.write_text(json.dumps({"rule": args.rule, "sweep": sweep}, indent=2) + "\n",
                          encoding="utf-8")
        print(f"written -> {report}")
        return
    m, scored = score(args.threshold if args.threshold is not None else FIRE_THRESHOLD)
    report = run_dir / f"device-evaluation-{args.device}-{args.rule}.json"
    report.write_text(json.dumps({"metrics": m, "results": scored}, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(m, indent=2))
    print(f"written -> {report}")


if __name__ == "__main__":
    main()
