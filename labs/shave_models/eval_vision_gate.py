"""Device evaluation for the vision-gate bundle (vision-gate/1).

MO FP16 -> IECore MYRIAD over the held-out frames; reports accuracy,
trigger recall, fire-rate on non-trigger, and sync batch-1 p50 latency
against vision_gate_spec.ACCEPTANCE. Honest caveat recorded: the diagnostic
held-out set is BBB-cartoon vs COCO-photo; the sealed set must add unseen
animations (other studios) before any generalization claim.

Usage (NCS1 Python 3.7 venv)::
    python eval_vision_gate.py bundles/vision-gate --fixtures D:/models/vision-gate
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))

from vision_gate_spec import ACCEPTANCE, TASK_SPEC_VERSION  # noqa: E402


def frame_to_nchw(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return ((arr - 0.5) / 0.5).transpose(2, 0, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--fixtures", type=Path, default=Path("D:/models/vision-gate"))
    ap.add_argument("--toolchain", type=Path, default=Path("D:/NTVStudy/tools/ncs1"))
    ap.add_argument("--device", choices=["myriad", "cpu"], default="myriad")
    ap.add_argument("--run-dir", type=Path, default="")
    ap.add_argument("--manifest", default="manifest.jsonl",
                    help="manifest file name (manifest-sealed.jsonl for the sealed set)")
    ap.add_argument("--split", default="test",
                    help="rows to evaluate (test for diagnostics, sealed for sealed manifests)")
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
                        "--output_dir", str(ir)], env=env, check=True, timeout=300)

    core = IECore()
    net = core.read_network(model=str(xml), weights=str(binary))
    input_name = list(net.input_info if hasattr(net, "input_info") else net.inputs)[0]
    start = time.perf_counter()
    executable = core.load_network(network=net,
                                    device_name="MYRIAD" if args.device == "myriad" else "CPU",
                                    num_requests=1)
    load_ms = (time.perf_counter() - start) * 1000

    all_rows = [json.loads(line) for line
                in (args.fixtures / args.manifest).read_text(encoding="utf-8").splitlines()]
    rows = [r for r in all_rows if r["split"] == args.split]
    print(f"{len(rows)} held-out frames, device {args.device}", flush=True)

    elapsed, results = [], []
    correct = trig_correct = trig_total = non_fired = non_total = 0
    finite = True
    for row in rows:
        x = frame_to_nchw(args.fixtures / row["wav"])[None]
        t0 = time.perf_counter()
        out = executable.infer({input_name: x})
        elapsed.append((time.perf_counter() - t0) * 1000)
        logits = np.asarray(list(out.values())[0]).reshape(-1)
        finite = finite and bool(np.isfinite(logits).all())
        pred = int(np.argmax(logits))
        is_trigger = row["label"] == "trigger"
        correct += int(pred == (1 if is_trigger else 0))
        if is_trigger:
            trig_total += 1
            trig_correct += int(pred == 1)
        else:
            non_total += 1
            non_fired += int(pred == 1)
        results.append({"id": row["id"], "kind": row["kind"], "label": row["label"],
                        "pred": "trigger" if pred == 1 else "non-trigger"})

    elapsed = np.asarray(elapsed)
    accuracy = correct / max(1, len(rows))
    recall = trig_correct / max(1, trig_total)
    fire = non_fired / max(1, non_total)
    metrics = {
        "device": "MYRIAD" if args.device == "myriad" else "CPU",
        "ncs1_runtime": "OpenVINO 2020.3.2",
        "spec_version": TASK_SPEC_VERSION,
        "frames": len(rows),
        "accuracy": round(accuracy, 4), "trigger_recall": round(recall, 4),
        "fire_rate_nontrigger": round(fire, 4),
        "p50_ms": round(float(np.percentile(elapsed, 50)), 1),
        "p95_ms": round(float(np.percentile(elapsed, 95)), 1),
        "load_ms": round(load_ms, 1), "all_finite": finite,
        "model_digests": {"onnx_sha256": info["onnx_sha256"],
                          "ir_xml_sha256": hashlib.sha256(xml.read_bytes()).hexdigest()},
        "caveat": "diagnostic held-out = BBB-cartoon vs COCO-photo; the sealed set must "
                  "add unseen animations (other studios) before any generalization claim",
    }
    metrics["acceptance"] = {
        "accuracy_min": {"required": ACCEPTANCE["accuracy_min"], "measured": round(accuracy, 4),
                         "pass": bool(accuracy >= ACCEPTANCE["accuracy_min"])},
        "fire_rate_nontrigger_max": {"required": ACCEPTANCE["fire_rate_on_nontrigger_max"],
                                     "measured": round(fire, 4),
                                     "pass": bool(fire <= ACCEPTANCE["fire_rate_on_nontrigger_max"])},
        "recall_on_trigger_min": {"required": ACCEPTANCE["recall_on_trigger_min"],
                                  "measured": round(recall, 4),
                                  "pass": bool(recall >= ACCEPTANCE["recall_on_trigger_min"])},
        "p50_ms_max": {"required": ACCEPTANCE["p50_ms_max"],
                       "measured": round(float(np.percentile(elapsed, 50)), 1),
                       "pass": bool(float(np.percentile(elapsed, 50)) <= ACCEPTANCE["p50_ms_max"])},
    }
    run_dir = (args.run_dir if args.run_dir else args.bundle).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    report = run_dir / f"device-evaluation-{args.device}.json"
    report.write_text(json.dumps({"metrics": metrics, "results": results}, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"written -> {report}")


if __name__ == "__main__":
    main()
