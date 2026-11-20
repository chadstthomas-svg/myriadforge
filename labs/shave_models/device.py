"""Convert, evaluate or query a fixed-shape bundle inside the NCS1 Python 3.7 environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import wave

import numpy as np

from features import audio_features, text_features
from grounding import encode_screen
from forms import ROLES, hit_test, pixels, points


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    sys.path.insert(0, str(args.toolchain.resolve()))
    import ncs1_paths
    ncs1_paths.setup()
    from openvino.inference_engine import IECore

    bundle = args.bundle.resolve()
    info = json.loads((bundle / "model.json").read_text(encoding="utf8"))
    source = bundle / "model.onnx"
    if digest(source) != info["onnx_sha256"]:
        raise ValueError("Model digest differs from the manifest")
    ir = bundle / "ir" / info["onnx_sha256"]
    ir.mkdir(parents=True, exist_ok=True)
    xml, binary = ir / "model.xml", ir / "model.bin"
    if not (xml.is_file() and binary.is_file()):
        env = dict(os.environ, PYTHONPATH=str(ncs1_paths.MO))
        subprocess.run([sys.executable, str(ncs1_paths.MO / "mo_onnx.py"),
                        "--input_model", str(source), "--data_type", "FP16",
                        "--output_dir", str(ir)], env=env, check=True, timeout=120)
    core = IECore()
    devices = list(core.available_devices)
    if not any(name.startswith("MYRIAD") for name in devices):
        raise RuntimeError("The MA2450 is unavailable to the MYRIAD plugin")
    net = core.read_network(model=str(xml), weights=str(binary))
    input_name, = net.input_info if hasattr(net, "input_info") else net.inputs
    output_name, = net.outputs
    start = time.perf_counter()
    executable = core.load_network(network=net, device_name="MYRIAD", num_requests=1)
    load_ms = (time.perf_counter() - start) * 1000

    def infer(x):
        return executable.infer({input_name: x[None]})[output_name].reshape(-1)

    if args.test:
        visual = info["model"] == "forms"
        with np.load(str(args.test), allow_pickle=False) as files:
            x, reference, ids = (files[k] for k in ("x", "scores", "ids"))
            y = files["boxes"] if visual else files["y"]
            good = files["good"] if info["model"] == "grounding" else None
            counts = files["counts"] if good is not None else None
        expected = tuple(info["output_shape"][1:]) if visual else (len(info["labels"]),)
        if not len(y) or len(x) != len(y) or reference.shape != (len(y),) + expected:
            raise ValueError("Unexpected evaluation fixture sizes")
        if list(x[:1].shape) != info["input_shape"] or len(set(ids.tolist())) != len(y):
            raise ValueError("Unexpected evaluation input or duplicate IDs")
        for _ in range(5):
            infer(x[0])
        predictions, elapsed, point_hits = [], [], []
        max_error = 0.0
        for i, sample in enumerate(x):
            start = time.perf_counter()
            scores = infer(sample)
            elapsed.append((time.perf_counter() - start) * 1000)
            if not np.isfinite(scores).all():
                raise ValueError("Non-finite output at fixture " + str(ids[i]))
            max_error = max(max_error, float(np.abs(scores - reference[i].ravel()).max()))
            if visual:
                point_hits.append(hit_test(scores, y[i]))
                predictions.append(scores.reshape(4, -1).argmax(1))
                continue
            if counts is not None:
                scores[counts[i]:] = -np.inf
                reference[i, counts[i]:] = -np.inf
            predictions.append(int(scores.argmax()))
        predictions = np.asarray(predictions)
        if visual:
            hits = np.array(point_hits)
            reference_choices = reference.reshape(len(y), 4, -1).argmax(2)
        else:
            hits = predictions == y if good is None else good[np.arange(len(y)), predictions]
            reference_choices = reference.argmax(1)
        report = dict(model=info["model"], device="MYRIAD", available_devices=devices,
                      ncs1_runtime="OpenVINO 2020.3.2", onnx_sha256=digest(source),
                      ir_xml_sha256=digest(xml), ir_bin_sha256=digest(binary),
                      fixtures_sha256=digest(args.test), examples=int(hits.size), requests=len(y),
                      correct=int(hits.sum()),
                      reference_agreement=float((predictions == reference_choices).mean()),
                      max_logit_error=max_error, load_ms=load_ms,
                      inference_p50_ms=float(np.percentile(elapsed, 50)),
                      inference_p95_ms=float(np.percentile(elapsed, 95)))
        if visual:
            report["per_role_correct"] = dict(zip(ROLES, hits.sum(0).tolist()))
        elif good is None:
            report["per_class"] = {label: dict(count=int((y == i).sum()),
                                               correct=int(((y == i) & hits).sum()))
                                   for i, label in enumerate(info["labels"])}
        else:
            report["shortlist_contains_target"] = int(good.any(1).sum())
            report["lexical_top1_correct"] = int(good[:, 0].sum())
        (bundle / "device-evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Correct on MYRIAD:", report["correct"], "/", report["examples"])
        return
    labels = info["labels"]
    if args.image and info["model"] == "forms":
        from PIL import Image
        with Image.open(args.image) as image:
            coords = points(infer(pixels(image))) * image.size
        print(json.dumps(dict(device="MYRIAD", points=dict(zip(ROLES, coords.astype(int).tolist())))))
        return
    elif args.screen and info["model"] == "grounding":
        screen = json.loads(args.screen.read_text(encoding="utf8"))
        x, labels = encode_screen(screen["task"], screen["controls"], screen.get("history", ""))
        if not labels:
            print(json.dumps(dict(model="grounding", device="MYRIAD", proposal=None)))
            return
    elif args.text is not None and info["model"] == "intent":
        x = text_features(args.text)
    elif args.wav and info["model"] == "voice":
        with wave.open(str(args.wav), "rb") as stream:
            if stream.getsampwidth() != 2 or stream.getnchannels() != 1:
                raise ValueError("Expected 16-bit PCM mono")
            x = audio_features(np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
                               .astype(np.float32) / 32768, stream.getframerate())
    else:
        raise ValueError("Select the matching input type for this model")
    scores = infer(x)
    print(json.dumps(dict(model=info["model"], device="MYRIAD",
                          proposal=labels[int(scores[:len(labels)].argmax())])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--toolchain", type=Path, default=Path(__file__).resolve().parents[2] / "tools/ncs1")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--test", type=Path)
    inputs.add_argument("--text")
    inputs.add_argument("--wav", type=Path)
    inputs.add_argument("--screen", type=Path)
    inputs.add_argument("--image", type=Path)
    run(parser.parse_args())
