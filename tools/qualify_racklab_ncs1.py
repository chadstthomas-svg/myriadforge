"""Smoke-test the compiled automotive Boolean head on an original Intel stick."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--toolchain", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labs" / "shave_models" / "boolean"))
    from benchmark_batch_heads import windows_usb_identity
    from runtime import Runtime

    device = windows_usb_identity("ncs1")
    runtime = Runtime(args.bundle, toolchain=args.toolchain)
    flags = list(itertools.product((False, True), repeat=len(runtime.policy.literals)))
    texts = [" ; ".join(" ".join(phrase) for phrase, present in zip(runtime.policy.literals, row) if present)
             for row in flags]
    expected = [runtime.policy.result(runtime.policy.reference(row)) for row in flags]
    samples = []
    for _ in range(6):
        started = time.perf_counter()
        outputs = runtime.decide_prepared([runtime.prepare_decision(text) for text in texts])
        samples.append((time.perf_counter() - started) * 1000)
        if [row["result"] for row in outputs] != expected:
            raise SystemExit("NCS1 output disagrees with the reference truth table")
    record = {
        "scope": "development smoke, Boolean head only",
        "device": device,
        "placement": {"policy": "ncs1", "routing_and_evidence": "host"},
        "assignments_passed": len(flags),
        "warm_batch_p50_ms": statistics.median(samples[1:]),
        "samples_ms": samples,
        "onnx_sha256": hashlib.sha256((args.bundle / "model.onnx").read_bytes()).hexdigest(),
    }
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
