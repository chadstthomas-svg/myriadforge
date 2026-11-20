"""Measure exact append-only decisions near the full Needle context limit."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark import CORPUS_SHA, exact_document, expected, quantiles
from benchmark_batch_heads import windows_usb_identity
from policy import LIMIT, load_tokenizer
from runtime import Runtime
from stream import DecisionStream


def evaluate(engine, corpus):
    raw = corpus.read_bytes()
    if hashlib.sha256(raw).hexdigest() != CORPUS_SHA:
        raise ValueError("Unexpected benchmark corpus")
    tokenizer = engine.tokenizer
    source = tokenizer.encode(raw.decode("utf-8-sig"), out_type=int)
    prefix = exact_document(tokenizer, source, LIMIT - 640, start="duplicate charge. review authorized.")
    fragments = [" Note %d filed." % i for i in range(24)]
    veto = " do not route"
    total = len(tokenizer.encode(prefix + "".join(fragments) + veto, out_type=int))
    if total >= LIMIT:
        raise AssertionError("Benchmark did not leave space for its final append")
    fragments.append(" a" * (LIMIT - total) + veto)
    stream = DecisionStream(engine)
    initial = stream.append(prefix)
    assert initial["needle_tokens"] == LIMIT - 640 and initial["result"] == expected([1, 4])
    whole = prefix
    samples = {"full_text": [], "stream": []}
    for index, delta in enumerate(fragments):
        whole += delta
        if index % 2:
            incremental, full = stream.append(delta), engine.decide(whole)
        else:
            full, incremental = engine.decide(whole), stream.append(delta)
        for key in ("needle_tokens", "bits", "atoms", "result"):
            if incremental[key] != full[key]:
                raise AssertionError("Incremental replay disagreed with full input on %s" % key)
        # Report the stable short-appends separately from the final padded veto.
        if index < len(fragments) - 1:
            samples["full_text"].append(full)
            samples["stream"].append(incremental)
    assert len(tokenizer.encode(whole, out_type=int)) == LIMIT
    assert incremental["needle_tokens"] == LIMIT and incremental["result"] == expected([1, 4, 5])
    saved = stream.state
    try:
        stream.append(" b")
    except ValueError as exc:
        if str(LIMIT + 1) not in str(exc):
            raise
    else:
        raise AssertionError("Expected overflow rejection")
    assert stream.state is saved
    timing = {}
    for name, records in samples.items():
        host = np.array([r["tokenization_ms"] + r["host_projection_ms"] for r in records])
        timing[name] = dict(host_preparation_ms=quantiles(host),
                            total_ms=quantiles([r["total_ms"] for r in records]))
    return dict(model=engine.model_name, backend=engine.target, openvino_version=engine.openvino_version,
                onnx_sha256=engine.info["onnx_sha256"], samples=len(samples["stream"]),
                initial_tokens=initial["needle_tokens"], initial_ms=initial["total_ms"],
                final_tokens=incremental["needle_tokens"], final_veto=incremental["result"],
                overflow_rejected=True, timing=timing)


def cpu_control(bundle):
    from openvino.runtime import Core, get_version

    engine = Runtime.__new__(Runtime)
    engine.folder = bundle
    engine.bundle_info = json.loads((bundle / "model.json").read_text(encoding="utf8"))
    engine._use_bank(None)
    engine.tokenizer = load_tokenizer()
    engine.fast_count = True
    engine.lane = "ncs2"
    engine.target = "CPU"
    engine.openvino_version = get_version()
    engine.compiled = Core().compile_model(str(engine.resolve_ir()), engine.target)
    engine.request = engine.compiled.create_infer_request()
    return engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--blob", type=Path)
    parser.add_argument("--toolchain", type=Path)
    parser.add_argument("--lane", choices=["ncs1", "ncs2"], default="ncs1")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--expect-stick", choices=["ncs1", "ncs2"])
    target.add_argument("--cpu", action="store_true", help="Use OpenVINO 2022.3's CPU plugin and an IR bundle")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cpu and (args.blob or args.bundle.is_file()):
        parser.error("--cpu requires an IR bundle directory")
    identity = windows_usb_identity(args.expect_stick) if args.expect_stick else None
    engine = (cpu_control(args.bundle) if args.cpu
              else Runtime(args.bundle, args.toolchain, lane=args.lane, blob=args.blob,
                           release_stick=args.expect_stick or "ncs1"))
    report = evaluate(engine, args.corpus)
    report["physical_device_verified"] = identity is not None
    if identity:
        report["physical_usb_preflight"] = identity
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report))
