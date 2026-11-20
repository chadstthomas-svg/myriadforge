"""Verify and time a static Gather vector against one request per decision."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

from adaptive import AdaptiveRuntime, Profiles
from benchmark import CORPUS_SHA, PHRASES, exact_document, expected, quantiles
from blob_layout import inspect_blob
from policy import LIMIT
from runtime import Runtime


def evaluate(engine, repeats, corpus):
    rows = []
    for flags in itertools.product((False, True), repeat=len(PHRASES)):
        selected = [i for i, present in enumerate(flags) if present]
        rows.append((" ; ".join(PHRASES[i] for i in selected), expected(selected)))
    random.Random(120).shuffle(rows)

    def dispatch(group_size):
        for start in range(0, len(rows), group_size):
            portion = rows[start:start + group_size]
            ready = [engine.prepare_decision(text) for text, _ in portion]
            results = engine.decide_prepared(ready)
            for item, (_, gold) in zip(results, portion):
                if item["result"] != gold or item["request_documents"] != len(portion):
                    raise AssertionError("Gather results differed from the truth table")

    widths = sorted({1, engine.batch_capacity} | {n for n in (2, 8, 16, 32, 64) if n < engine.batch_capacity})
    samples = {width: [] for width in widths}
    for width in widths:
        dispatch(width)
    for trial in range(repeats):
        order = widths if trial % 2 == 0 else list(reversed(widths))
        for width in order:
            start = time.perf_counter()
            dispatch(width)
            samples[width].append((time.perf_counter() - start) * 1000)
    timings = {str(width): dict(ms_per_128_documents=quantiles(samples[width]),
                                device_requests=(128 + width - 1) // width) for width in widths}

    payload = corpus.read_bytes()
    if hashlib.sha256(payload).hexdigest() != CORPUS_SHA:
        raise ValueError("Unexpected benchmark corpus")
    source = engine.tokenizer.encode(payload.decode("utf-8-sig"))
    documents = [
        {"id": "early", "text": exact_document(engine.tokenizer, source, LIMIT, start="charged twice review authorized")},
        {"id": "overflow", "text": exact_document(engine.tokenizer, source, LIMIT + 1)},
        {"id": "late", "text": exact_document(engine.tokenizer, source, LIMIT, end="package missing review authorized")},
        {"id": "veto", "text": exact_document(engine.tokenizer, source, LIMIT, start="parcel missing review authorized",
                                               end="review cancelled")},
        {"id": "empty", "text": ""},
    ]
    profiles = Profiles()
    try:
        routed = AdaptiveRuntime({"support": lambda: engine}, profiles).batch(
            {"vertical": "support", "documents": documents}
        )
    finally:
        profiles.close()
    want = [expected([0, 4]), None, expected([2, 4]), expected([3, 4, 6]), expected([])]
    if [row["id"] for row in routed["documents"]] != [row["id"] for row in documents]:
        raise AssertionError("Grouped dispatch changed response ordering")
    for row, gold in zip(routed["documents"], want):
        if gold is None:
            if str(LIMIT + 1) not in row.get("error", ""):
                raise AssertionError("Overlong input reached the stick")
            continue
        if row["result"] != gold or row["needle_tokens"] != (0 if row["id"] == "empty" else LIMIT):
            raise AssertionError("Unexpected result for mixed-length grouped input")
    return dict(model=engine.model_name, backend=engine.target, lane=engine.lane,
                toolchain=engine.openvino_version, capacity=engine.batch_capacity,
                onnx_sha256=engine.info["onnx_sha256"],
                truth_table_cases=len(rows), repeats=repeats, timing=timings,
                mixed_length_device_requests=routed["device_requests"],
                mixed_length_results=[{"id": r["id"], "tokens": r.get("needle_tokens"),
                                       "result": r.get("result"), "error": r.get("error")}
                                      for r in routed["documents"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--blob", type=Path)
    parser.add_argument("--toolchain", type=Path)
    parser.add_argument("--lane", choices=("ncs1", "ncs2"), default="ncs1")
    parser.add_argument("--target")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=13)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    runtime = Runtime(args.bundle, args.toolchain, lane=args.lane, target=args.target, blob=args.blob)
    report = evaluate(runtime, args.repeats, args.corpus)
    if args.blob:
        compiled = args.blob.read_bytes()
        report["compiled_blob_bytes"] = len(compiled)
        report["compiled_blob_sha256"] = hashlib.sha256(compiled).hexdigest()
        if args.lane == "ncs1":
            report["legacy_layout"] = inspect_blob(compiled)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"timing": report["timing"], "mixed_length_requests": report["mixed_length_device_requests"]}))
