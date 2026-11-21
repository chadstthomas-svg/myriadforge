"""Compare the three portable MagicEightBall profiles using the pinned CPU workloads."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_compare import CORPUS_SHA, SUPPORT, TOKENIZER, digest, dispatch, file_details, fixtures, quantiles
from policy import LIMIT, Policy
from portable_runtime import PortableRuntime


ARTIFACTS = {
    "onnx": ("model.onnx", "gather", "ONNX Runtime CPUExecutionProvider"),
    "fp32": ("model.onnx", "dense", "ONNX Runtime CPUExecutionProvider"),
    "gguf": ("model.gguf", "gguf", "MagicEightBall CPU table reader"),
}


def prepared_heads(engine, facts, expected):
    start = time.perf_counter()
    if engine.batch_capacity == len(facts):
        actual = engine.gather_heads(facts)
    else:
        actual = [engine.head(row) for row in facts]
    elapsed_ms = (time.perf_counter() - start) * 1000
    if not np.array_equal(actual, expected):
        raise AssertionError("Wrong bits from precomputed Boolean assignments")
    return {"wall_ms": elapsed_ms}


def verify_overflow(engine, document):
    try:
        engine.decide(document)
    except ValueError as exc:
        if "maximum" not in str(exc):
            raise AssertionError("Overflow failed at the wrong stage") from exc
    else:
        raise AssertionError(f"Expected rejection at {LIMIT + 1} tokens")


def describe_fixture(row):
    return {"name": row.name, "input_sha256": digest(row.text.encode("utf8")),
            "tokens": row.tokens, "atoms": row.atoms, "bits": row.bits, "result": row.result}


def workload_specs(engine, short, allow, veto, facts, expected):
    capacity = engine.batch_capacity
    cases = {
        "short_128_sequential": (len(short), 1, lambda: dispatch(engine, short, 1)),
        "prepared_head_128_native": (len(short), capacity, lambda: prepared_heads(engine, facts, expected)),
        "full_authorized_one": (1, 1, lambda: dispatch(engine, [allow], 1)),
        "full_late_veto_one": (1, 1, lambda: dispatch(engine, [veto], 1)),
    }
    if capacity > 1:
        cases["short_128_native"] = (len(short), capacity, lambda: dispatch(engine, short, capacity))
    return cases


def summarize(samples):
    summary = {}
    for key in samples[0]:
        series = [sample[key] for sample in samples]
        summary[key] = {"min": min(series), **quantiles(series)}
    return summary


def evaluate(export_root, corpus, repeats, warmups, check_only):
    policy_spec = json.loads(SUPPORT.read_text(encoding="utf8"))
    policy_rules = {key: value for key, value in policy_spec.items() if key != "name"}
    engines = {name: PortableRuntime(export_root / name / spec[0]) for name, spec in ARTIFACTS.items()}
    short, allow, veto, overflow = fixtures(engines["onnx"].tokenizer, Policy(policy_spec), corpus)
    details = [describe_fixture(row) for row in [*short, allow, veto]]
    facts = np.asarray([row.atoms for row in short], dtype=np.float32)
    expected = np.asarray([row.bits for row in short], dtype=np.float32)
    profiles = {}
    cases = []
    for name, (filename, variant, backend) in ARTIFACTS.items():
        engine = engines[name]
        export_rules = {key: value for key, value in engine.info["policy"].items() if key != "name"}
        expected_capacity = len(short) if variant == "gather" else 1
        if (export_rules != policy_rules or engine.info["variant"] != variant
                or engine.batch_capacity != expected_capacity):
            raise AssertionError(f"Unexpected policy, variant, or capacity for {name}")
        verify_overflow(engine, overflow)
        artifact = export_root / name / filename
        workloads = {}
        profiles[name] = {"variant": variant, "engine": backend, "artifact": file_details(artifact),
                          "batch_capacity": engine.batch_capacity, "workloads": workloads}
        for key, (documents, width, run) in workload_specs(engine, short, allow, veto, facts, expected).items():
            run()
            workload = {"documents": documents, "group_width": width, "head_calls": (documents + width - 1) // width}
            workloads[key] = workload
            cases.append((run, workload))
        if engine.batch_capacity == 1:
            workloads["short_128_native"] = {"same_workload_as": "short_128_sequential"}

    if not check_only:
        for trial in range(warmups + repeats):
            for run, workload in cases if trial % 2 == 0 else reversed(cases):
                sample = run()
                if trial >= warmups:
                    workload.setdefault("samples", []).append(sample)
        for _, workload in cases:
            workload["summary"] = summarize(workload["samples"])

    return {
        "environment": {
            "platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version(),
            "packages": {package: version(package) for package in ("gguf", "numpy", "onnxruntime", "sentencepiece")},
            "onnxruntime_intra_op_threads": 1,
        },
        "corpus_sha256": CORPUS_SHA,
        "tokenizer": file_details(TOKENIZER),
        "fixture_contract_sha256": digest(json.dumps(details, sort_keys=True).encode("utf8")),
        "truth_table_assignments": len(short),
        "full_context": details[-2:],
        "overflow": {"tokens": LIMIT + 1, "rejected": True, "input_sha256": digest(overflow.encode("utf8"))},
        "repeats": 0 if check_only else repeats,
        "warmups": 0 if check_only else warmups,
        "profiles": profiles,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports", required=True, type=Path, help="folder containing onnx, fp32 and gguf exports")
    parser.add_argument("--corpus", required=True, type=Path, help="pinned decision-corpus.txt")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", default=31, type=int)
    parser.add_argument("--warmups", default=4, type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 5 or args.warmups < 0:
        parser.error("use at least five repeats and zero or more warmups")
    report = evaluate(args.exports, args.corpus, args.repeats, args.warmups, args.check_only)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(args.output)


if __name__ == "__main__":
    main()
