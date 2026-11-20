"""Compare exact MagicEightBall decisions on a physical NCS1 and CPU baselines."""

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import random
import time
from typing import NamedTuple

import numpy as np

from benchmark import CORPUS_SHA, PHRASES, exact_document, expected, quantiles
from benchmark_batch_heads import windows_usb_identity
from benchmark_stream import cpu_control
from cpu_tree_control import decide_tree, fit_tree
from policy import LIMIT, Policy, load_tokenizer
from release import read_release
from runtime import Runtime


HERE = Path(__file__).resolve().parent
SUPPORT = HERE / "support.json"
BUNDLE = HERE / "bundles" / "magiceightball-batch"
RELEASE = HERE / "bundles" / "magiceightball.m8b"
TOKENIZER = HERE / "needle3-tokenizer.m8t.xz"


def bit_mask(bits):
    return sum(int(bit) << index for index, bit in enumerate(bits))


class LocalRuntime(Runtime):
    """Reuse the runner's text and result paths with a CPU decision head."""

    def __init__(self, spec, backend):
        self.folder = HERE
        self.bundle_info = {"variant": "gather", "input_shapes": {"features": [128]}, "policy": spec}
        self._use_bank(None)
        self.tokenizer = load_tokenizer()
        self.fast_count = True
        self.target = "CPU"
        self.table = self.tree = None
        if backend != "policy-cpu":
            atom_count = len(self.policy.literals)
            codes = bytes(bit_mask(self.policy.reference([(mask >> i) & 1 for i in range(atom_count)]))
                          for mask in range(1 << atom_count))
            if backend == "table-cpu":
                self.table = codes
            else:
                self.tree = fit_tree(codes)

    def gather_heads(self, facts):
        if self.table is not None:
            words = [self.table[bit_mask(row)] for row in facts]
        elif self.tree is not None:
            words = [decide_tree(self.tree, bit_mask(row)) for row in facts]
        else:
            return [self.policy.reference(row) for row in facts]
        return np.asarray([(word >> i) & 1 for word in words for i in range(len(self.policy.outputs))],
                          dtype=np.float32).reshape(len(words), len(self.policy.outputs))


class InferenceTimer:
    def __init__(self, request):
        self.request = request
        self.calls = 0
        self.seconds = 0.0

    def infer(self, inputs):
        start = time.perf_counter()
        result = self.request.infer(inputs)
        self.seconds += time.perf_counter() - start
        self.calls += 1
        return result


class Fixture(NamedTuple):
    name: str
    text: str
    tokens: int
    atoms: list
    bits: list
    result: dict


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_details(path):
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": digest(data)}


def fixtures(tokenizer, policy, corpus):
    payload = corpus.read_bytes()
    if digest(payload) != CORPUS_SHA:
        raise ValueError("Unexpected benchmark corpus")
    text = payload.decode("utf-8-sig")
    if any(policy.atoms(policy.project(text))):
        raise AssertionError("Benchmark corpus unexpectedly contains a policy phrase")
    source = tokenizer.encode(text, out_type=int)

    def row(name, text, present, length=None):
        tokens = len(tokenizer.encode(text, out_type=int))
        if length is not None and tokens != length:
            raise AssertionError("Unexpected SentencePiece count for %s" % name)
        phrases = {PHRASES[index] for index in present}
        atoms = [float(" ".join(literal) in phrases) for literal in policy.literals]
        return Fixture(name, text, tokens, atoms, policy.reference(atoms).tolist(), expected(present))

    short = []
    for index, flags in enumerate(itertools.product((False, True), repeat=len(PHRASES))):
        present = [i for i, flag in enumerate(flags) if flag]
        short.append(row("assignment_%03d" % index, " ; ".join(PHRASES[i] for i in present), present))
    random.Random(120).shuffle(short)
    allow = row("full_authorized", exact_document(tokenizer, source, LIMIT, start="duplicate charge",
                                                  end="review authorized"), [1, 4], LIMIT)
    veto = row("full_late_veto", exact_document(tokenizer, source, LIMIT, start="duplicate charge",
                                                middle="review authorized", end="review cancelled"),
               [1, 4, 6], LIMIT)
    overflow = exact_document(tokenizer, source, LIMIT + 1)
    return short, allow, veto, overflow


def dispatch(engine, rows, width):
    records = []
    prepare_seconds = decide_seconds = 0.0
    start = time.perf_counter()
    for offset in range(0, len(rows), width):
        group = rows[offset:offset + width]
        started = time.perf_counter()
        prepared = [engine.prepare_decision(row.text) for row in group]
        ready = time.perf_counter()
        decisions = engine.decide_prepared(prepared)
        finished = time.perf_counter()
        records.extend(decisions)
        prepare_seconds += ready - started
        decide_seconds += finished - ready
    elapsed = (time.perf_counter() - start) * 1000
    if len(records) != len(rows):
        raise AssertionError("Unexpected number of decisions")
    for got, row in zip(records, rows):
        actual = (got["needle_tokens"], got["atoms"], got["bits"], got["result"])
        if actual != (row.tokens, row.atoms, row.bits, row.result):
            raise AssertionError("Wrong decision on %s" % row.name)
    return {"wall_ms": elapsed, "host_prepare_ms": prepare_seconds * 1000,
            "decision_and_decode_ms": decide_seconds * 1000}


def head_only(engine, facts, want_bits):
    start = time.perf_counter()
    actual = engine.gather_heads(facts)
    elapsed = (time.perf_counter() - start) * 1000
    if not np.array_equal(actual, want_bits):
        raise AssertionError("Wrong prepared head outputs")
    return {"wall_ms": elapsed}


def evaluate(engine, reference, corpus, repeats, warmups, include_fixtures=False):
    short, allow, veto, overflow = fixtures(engine.tokenizer, reference, corpus)
    facts = np.asarray([row.atoms for row in short], dtype=np.float32)
    want_bits = [row.bits for row in short]
    cases = {
        "short_128_sequential": lambda: dispatch(engine, short, 1),
        "short_128_vector": lambda: dispatch(engine, short, 128),
        "full_authorized_one": lambda: dispatch(engine, [allow], 1),
        "full_late_veto_one": lambda: dispatch(engine, [veto], 1),
        "prepared_head_128_vector": lambda: head_only(engine, facts, want_bits),
    }
    logical = isinstance(engine, LocalRuntime)
    timer = InferenceTimer(engine.request) if not logical else None
    if timer is not None:
        engine.request = timer
    try:
        engine.decide(overflow)
    except ValueError as exc:
        if "maximum" not in str(exc) or (timer is not None and timer.calls):
            raise AssertionError("Overflow failed at the wrong stage") from exc
    else:
        raise AssertionError("Expected rejection at %d tokens" % (LIMIT + 1))

    collected = {name: [] for name in cases}
    case_names = list(cases)
    for trial in range(warmups + repeats):
        order = case_names if trial % 2 == 0 else reversed(case_names)
        for name in order:
            calls_before = timer.calls if timer is not None else 0
            seconds_before = timer.seconds if timer is not None else 0.0
            sample = cases[name]()
            sample["openvino_requests"] = timer.calls - calls_before if timer is not None else 0
            if timer is not None:
                sample["openvino_infer_ms"] = (timer.seconds - seconds_before) * 1000
                wanted = 128 if name == "short_128_sequential" else 1
                if sample["openvino_requests"] != wanted:
                    raise AssertionError("Unexpected request count for %s" % name)
            if trial >= warmups:
                collected[name].append(sample)
    details = [{"name": row.name, "input_sha256": digest(row.text.encode("utf8")),
                "tokens": row.tokens, "atoms": row.atoms, "bits": row.bits, "result": row.result}
               for row in short + [allow, veto]]
    report = {
        "corpus_sha256": CORPUS_SHA,
        "fixture_contract_sha256": digest(json.dumps(details, sort_keys=True).encode("utf8")),
        "truth_table_assignments": len(short),
        "full_context": details[-2:],
        "overflow": {"tokens": LIMIT + 1, "rejected": True,
                     "input_sha256": digest(overflow.encode("utf8"))},
        "repeats": repeats,
        "warmups": warmups,
        "timings": {},
    }
    for name, samples in collected.items():
        report["timings"][name] = {key: quantiles([sample[key] for sample in samples])
                                   for key in samples[0] if key.endswith("_ms")}
        report["timings"][name]["openvino_requests"] = samples[0]["openvino_requests"]
    if include_fixtures:
        report["fixture_oracles"] = details
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True,
                        choices=["ncs1", "openvino-cpu", "policy-cpu", "table-cpu", "tree-cpu"])
    parser.add_argument("--bundle", type=Path, help="Override the shipped archive or matching CPU IR bundle")
    parser.add_argument("--toolchain", type=Path, help="OpenVINO 2020.3.2 directory for NCS1")
    parser.add_argument("--expect-cold-ncs1", action="store_true", help="Attest physical USB identity before import")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=13)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--include-fixtures", action="store_true", help="Include all expected outputs and input hashes")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error("repeats must be positive and warmups must be nonnegative")
    if args.backend != "ncs1" and (args.expect_cold_ncs1 or args.toolchain):
        parser.error("cold identity and legacy toolchain are NCS1 options")
    if args.backend in ("policy-cpu", "table-cpu", "tree-cpu") and args.bundle is not None:
        parser.error("direct CPU baselines use the checked-in support policy")

    spec = json.loads(SUPPORT.read_text(encoding="utf8"))
    reference = Policy(spec)
    identity = windows_usb_identity("ncs1") if args.expect_cold_ncs1 else None
    artifacts = {"shared_tokenizer_archive": file_details(TOKENIZER)}
    started = time.perf_counter()
    if args.backend == "ncs1":
        bundle = args.bundle or RELEASE
        if bundle.suffix.lower() != ".m8b":
            parser.error("NCS1 comparison requires the shipped archive format")
        manifest, blobs = read_release(bundle)
        artifacts["dual_target_release_archive"] = file_details(bundle)
        artifacts["ncs1_native_blob"] = {"bytes": len(blobs["ncs1"]), "sha256": digest(blobs["ncs1"])}
        artifacts["native_fp16_coefficients_bytes"] = manifest["fp16_coefficient_bytes"]
        engine = Runtime(bundle, args.toolchain)
    elif args.backend == "openvino-cpu":
        bundle = args.bundle or BUNDLE
        engine = cpu_control(bundle)
        ir = engine.resolve_ir()
        artifacts.update(onnx=file_details(bundle / "model.onnx"), ir_xml=file_details(ir),
                         ir_weights=file_details(ir.with_suffix(".bin")), cpu_bundle_manifest=file_details(bundle / "model.json"))
    else:
        engine = LocalRuntime(spec, args.backend)
        artifacts["source_policy_json"] = file_details(SUPPORT)
        if engine.table is not None:
            artifacts["generated_uint8_truth_table"] = {"bytes": len(engine.table), "sha256": digest(engine.table)}
        if engine.tree is not None:
            artifacts["generated_binary_decision_diagram"] = {"bytes": len(engine.tree), "sha256": digest(engine.tree)}
    initialized_ms = (time.perf_counter() - started) * 1000
    if engine.info["policy"] != spec or engine.batch_capacity != 128:
        raise ValueError("Comparison requires the same support policy and 128 document slots")
    report = evaluate(engine, reference, args.corpus, args.repeats, args.warmups, args.include_fixtures)
    report.update(backend=args.backend, device=engine.target, openvino_version=getattr(engine, "openvino_version", None),
                  onnx_sha256=engine.info.get("onnx_sha256"), artifacts=artifacts,
                  host=dict(processor=platform.processor(), platform=platform.platform(), logical_cpus=os.cpu_count(),
                            python=platform.python_version()), runner_initialization_ms=initialized_ms,
                  physical_usb_preflight=identity,
                  timing_scope="Resident in-process UTF-8 text decisions. Wall time includes tokenization, evidence, "
                  "submission and output records. Corpus reads, validation and JSON serialization are outside it. "
                  "Prepared head timing includes packing and unpacking without text preparation. OpenVINO infer "
                  "time includes plugin dispatch and any device transfer, not isolated accelerator compute.",
                  artifact_scope="File and generated data sizes exclude runtime libraries, firmware and host code.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps({name: result["wall_ms"] for name, result in report["timings"].items()}))


if __name__ == "__main__":
    main()
