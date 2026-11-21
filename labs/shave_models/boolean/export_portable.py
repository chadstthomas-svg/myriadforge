"""Export and replay CPU packages for the ONNX, FP32 and GGUF profiles."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import numpy as np

from benchmark_compare import fixtures
from policy import LIMIT, Policy
from portable_runtime import ARCHITECTURE, PROFILE_REVISION, TABLE, PortableRuntime


LAB = Path(__file__).resolve().parent
SUPPORT = LAB / "support.json"
CPU_SOURCES = ("portable_runtime.py", "runtime.py", "policy.py", "radix.py", "compact_tokenizer.py",
               "release.py", "blob_layout.py", "needle3-tokenizer.m8t.xz")
ONNX_PROFILES = {"onnx": "magiceightball-radix", "fp32": "dense"}


def write_gguf(path, spec):
    from gguf import GGUFWriter

    policy = Policy(spec)
    facts = len(policy.literals)
    if facts > 10 or not 1 <= len(policy.outputs) <= 7:
        raise ValueError("The GGUF I8 profile supports at most ten facts and one to seven Boolean outputs")
    table = []
    for address in range(1 << facts):
        row = [(address >> index) & 1 for index in range(facts)]
        table.append(sum(int(bit) << index for index, bit in enumerate(policy.reference(row))))
    writer = GGUFWriter(path, ARCHITECTURE)
    writer.add_name("MagicEightBall")
    writer.add_string("magiceightball.policy_json", json.dumps(spec, separators=(",", ":")))
    writer.add_uint32("magiceightball.profile_revision", PROFILE_REVISION)
    writer.add_uint32("magiceightball.max_tokens", LIMIT)
    writer.add_tensor(TABLE, np.asarray(table, dtype=np.int8))
    writer.write_header_to_file()
    try:
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    return len(table)


def verify(model, runtime, policy, corpus):
    facts = [np.asarray(row, dtype=np.float32)
             for row in itertools.product((0, 1), repeat=len(policy.literals))]
    expected = [policy.reference(row) for row in facts]
    actual = runtime.gather_heads(facts) if runtime.batch_capacity == len(facts) else [runtime.head(row) for row in facts]
    if not np.array_equal(expected, actual):
        raise AssertionError("Raw fact replay differs from the policy: " + str(model))

    short, allow, veto, overflow = fixtures(runtime.tokenizer, policy, corpus)
    cases = short + [allow, veto]
    for start in range(0, len(cases), runtime.batch_capacity):
        group = cases[start:start + runtime.batch_capacity]
        received = runtime.decide_prepared([runtime.prepare_decision(item.text) for item in group])
        if len(received) != len(group):
            raise AssertionError("Missing decision in " + str(model))
        for case, outcome in zip(group, received):
            if (outcome["result"] != case.result or outcome["bits"] != case.bits
                    or outcome["atoms"] != case.atoms or outcome["needle_tokens"] != case.tokens):
                raise AssertionError("Text replay mismatch in %s for %s" % (model, case.name))
    try:
        runtime.decide(overflow)
    except ValueError as error:
        if str(LIMIT + 1) not in str(error):
            raise
    else:
        raise AssertionError("An over-limit input was accepted")

    return {"fact_assignments": len(facts), "short_documents": len(short),
            "context_cases": [dict(name=row.name, tokens=row.tokens, result=row.result) for row in (allow, veto)],
            "first_rejected_length": LIMIT + 1, "artifact_bytes": model.stat().st_size,
            "artifact_sha256": hashlib.sha256(model.read_bytes()).hexdigest(), "backend": runtime.target}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(SUPPORT.read_bytes())
    policy = Policy(spec)
    args.output.mkdir(parents=True, exist_ok=False)
    for flavor in ("onnx", "fp32", "gguf"):
        destination = args.output / flavor
        destination.mkdir()
        for name in CPU_SOURCES:
            shutil.copyfile(LAB / name, destination / name)
        if flavor in ONNX_PROFILES:
            source = LAB / "bundles" / ONNX_PROFILES[flavor]
            for name in ("model.onnx", "model.json"):
                shutil.copyfile(source / name, destination / name)
            artifact = destination / "model.onnx"
        else:
            artifact = destination / "model.gguf"
            decision_bytes = write_gguf(artifact, spec)
        runtime = PortableRuntime(artifact)
        evidence = verify(artifact, runtime, policy, args.corpus)
        if flavor == "gguf":
            evidence.update(gguf_architecture=ARCHITECTURE, decision_tensor_bytes=decision_bytes)
        (destination / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf8")
        print("%s: %d bytes, %d exact assignments" % (flavor, evidence["artifact_bytes"], evidence["fact_assignments"]))


if __name__ == "__main__":
    main()
