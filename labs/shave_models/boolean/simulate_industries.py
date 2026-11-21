"""Replay labeled industry scenarios on the resident CPU or NCS1 model."""

import argparse
import hashlib
from itertools import islice, product
import json
from pathlib import Path
import time

from adaptive import AdaptiveRuntime, Profiles
from benchmark import CORPUS_SHA, exact_document, quantiles
from benchmark_batch_heads import windows_usb_identity
from benchmark_stream import cpu_control
from policy import LIMIT
from runtime import Runtime


SCENARIOS = Path(__file__).with_name("industries") / "scenarios.json"


def evaluate(engine, corpus, scenarios, repeats):
    payload = corpus.read_bytes()
    if hashlib.sha256(payload).hexdigest() != CORPUS_SHA:
        raise ValueError("Unexpected benchmark corpus")
    source = engine.tokenizer.encode(payload.decode("utf-8-sig"), out_type=int)
    groups, controls = {}, {}
    for bank, suite in scenarios.items():
        policy = engine.policy_view(bank).policy
        auth = tuple(suite["authorization"].split())
        vetoes = {tuple(phrase.split()) for phrase in suite["vetoes"]}
        for case in suite["cases"]:
            if "record" in case:
                kind = "structured"
                evidence = case["record"]["evidence"]
                text = json.dumps(case["record"], ensure_ascii=False)
            elif "context" in case:
                kind = "full_context"
                evidence = text = exact_document(engine.tokenizer, source, LIMIT, **case["context"])
            else:
                kind = "text"
                evidence = text = case["text"]
            key = (bank, kind)
            groups.setdefault(key, []).append(dict(id=case["id"], text=text, expected=case["expected"]))

            # Keep the classifier and corroboration rules. Remove only the
            # approval and hold requirements to measure an ungated baseline.
            facts = policy.text_atoms(evidence)
            for index, phrase in enumerate(policy.literals):
                if phrase == auth:
                    facts[index] = 1
                elif phrase in vetoes:
                    facts[index] = 0
            controls[case["id"]] = policy.result(policy.reference(facts))

    keys = {bank + "-" + kind: (bank, kind) for bank, kind in groups}
    factory = lambda: engine
    profiles = Profiles()
    try:
        runner = AdaptiveRuntime({key: factory for key in keys}, profiles,
                                 scopes={key: "evidence" for key, (_, kind) in keys.items() if kind == "structured"},
                                 banks={key: bank for key, (bank, _) in keys.items()})
        results, timings = [], {}
        for vertical, (bank, kind) in keys.items():
            batch = {"vertical": vertical, "documents": groups[bank, kind]}
            elapsed = []
            for repeat in range(repeats + 4):
                started = time.perf_counter()
                output = runner.batch(batch)
                duration = (time.perf_counter() - started) * 1000
                if repeat >= 4:
                    elapsed.append(duration)
                if output["device_requests"] != 1 or len(output["documents"]) != len(batch["documents"]):
                    raise AssertionError("Unexpected inference grouping for " + vertical)
                for sent, observed in zip(batch["documents"], output["documents"]):
                    if observed.get("id") != sent["id"] or observed.get("result") != sent["expected"]:
                        raise AssertionError("Fixture failed: %s: %s" % (sent["id"], observed))
                    if kind == "full_context" and observed["needle_tokens"] != LIMIT:
                        raise AssertionError("Context fixture did not reach the token limit")
            for sent, observed in zip(batch["documents"], output["documents"]):
                results.append(dict(vertical=bank, workload=kind, id=sent["id"],
                                    tokens=observed["needle_tokens"], expected=sent["expected"],
                                    actual=observed["result"], ungated=controls[sent["id"]]))
            timings[vertical] = quantiles(elapsed)
    finally:
        profiles.close()

    blocked = [row for row in results if row["expected"]["action"] is None]
    return {"simulated_cases": len(results), "matched_cases": len(results),
            "ungated_matched_cases": sum(row["ungated"] == row["expected"] for row in results),
            "expected_no_action_cases": len(blocked),
            "wrong_proposals": {"magiceightball": 0, "ungated": sum(row["ungated"]["action"] is not None
                                                                      for row in blocked)},
            "repeats": repeats, "workload_ms": timings,
            "cases": results}



def verify_assignments(engine, banks):
    counts = {}
    for bank in banks:
        view = engine.policy_view(bank)
        assignments = product((0, 1), repeat=len(view.policy.literals))
        counts[bank] = 0
        while True:
            batch = list(islice(assignments, view.batch_capacity))
            if not batch:
                break
            expected = [list(view.policy.reference(facts)) for facts in batch]
            if [list(bits) for bits in view.gather_heads(batch)] != expected:
                raise AssertionError("Compiled Boolean assignments diverged for " + bank)
            counts[bank] += len(batch)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--backend", choices=("cpu", "ncs1"), default="cpu")
    parser.add_argument("--blob", type=Path)
    parser.add_argument("--toolchain", type=Path)
    parser.add_argument("--scenarios", type=Path, default=SCENARIOS)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--verify-assignments", action="store_true",
                        help="Also check every prepared Boolean assignment for each policy bank")
    args = parser.parse_args()
    if args.repeats < 1 or (args.backend == "cpu" and args.blob):
        parser.error("Repeats must be positive and a native blob requires --backend ncs1")
    physical = windows_usb_identity("ncs1") if args.backend == "ncs1" else None
    engine = Runtime(args.bundle, args.toolchain, blob=args.blob) if physical else cpu_control(args.bundle)
    cases = args.scenarios.read_bytes()
    scenarios = json.loads(cases)
    report = evaluate(engine, args.corpus, scenarios, args.repeats)
    if args.verify_assignments:
        report["verified_assignments_by_bank"] = verify_assignments(engine, scenarios)
    report.update(backend=engine.target, physical_usb_preflight=physical, runtime=engine.openvino_version,
                  onnx_sha256=engine.info["onnx_sha256"], scenario_sha256=hashlib.sha256(cases).hexdigest(),
                  manifest_sha256=hashlib.sha256((args.bundle / "model.json").read_bytes()).hexdigest(),
                  corpus_sha256=CORPUS_SHA, evidence_type="hand-authored simulation")
    if args.blob:
        report["native_blob_sha256"] = hashlib.sha256(args.blob.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps({name: report[name] for name in ("backend", "simulated_cases", "wrong_proposals")}))


if __name__ == "__main__":
    main()
