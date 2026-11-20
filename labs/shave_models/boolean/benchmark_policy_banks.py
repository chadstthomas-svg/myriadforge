"""Check that two independent policies share one native decision graph."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

from benchmark import CORPUS_SHA, PHRASES, exact_document, expected, quantiles
from benchmark_batch_heads import windows_usb_identity
from blob_layout import inspect_blob
from policy import LIMIT
from runtime import Runtime


ONCALL = ("gateway unreachable", "connection reset", "queue backed up", "workers saturated",
          "paging approved", "maintenance window", "page cancelled")


def oncall_expected(indices):
    facts = set(indices)
    network = bool(facts & {0, 1})
    capacity = bool(facts & {2, 3})
    ready = (network or {2, 3} <= facts) and 4 in facts and not (facts & {5, 6})
    category = "mixed_incident" if network and capacity else "network" if network else "capacity" if capacity else None
    actions = {"mixed_incident": "page_incident", "network": "page_network", "capacity": "page_capacity"}
    return {"answers": {"may_page": ready}, "category": category, "action": actions[category] if ready else None}


def literal_fixtures():
    for flags in itertools.product((False, True), repeat=7):
        selected = [i for i, present in enumerate(flags) if present]
        for bank, names, gold in (("support", PHRASES, expected), ("oncall", ONCALL, oncall_expected)):
            yield {"bank": bank, "text": "; ".join(names[i] for i in selected)}, gold(selected)


def evaluate(engine, corpus, repeats):
    cases = list(literal_fixtures())

    def check(fixtures, separate=False):
        rows = [dict(request, id=index) for index, (request, _) in enumerate(fixtures)]
        if separate:
            replies = [engine.decide_mixed([row for row in rows if row["bank"] == bank])
                       for bank in ("support", "oncall")]
            outcomes = sorted((row for reply in replies for row in reply["documents"]), key=lambda row: row["id"])
            calls = sum(reply["device_requests"] for reply in replies)
        else:
            reply = engine.decide_mixed(rows)
            outcomes, calls = reply["documents"], reply["device_requests"]
        for index, outcome in enumerate(outcomes):
            request, want = fixtures[index]
            if outcome.get("id") != index or outcome.get("bank") != request["bank"] or outcome.get("result") != want:
                raise AssertionError("Wrong bank, order or result")
        if len(outcomes) != len(fixtures):
            raise AssertionError("Missing decisions")
        return calls

    exhaustive_calls = check(cases)
    group = cases[64:192]
    timings = {name: [] for name in ("mixed", "separate")}
    for index in range(repeats):
        for name in (("mixed", "separate") if index % 2 == 0 else ("separate", "mixed")):
            start = time.perf_counter()
            check(group, separate=name == "separate")
            timings[name].append((time.perf_counter() - start) * 1000)

    payload = corpus.read_bytes()
    if hashlib.sha256(payload).hexdigest() != CORPUS_SHA:
        raise ValueError("Unexpected benchmark corpus")
    tokenizer = engine.tokenizer
    source = tokenizer.encode(payload.decode("utf-8-sig"))
    long = [
        ("support", exact_document(tokenizer, source, LIMIT, start="duplicate charge",
                                    middle="queue backed up maintenance window", end="review authorized"), expected([1, 4])),
        ("oncall", exact_document(tokenizer, source, LIMIT, start="queue backed up paging approved",
                                   middle="package missing review cancelled"), oncall_expected([2, 4])),
        ("oncall", exact_document(tokenizer, source, LIMIT, start="queue backed up paging approved",
                                   middle="package missing review cancelled", end="workers saturated"), oncall_expected([2, 3, 4])),
    ]
    requests = [dict(id=index, bank=bank, text=text) for index, (bank, text, _) in enumerate(long)]
    requests.insert(1, dict(id="overflow", bank="support", text=exact_document(tokenizer, source, LIMIT + 1)))
    reply = engine.decide_mixed(requests)
    if [row["id"] for row in reply["documents"]] != [row["id"] for row in requests]:
        raise AssertionError("Mixed-context responses were lost or reordered")
    for sent, outcome in zip(requests, reply["documents"]):
        if sent["id"] == "overflow":
            if str(LIMIT + 1) not in outcome.get("error", ""):
                raise AssertionError("Missing overflow rejection")
        elif outcome["needle_tokens"] != LIMIT or outcome["result"] != long[sent["id"]][2]:
            raise AssertionError("Long context was truncated or reached the wrong policy")
    if reply["device_requests"] != (len(long) + engine.batch_capacity - 1) // engine.batch_capacity:
        raise AssertionError("Unexpected number of mixed-context requests")
    return dict(model=engine.model_name, backend=engine.target, runtime=engine.openvino_version,
                onnx_sha256=engine.info["onnx_sha256"], capacity=engine.batch_capacity,
                constants_bytes=engine.info["fp16_coefficient_bytes"],
                distinct_policy_cases=len(cases), full_truth_table_requests=exhaustive_calls,
                full_context_cases=len(long), overflow_rejected=True,
                full_context_requests=reply["device_requests"], repeated_trials=repeats,
                timing_128_documents={name: dict(wall_ms=quantiles(values), calls=check(group, name == "separate"))
                                      for name, values in timings.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--blob", type=Path)
    parser.add_argument("--toolchain", type=Path)
    parser.add_argument("--lane", choices=["ncs1", "ncs2"], default="ncs1")
    parser.add_argument("--expect-stick", choices=["ncs1", "ncs2"])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=13)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    identity = windows_usb_identity(args.expect_stick) if args.expect_stick else None
    engine = Runtime(args.bundle, args.toolchain, lane=args.lane, blob=args.blob)
    report = evaluate(engine, args.corpus, args.repeats)
    if identity:
        report["physical_usb_preflight"] = identity
    if args.blob and args.lane == "ncs1":
        report["blob"] = inspect_blob(args.blob.read_bytes())
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report))
