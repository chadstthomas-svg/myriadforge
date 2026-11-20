"""Replay held-out SLURP utterances and probe document-wide distractors on MYRIAD."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from adaptive import AdaptiveRuntime, Profiles
from benchmark import CORPUS_SHA, exact_document, quantiles
from benchmark_adaptive import json_document
from policy import LIMIT
from runtime import Runtime


def main(args):
    engine = Runtime(args.bundle, args.toolchain, lane=args.lane, target=args.target)
    samples_bytes = args.samples.read_bytes()
    rows = [json.loads(line) for line in samples_bytes.decode("utf8").splitlines()]
    timing = []
    correct_category = proposals = correct_proposal = 0
    for row in rows:
        record = engine.decide(row["text"])
        result = record["result"]
        if result != row["expected"] or any(bit not in (0, 1) for bit in record["bits"]):
            raise AssertionError((row, record))
        correct_category += result["category"] == row["category_label"]
        proposals += result["action"] is not None
        correct_proposal += result["action"] == row["action_label"]
        timing.append(record["total_ms"])

    book = args.corpus.read_bytes()
    if hashlib.sha256(book).hexdigest() != CORPUS_SHA:
        raise ValueError("Unexpected benchmark corpus")
    source = engine.tokenizer.encode(book.decode("utf-8-sig"), out_type=int)
    base = engine.decide(exact_document(engine.tokenizer, source, LIMIT))
    positions = ["early", "middle", "late"]
    rng = random.Random(1120)
    accepted = [row for row in rows if row["expected"]["action"] is not None]
    abstained = [row for row in rows if row["expected"]["action"] is None]
    selected = rng.sample(accepted, 6) + rng.sample(abstained, 6)
    long_cases = []
    for num, row in enumerate(selected):
        location = num % len(positions)
        slots = [""] * len(positions)
        slots[location] = " ; " + row["text"] + " ; "
        document = exact_document(engine.tokenizer, source, LIMIT, *slots)
        record = engine.decide(document)
        bits = engine.policy.reference(engine.policy.atoms(engine.policy.project(document)))
        if record["bits"] != bits.tolist() or record["needle_tokens"] != LIMIT:
            raise AssertionError("Device differed from the long-context Boolean reference")
        long_cases.append(dict(position=positions[location], clean=row["expected"],
                               category_label=row["category_label"], action_label=row["action_label"],
                               with_distractor=record["result"], total_ms=record["total_ms"]))

    output = dict(backend=engine.target, lane=engine.lane, openvino_version=engine.openvino_version,
                  onnx_sha256=engine.info["onnx_sha256"], load_ms=engine.load_ms,
                  samples_sha256=hashlib.sha256(samples_bytes).hexdigest(), examples=len(rows),
                  exact_device_agreement=len(rows), correct_category=correct_category,
                  proposals=proposals, correct_proposal=correct_proposal,
                  single_utterance_total_ms=quantiles(timing), long_context_tokens=LIMIT,
                  corpus_sha256=CORPUS_SHA, distractor_only=base["result"], long_cases=long_cases,
                  long_matches_clean=sum(case["clean"] == case["with_distractor"] for case in long_cases))
    if args.scoped:
        profiles = Profiles()
        try:
            query_only = AdaptiveRuntime({"request": lambda: engine}, profiles, scopes={"request": "evidence"})
            records = []
            for i, row in enumerate(selected):
                body = engine.tokenizer.decode(source[1800 * i:1800 * i + LIMIT - 10000])
                document = json_document(engine.tokenizer, " ".join(body.split()), row["text"])
                records.append({"id": i, "text": document, "expected": row["expected"]})
            scoped = query_only.batch({"vertical": "request", "documents": records})
            proposals = [(row, decision) for row, decision in zip(selected, scoped["documents"])
                         if decision["result"]["action"] is not None]
            assert all(decision["needle_tokens"] == LIMIT for decision in scoped["documents"])
            output["scoped_json"] = {
                "sample_count": len(selected),
                "matches_clean": sum(d["feedback"]["joint"] for d in scoped["documents"]),
                "category_correct": sum(d["result"]["category"] == r["category_label"]
                                        for r, d in zip(selected, scoped["documents"])),
                "action_proposed": len(proposals),
                "action_correct": sum(d["result"]["action"] == r["action_label"] for r, d in proposals),
            }
        finally:
            profiles.close()
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf8")
    print(json.dumps({key: value for key, value in output.items() if key != "long_cases"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--toolchain", type=Path)
    parser.add_argument("--lane", choices=["ncs1", "ncs2"], default="ncs1")
    parser.add_argument("--target")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scoped", action="store_true", help="Also replay with the query in a declared JSON field")
    main(parser.parse_args())
