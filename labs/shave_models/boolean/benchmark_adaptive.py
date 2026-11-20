"""Compare fixed and adaptive lanes on unequal document structures with known decisions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from adaptive import AdaptiveRuntime, Profiles
from benchmark import CORPUS_SHA, exact_document, expected, quantiles
from policy import LIMIT, count_tokens
from runtime import Runtime


def json_document(tokenizer, body, evidence):
    header = json.dumps({"evidence": evidence, "detail": body}, ensure_ascii=False)[:-1] + ', "pad": "'
    trailer = '"}'
    gap = LIMIT - len(tokenizer.encode(header + trailer))
    if gap < 0:
        raise AssertionError("Need more room for JSON escapes")
    document = header + " a" * gap + trailer
    difference = LIMIT - len(tokenizer.encode(document))
    document = header + " a" * (gap + difference) + trailer
    assert len(tokenizer.encode(document)) == LIMIT
    json.loads(document)
    return document


def fixtures(tokenizer, source, rounds):
    entries = [
        ("duplicate charge. review authorized", [1, 4]),
        ("parcel missing. review authorized. review cancelled", [3, 4, 6]),
        ("charged twice. package missing. review authorized", [0, 2, 4]),
        ("account discussion only", []),
    ]
    for index in range(rounds):
        for vertical in ("support_notes", "case_records"):
            shift = 2400 * index + (1100 if vertical == "case_records" else 0)
            phrase, facts = entries[index % len(entries)]
            if vertical == "support_notes":
                text = exact_document(tokenizer, source[shift:], LIMIT, end=phrase)
            else:
                body = tokenizer.decode(source[shift:shift + LIMIT - 10000])
                body = " ".join(body.split())
                text = json_document(tokenizer, body, phrase)
            yield vertical, index, text, expected(facts)


def fixed_decision(engine, text, scope):
    if scope is None:
        return engine.decide(text)
    evidence = json.loads(text)[scope]

    def prepare(document):
        count = count_tokens(engine.tokenizer, document, fast=True)
        counted = time.perf_counter()
        return count, engine.policy.text_atoms(evidence), counted

    return engine.decide(text, prepare=prepare)


def measure(engine, corpus_path, batches):
    corpus = corpus_path.read_bytes()
    if hashlib.sha256(corpus).hexdigest() != CORPUS_SHA:
        raise ValueError("Unexpected benchmark corpus")
    source = engine.tokenizer.encode(corpus.decode("utf-8-sig"))
    profiles = Profiles()
    scopes = {"case_records": "evidence"}
    batcher = AdaptiveRuntime({"support_notes": lambda: engine, "case_records": lambda: engine},
                              profiles, scopes=scopes)
    results = {"support_notes": [], "case_records": []}
    # Two unique full-context documents per vertical in each batch.
    pending = {vertical: [] for vertical in results}
    try:
        for vertical, index, text, want in fixtures(engine.tokenizer, source, 2 * batches):
            pending[vertical].append((text, want))
            if len(pending[vertical]) < 2:
                continue
            pairs, pending[vertical] = pending[vertical], []
            baseline, adaptive = {}, {}
            request = {"vertical": vertical, "documents": [
                {"text": text, "expected": gold} for text, gold in pairs
            ]}
            # Swap measurement order across batches to reduce ordering effects.
            for strategy in (("fixed", "adaptive") if index % 4 == 1 else ("adaptive", "fixed")):
                start = time.perf_counter()
                if strategy == "fixed":
                    baseline["records"] = [fixed_decision(engine, text, scopes.get(vertical)) for text, _ in pairs]
                    baseline["wall_ms"] = (time.perf_counter() - start) * 1000
                else:
                    adaptive = batcher.batch(request)
                    adaptive["wall_ms"] = (time.perf_counter() - start) * 1000
            for (_, gold), fixed, routed in zip(pairs, baseline["records"], adaptive["documents"]):
                if fixed["result"] != gold or routed["result"] != gold or routed["needle_tokens"] != LIMIT:
                    raise AssertionError("Unexpected result on the held-out fixture")
            lanes = Counter()
            for profile in adaptive["routes"]:
                lanes.update(profile["used"])
                if profile["quarantined"]:
                    raise AssertionError("Unexpected route quarantine on valid data")
            results[vertical].append(dict(
                documents=len(pairs), fixed_ms=baseline["wall_ms"], adaptive_ms=adaptive["wall_ms"],
                fixed_device_requests=sum(row["requests"] for row in baseline["records"]),
                adaptive_device_requests=adaptive["device_requests"],
                lanes=dict(lanes), fixture_sha256=[hashlib.sha256(text.encode("utf8")).hexdigest() for text, _ in pairs]
            ))
    finally:
        profiles.close()
    stats = {}
    for name, rows in results.items():
        counts = Counter()
        for row in rows:
            counts.update(row["lanes"])
        stats[name] = dict(
            documents=sum(row["documents"] for row in rows), lanes=dict(counts),
            fixed_device_requests=sum(row["fixed_device_requests"] for row in rows),
            adaptive_device_requests=sum(row["adaptive_device_requests"] for row in rows),
            fixed_ms=quantiles([row["fixed_ms"] for row in rows]),
            adaptive_ms=quantiles([row["adaptive_ms"] for row in rows]),
        )
    return dict(backend=engine.target, lane=engine.lane, model=engine.model_name,
                max_tokens=LIMIT, batch_size=2, scopes=scopes,
                corpus_sha256=CORPUS_SHA, stats=stats, batches=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--blob", type=Path)
    parser.add_argument("--toolchain", type=Path)
    parser.add_argument("--lane", choices=("ncs1", "ncs2"), default="ncs1")
    parser.add_argument("--target")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.batches <= 14:
        parser.error("The pinned corpus supports 1–14 batches at each structure")
    runtime = Runtime(args.bundle, args.toolchain, lane=args.lane, target=args.target, blob=args.blob)
    report = measure(runtime, args.corpus, args.batches)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report["stats"], indent=2))
