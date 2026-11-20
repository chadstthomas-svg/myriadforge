"""Measure the support policy end to end, including real Needle tokenization."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from policy import LIMIT, count_tokens
from runtime import Runtime


PHRASES = ("charged twice", "duplicate charge", "package missing", "parcel missing",
           "review authorized", "do not route", "review cancelled")
CORPUS_SHA = "48e0522844402a86a3ea98f0947ba85ea54838db3c59022299298ed968a5a163"


def expected(indices):
    present = set(indices)
    billing = bool(present & {0,1})
    delivery = bool(present & {2,3})
    kind = "mixed" if billing and delivery else "billing" if billing else "delivery" if delivery else None
    ready = bool(kind and 4 in present and not present & {5,6})
    actions = {"mixed":"triage_mixed","billing":"review_billing","delivery":"review_delivery"}
    return {"answers":{"may_route":ready}, "category":kind, "action":actions[kind] if ready else None}


def exact_document(tok, source_ids, size, start="", middle="", end=""):
    # Leave room for insertions, then fill the small difference before the
    # suffix. The final assertion measures the finished string independently.
    body = tok.decode(source_ids[:max(1, size-70)])
    halves = body.split(" ")
    cut = len(halves)//2
    prefix = " ".join([start, " ".join(halves[:cut]), middle, " ".join(halves[cut:])]).strip()
    suffix = " " + end if end else ""
    doc = prefix + suffix
    gap = size - len(tok.encode(doc, out_type=int))
    if gap < 0:
        raise ValueError("Too little reserved space in fixture")
    doc = prefix + " a"*gap + suffix
    if len(tok.encode(doc, out_type=int)) != size:
        raise AssertionError("Wrong fixture token count")
    return doc


def quantiles(values):
    return {k: float(np.percentile(values, v)) for k,v in (("p50",50),("p95",95),("max",100))}


def evaluate(engine, corpus, repeats, session, export_path=None):
    policy, tok = engine.policy, engine.tokenizer
    corpus_bytes = corpus.read_bytes()
    if hashlib.sha256(corpus_bytes).hexdigest() != CORPUS_SHA:
        raise ValueError("Unexpected benchmark corpus")
    text = corpus_bytes.decode("utf-8-sig")
    if any(policy.atoms(policy.project(text))):
        raise AssertionError("Reference text unexpectedly contains an evidence phrase")
    source = tok.encode(text, out_type=int)
    if len(source) < LIMIT:
        raise AssertionError("Need at least a full window of natural text")
    def check(document, indices):
        record = engine.decide(document)
        want_atoms = [float(" ".join(literal) in {PHRASES[i] for i in indices})
                      for literal in policy.literals]
        if record["result"] != expected(indices) or record["atoms"] != want_atoms:
            raise AssertionError((indices, record, want_atoms))
        if any(v not in (0,1) for v in record["bits"]):
            raise AssertionError("Graph returned a non-Boolean value")
        return record

    checks = 0
    for combination in itertools.product((False,True), repeat=len(PHRASES)):
        indices = [i for i,on in enumerate(combination) if on]
        check(" ; ".join(PHRASES[i] for i in indices), indices)
        checks += 1
    targeted = [
        ("", []), ("CHARGED TWICE. REVIEW AUTHORIZED.", [0,4]),
        ("duplicated charge; package apparently missing; preview authorized", []),
        ("duplicate;charge; review-cancelled", []),
        ("duplicate charge\nreview\tauthorized", [1,4]),
        ("package missing but do not route. review authorized.", [2,4,5]),
        ("Ångström 東京 charged twice 📃 review authorized", [0,4])]
    for where in [63, 4095, 8191]:
        targeted.append(("note "*where + "charged twice. review authorized", [0,4]))
        targeted.append(("note "*where + "do not route. parcel missing. review authorized", [5,3,4]))
    for doc, indices in targeted:
        check(doc,indices)

    long_cases = [
        ("early", "charged twice review authorized", "", "", [0,4]),
        ("middle", "", "charged twice review authorized", "", [0,4]),
        ("last", "", "", "charged twice review authorized", [0,4]),
        ("far_apart", "package missing", "", "review authorized", [2,4]),
        ("late_veto", "duplicate charge", "review authorized", "review cancelled", [1,4,6]),
        ("late_category", "parcel missing", "review authorized", "charged twice", [3,4,0]),
        ("absent", "", "", "", [])]
    cases = {}
    for label, first, mid, last, indices in long_cases:
        doc = exact_document(tok, source, LIMIT, first, mid, last)
        result = check(doc, indices)
        cases[label] = dict(tokens=result["needle_tokens"], result=result["result"],
                            input_sha256=hashlib.sha256(doc.encode("utf8")).hexdigest())
    try:
        engine.decide(exact_document(tok, source, LIMIT+1))
    except ValueError as exc:
        if "maximum" not in str(exc):
            raise
    else:
        raise AssertionError("Over-limit input accepted")

    timing = {}
    for size in [1024, 32768, LIMIT]:
        doc = exact_document(tok, source, size, PHRASES[1], "", PHRASES[4])
        for _ in range(3):
            check(doc,[1,4])
        runs = []
        cpu_ms = []
        for _ in range(repeats):
            runs.append(check(doc,[1,4]))
            start = time.perf_counter()
            count_tokens(tok, doc, fast=engine.fast_count)
            atoms = (policy.text_atoms(doc) if engine.fast_count and engine.info["variant"] != "scan"
                     else policy.atoms(policy.project(doc)))
            cpu_result = policy.result(policy.reference(atoms))
            cpu_ms.append(1000*(time.perf_counter()-start))
            if cpu_result != expected([1,4]):
                raise AssertionError("CPU policy reference disagreed")
        data = {name:quantiles([r[name] for r in runs]) for name in
                ("total_ms","tokenization_ms","host_projection_ms","inference_path_ms")}
        data.update(cpu_reference_ms=quantiles(cpu_ms), requests=runs[0]["requests"],
                    lexemes=runs[0]["lexemes"], utf8_bytes=len(doc.encode("utf8")),
                    documents=repeats, tokens_per_second_p50=size*1000/data["total_ms"]["p50"])
        timing[str(size)] = data
    ir = engine.ir
    sizes = dict(onnx=(engine.folder/"model.onnx").stat().st_size, ir_xml=ir.stat().st_size,
                 ir_weights=ir.with_suffix(".bin").stat().st_size)
    with tempfile.TemporaryDirectory() as tmp:
        export = Path(tmp)/"myriad.blob"
        engine.export_blob(export)
        sizes["compiled_blob"] = export.stat().st_size
        if export_path is not None:
            export_path.write_bytes(export.read_bytes())
    return dict(session=session, variant=engine.info["variant"], fast_count=engine.fast_count,
                onnx_sha256=engine.info["onnx_sha256"],
                corpus_sha256=CORPUS_SHA, load_ms=engine.load_ms, devices=engine.devices,
                backend=engine.target, lane=engine.lane, openvino_version=engine.openvino_version,
                exhaustive_combinations=checks, adversarial_cases=len(targeted),
                context_cases=cases, rejected_131073=True, bytes=sizes, timing=timing)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle", type=Path)
    p.add_argument("--toolchain", type=Path)
    p.add_argument("--lane", choices=["ncs1", "ncs2"], default="ncs1")
    p.add_argument("--target")
    p.add_argument("--export", type=Path)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--session", type=int, default=1)
    p.add_argument("--fast-count", action="store_true")
    args = p.parse_args()
    if args.repeats < 1:
        p.error("--repeats must be positive")
    engine = Runtime(args.bundle, args.toolchain, fast_count=args.fast_count, lane=args.lane, target=args.target)
    r = evaluate(engine, args.corpus, args.repeats, args.session, args.export)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(r,indent=2)+"\n",encoding="utf8")
    print(args.bundle.name, r["bytes"], r["timing"][str(LIMIT)]["total_ms"])
