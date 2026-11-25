"""Compare a fitted static embedder with its CPU reference and the teacher."""

from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import perf_counter_ns

import numpy as np

from apps.static_embedder import StaticEncoder, TOKEN_LIMIT, fit_lexical_weights


def vectors(encoder, rows):
    return np.stack([encoder.encode(row["text"]) for row in rows])


def retrieval(rows, qrels, embeddings):
    relevant = defaultdict(set)
    for qrel in qrels:
        relevant[qrel["query-id"]].add(qrel["corpus-id"])
    docs = [i for i, row in enumerate(rows) if row.get("kind") != "query"]
    queries = [i for i, row in enumerate(rows) if row.get("kind") == "query"]
    scores = embeddings[queries] @ embeddings[docs].T
    discount = 1 / np.log2(np.arange(2, 12))
    ndcg, recall, rr, top1 = [], [], [], []
    for i, ranking in zip(queries, np.argsort(-scores, axis=1, kind="stable")[:, :10]):
        labels = relevant[rows[i]["id"]]
        hit = np.array([rows[docs[position]]["id"] in labels for position in ranking])
        first = np.flatnonzero(hit)
        ndcg.append(float(discount[:len(hit)] @ hit / discount[:len(labels)].sum()))
        recall.append(float(hit.sum() / len(labels)))
        rr.append(1 / int(first[0] + 1) if len(first) else 0)
        top1.append(bool(hit[0]))
    return {"ndcg_at_10": float(np.mean(ndcg)), "recall_at_10": float(np.mean(recall)),
            "mrr_at_10": float(np.mean(rr)), "top1_correct": sum(top1),
            "queries": len(queries), "documents": len(docs)}, np.array(ndcg)


def timing_summary(values):
    return {"n": len(values), "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)), "min": min(values), "max": max(values),
            "samples": values}


def speeds(encoder, rows):
    for row in rows[:5]:
        encoder.encode(row["text"])
    values = []
    for row in rows:
        start = perf_counter_ns()
        encoder.encode(row["text"])
        values.append((perf_counter_ns() - start) / 1e6)
    return timing_summary(values)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--shaves", type=int, default=16)
    parser.add_argument("--industry", type=Path)
    parser.add_argument("--dataset", choices=("scifact", "hotpot"), default="scifact")
    parser.add_argument("--variant", choices=("distilled", "static1024", "hybrid"), default="distilled")
    args = parser.parse_args()
    root = args.directory
    fixture = json.loads((root / (args.dataset + "-evaluation.json")).read_text(encoding="utf8"))
    rows, labels = fixture["rows"], fixture["qrels"]
    timing = [rows[i] for i in fixture["timing_rows"]]
    tokenizer = root / "tokenizer.json"
    fp16 = root / ("student256-fp16.npy" if args.variant == "distilled" else "base1024-fp16.npy")
    lexical = None
    if args.variant == "hybrid":
        lexical = root / (args.dataset + "-lexical4096.npy")
        counts = fit_lexical_weights([row["text"] for row in rows if row.get("kind") != "query"])
        np.save(lexical, counts, allow_pickle=False)
    base = StaticEncoder(tokenizer, root / "base256.npy")
    cpu = StaticEncoder(tokenizer, fp16, lexical_path=lexical)
    student = StaticEncoder(tokenizer, root / "student256.npy") if args.variant == "distilled" else cpu
    start = perf_counter_ns()
    device = StaticEncoder(tokenizer, fp16, backend="myriad", shaves=args.shaves, lexical_path=lexical)
    compile_ms = (perf_counter_ns() - start) / 1e6

    actual = vectors(device, rows)
    expected = vectors(student, rows)
    teacher = np.load(root / (args.dataset + "-teacher.npy"), allow_pickle=False)
    teacher_full = np.load(root / (args.dataset + "-teacher768.npy"), allow_pickle=False)
    metrics = {}
    for name, embedding in (("teacher_cpu_256d", teacher), ("teacher_cpu_768d", teacher_full),
                            ("base_cpu_256d", vectors(base, rows)),
                            ("student_cpu", expected), ("student_myriad", actual)):
        metrics[name], ndcg = retrieval(rows, labels, embedding)
        if name == "teacher_cpu_768d":
            teacher_ndcg = ndcg
    delta = ndcg - teacher_ndcg
    samples = np.random.default_rng(20261124).choice(delta, (10000, len(delta)), replace=True)
    norms = np.linalg.norm(actual, axis=1)
    cosines = np.einsum("ij,ij->i", actual, expected) / norms
    report = {
        "schema": "myriadforge.ncs2-static-embedder/1",
        "variant": args.variant,
        "dataset": args.dataset,
        "device": device.identity,
        "platform": platform.platform(),
        "max_tokens": TOKEN_LIMIT,
        "dimensions": int(actual.shape[1]),
        "compile_and_load_ms": compile_ms,
        "weights": {"parameters": int(cpu.table.size), "file_bytes": fp16.stat().st_size,
                    "sha256": digest(fp16), "tokenizer_sha256": digest(tokenizer)},
        "reference_agreement": {"worst_cosine": float(np.min(cosines)),
                                "max_norm_error": float(np.max(np.abs(norms - 1))),
                                "rows": len(rows)},
        "retrieval": metrics,
        "paired_ndcg_delta": {"myriad_minus_teacher_768d": float(np.mean(delta)),
                              "bootstrap_95": np.percentile(samples.mean(axis=1), [2.5, 97.5]).tolist()},
        "latency_ms": {"teacher_cpu_threads": fixture["teacher_threads"],
                       "teacher_cpu": timing_summary(fixture["teacher_latencies_ms"]),
                       "student_cpu": speeds(cpu, timing), "student_myriad": speeds(device, timing)},
    }
    if lexical:
        report["weights"]["lexical_file_bytes"] = lexical.stat().st_size
        report["weights"]["lexical_sha256"] = digest(lexical)
    if args.industry:
        industry = [json.loads(line) for line in args.industry.read_text(encoding="utf8").splitlines()]
        industry_labels = [{"query-id": row["id"], "corpus-id": row["expected"]}
                           for row in industry if row.get("kind") == "query"]
        report["racklab"] = {name: retrieval(industry, industry_labels, vectors(encoder, industry))[0]
                             for name, encoder in (("base_cpu", base), ("student_myriad", device))}
    result = json.dumps(report, indent=2, allow_nan=False)
    filename = f"{args.dataset}-{args.variant}-physical-{args.shaves}.json"
    (root / filename).write_text(result + "\n", encoding="utf8")
    print(result)


if __name__ == "__main__":
    main()
