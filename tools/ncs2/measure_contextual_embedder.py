"""Measure contextual retrieval, CPU agreement, and request latency on NCS2."""

from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import platform
from time import perf_counter_ns

import numpy as np

from apps.static_embedder import StaticEncoder, fit_lexical_weights, lexical_counts
from apps.static_embedder.contextual import VARIANTS, grouped_features
from tools.ncs2.measure_static_embedder import timing_summary

VARIANT_NAMES = ("baseline", *VARIANTS)
DATASETS = ("nfcorpus", "scifact", "hotpot")
DISCOUNT = 1 / np.log2(np.arange(2, 12))


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def write(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temporary.replace(path)


def relevance(fixture, scores):
    docs = sorted((row["id"], i) for i, row in enumerate(fixture["rows"]) if row["kind"] != "query")
    queries = [row["id"] for row in fixture["rows"] if row["kind"] == "query"]
    relevance = defaultdict(dict)
    for row in fixture["qrels"]:
        relevance[row["query-id"]][row["corpus-id"]] = int(row.get("score", 1))
    scores = scores[:, [i for _, i in docs]]
    if len(scores) != len(queries):
        raise ValueError("Retrieval scores do not match the query manifest")
    ranking = np.argsort(-scores, axis=1, kind="stable")[:, :10]
    ndcg, recall = {}, {}
    for query, order in zip(queries, ranking):
        labels = relevance[query]
        gains = sorted((gain for gain in labels.values() if gain > 0), reverse=True)[:10]
        if not gains:
            raise ValueError(f"No positive judgments for {query}")
        actual = [labels.get(docs[i][0], 0) for i in order]
        ndcg[query] = float((DISCOUNT[:len(actual)] @ actual) / (DISCOUNT[:len(gains)] @ gains))
        recall[query] = sum(gain > 0 for gain in actual) / sum(gain > 0 for gain in labels.values())
    return {"ndcg_at_10": float(np.mean(list(ndcg.values()))),
            "recall_at_10": float(np.mean(list(recall.values()))),
            "per_query_ndcg": ndcg, "queries": len(queries), "documents": len(docs)}


def retrieval(fixture, vectors):
    docs = sum(row["kind"] != "query" for row in fixture["rows"])
    return relevance(fixture, vectors[docs:] @ vectors[:docs].T)


def industry_scores(encoder, examples, reference=None):
    texts = sorted({row[field] for row in examples for field in ("query", "positive", "negative")})
    values = np.stack([encoder.encode(text) for text in texts])
    vectors = dict(zip(texts, values))
    rows = {}
    for row in examples:
        q = vectors[row["query"]]
        p, n = vectors[row["positive"]], vectors[row["negative"]]
        margin = float(q @ p - q @ n)
        rows[row["id"]] = {"correct": margin > 0, "margin": margin,
                           "family": row["family"], "vertical": row["vertical"],
                           "phenomenon": row["phenomenon"]}
    report = {"pair_accuracy": float(np.mean([row["correct"] for row in rows.values()])),
              "ties": sum(row["margin"] == 0 for row in rows.values()),
              "by_vertical": {name: float(np.mean([r["correct"] for r in rows.values() if r["vertical"] == name]))
                              for name in sorted({r["vertical"] for r in rows.values()})},
              "by_phenomenon": {name: float(np.mean([r["correct"] for r in rows.values()
                                                     if r["phenomenon"] == name]))
                               for name in sorted({r["phenomenon"] for r in rows.values()})},
              "rows": rows}
    if reference:
        report["cpu_agreement"] = compare_vectors(values, np.stack([reference.encode(text) for text in texts]))
    return report


def compare_vectors(actual, expected):
    norms = np.linalg.norm(actual, axis=1)
    cosine = np.einsum("ij,ij->i", actual, expected) / (norms * np.linalg.norm(expected, axis=1))
    return {"worst_cosine": float(cosine.min()), "max_norm_error": float(np.abs(norms - 1).max()),
            "vectors": len(actual)}


def edge_checks(device, cpu, contextual):
    reversal = ("Inspect the tag before you clear the hold.", "Clear the hold before you inspect the tag.")
    long = "review " * 128
    texts = [*reversal, "8", "🎱", long, long + " irrelevant suffix"]
    vectors = [device.encode(text) for text in texts]
    result = compare_vectors(np.stack(vectors), np.stack([cpu.encode(text) for text in texts]))
    result["truncation_max_abs"] = float(np.max(np.abs(vectors[-1] - vectors[-2])))
    result["reversal_distance"] = float(np.linalg.norm(vectors[0] - vectors[1]))
    try:
        device.encode(" ")
    except ValueError:
        result["blank_rejected"] = True
    else:
        result["blank_rejected"] = False
    if contextual:
        ids = device.tokenizer.encode("tag", add_special_tokens=False).ids
        features, mask = grouped_features(ids)
        original = device.infer(ids, lexical_counts("tag"), (features, mask))
        features[mask[:, 0] == 0, :3] = 1
        perturbed = device.infer(ids, lexical_counts("tag"), (features, mask))
        result["padding_max_abs"] = float(np.max(np.abs(original - perturbed)))
    return result


def timings(encoder, fixture, texts):
    ordered = []
    for kind in ("query", "document"):
        eligible = (i for i, row in enumerate(fixture["rows"]) if row["kind"] == kind)
        ordered.append(sorted(eligible, key=lambda i: sha256(fixture["rows"][i]["id"].encode()).digest())[:50])
    workload = [texts[i] for pair in zip(*ordered) for i in pair]
    if len(workload) != 100:
        raise ValueError("Expected 50 queries and 50 documents for latency measurement")
    for text in workload[:20]:
        encoder.encode(text)
    runs = []
    for _ in range(3):
        samples = []
        for text in workload:
            start = perf_counter_ns()
            encoder.encode(text)
            samples.append((perf_counter_ns() - start) / 1e6)
        runs.append(timing_summary(samples))
    return runs


def fixtures(root, static, dataset):
    source = root if dataset == "nfcorpus" else static
    fixture = read(source / f"{dataset}-evaluation.json")
    texts = [row["text"] for row in fixture["rows"]]
    if dataset == "nfcorpus":
        texts = [f'{row["title"]}. {row["text"]}' if row.get("title") else row["text"]
                 for row in fixture["rows"]]
    return fixture, texts


def make_encoder(static, lexical, adapter, backend, shaves, profile=False):
    return StaticEncoder(static / "tokenizer.json", static / "base1024-fp16.npy",
                         lexical_path=lexical, context_path=adapter, backend=backend, shaves=shaves,
                         profile=profile)


def run_dataset(root, static, dataset, adapter, shaves, *, backend):
    fixture, texts = fixtures(root, static, dataset)
    lexical = root / f"{dataset}-lexical.npy"
    docs = sum(row["kind"] != "query" for row in fixture["rows"])
    np.save(lexical, fit_lexical_weights(texts[:docs]), allow_pickle=False)
    started = perf_counter_ns()
    encoder = make_encoder(static, lexical, adapter, backend, shaves, profile=dataset == "nfcorpus")
    load_ms = (perf_counter_ns() - started) / 1e6
    result = {"load_ms": load_ms, "runtime": encoder.identity}
    actual = np.stack([encoder.encode(text) for text in texts])
    result["retrieval"] = retrieval(fixture, actual)
    if backend == "myriad":
        reference = make_encoder(static, lexical, adapter, "cpu", shaves)
        expected = np.stack([reference.encode(text) for text in texts])
        result["cpu_agreement"] = compare_vectors(actual, expected)
        result["cpu_retrieval"] = retrieval(fixture, expected)
        if dataset == "nfcorpus":
            result["edge_checks"] = edge_checks(encoder, reference, adapter is not None)
    if dataset == "nfcorpus":
        result["latency_ms"] = timings(encoder, fixture, texts)
    return result


def run_industry(root, static, adapter, shaves, backend):
    pairs = read(root / "pairs.json")["industry_test"]
    documents = {text for row in pairs for text in (row["positive"], row["negative"])}
    lexical = root / "industry-lexical.npy"
    np.save(lexical, fit_lexical_weights(sorted(documents)), allow_pickle=False)
    encoder = make_encoder(static, lexical, adapter, backend, shaves)
    reference = make_encoder(static, lexical, adapter, "cpu", shaves) if backend == "myriad" else None
    return industry_scores(encoder, pairs, reference)


def interval(a, b, clustered=None):
    ids = sorted(a)
    delta = np.asarray([a[key] - b[key] for key in ids])
    if clustered:
        groups = defaultdict(list)
        for index, key in enumerate(ids):
            groups[clustered[key]].append(index)
        delta = np.array([delta[indices].mean() for indices in groups.values()])
    sampled = np.random.default_rng(20261124).choice(delta, (10000, len(delta)), replace=True)
    return np.percentile(sampled.mean(axis=1), [2.5, 97.5]).tolist()


def metric(run, dataset):
    return run["datasets"][dataset]["retrieval"]["ndcg_at_10"]


def physical_pass(run):
    checks = [row["cpu_agreement"] for row in run["datasets"].values()]
    checks.append(run["industry"]["cpu_agreement"])
    primary = run["datasets"]["nfcorpus"]
    edges = primary["edge_checks"]
    return (all(c["worst_cosine"] >= .999 and c["max_norm_error"] <= .005 for c in [*checks, edges])
            and edges["blank_rejected"] and edges["truncation_max_abs"] <= .0001
            and edges.get("padding_max_abs", 0) <= .0001
            and all(row["p95"] <= 25 for row in primary["latency_ms"]))


def summarize(root, static):
    runs = {name: read(root / f"context-{name}-myriad.json") for name in VARIANT_NAMES}
    fixture, _ = fixtures(root, static, "nfcorpus")
    teacher = retrieval(fixture, np.load(root / "nfcorpus-teacher.npy", allow_pickle=False))
    baseline = runs["baseline"]
    baseline_public = baseline["datasets"]["nfcorpus"]["retrieval"]["per_query_ndcg"]
    baseline_industry = {key: int(row["correct"]) for key, row in baseline["industry"]["rows"].items()}
    result = {
        "schema": "myriadforge.ncs2-contextual-results/1",
        "teacher": teacher["ndcg_at_10"],
        "public_dataset": {"queries": teacher["queries"], "documents": teacher["documents"]},
        "industry_pairs": len(baseline_industry),
        "input_sha256": {name: sha256((root / name).read_bytes()).hexdigest()
                         for name in ("squad.json", "nfc-corpus.parquet", "nfc-queries.parquet", "nfc-test.tsv")},
        "device": baseline["datasets"]["nfcorpus"]["runtime"]["device"],
        "runtime": baseline["datasets"]["nfcorpus"]["runtime"]["runtime"],
        "shaves": baseline["datasets"]["nfcorpus"]["runtime"]["shaves"],
        "training": read(root / "training.json"),
        "variants": {},
    }
    for variant, run in runs.items():
        public = metric(run, "nfcorpus")
        primary = run["datasets"]["nfcorpus"]
        row_scores = primary["retrieval"]["per_query_ndcg"]
        industry = run["industry"]["pair_accuracy"]
        pairs = {key: int(row["correct"]) for key, row in run["industry"]["rows"].items()}
        families = {key: row["family"] for key, row in run["industry"]["rows"].items()}
        teacher_ci = interval(row_scores, teacher["per_query_ndcg"])
        result["variants"][variant] = {
            "public_ndcg": public, "public_delta": public - metric(baseline, "nfcorpus"),
            "public_delta_95": interval(row_scores, baseline_public),
            "teacher_delta": public - teacher["ndcg_at_10"], "teacher_delta_95": teacher_ci,
            "parity": teacher_ci[0] >= -.03,
            "industry_accuracy": industry, "industry_delta": industry - baseline["industry"]["pair_accuracy"],
            "industry_delta_95": interval(pairs, baseline_industry, families),
            "industry_ties": run["industry"]["ties"],
            "industry_by_vertical": run["industry"]["by_vertical"],
            "industry_by_phenomenon": run["industry"]["by_phenomenon"],
            "regression_deltas": {name: metric(run, name) - metric(baseline, name) for name in DATASETS[1:]},
            "p95_ms": [row["p95"] for row in primary["latency_ms"]],
            "timing_samples_ms": [row["samples"] for row in primary["latency_ms"]],
            "edge_checks": primary["edge_checks"],
            "compiled_blob_bytes": primary["runtime"]["compiled_blob_bytes"],
            "adapter_bytes": run["adapter_bytes"],
            "worst_cosine": min(row["cpu_agreement"]["worst_cosine"]
                                for row in [*run["datasets"].values(), run["industry"]]),
            "max_norm_error": max(row["cpu_agreement"]["max_norm_error"]
                                  for row in [*run["datasets"].values(), run["industry"]]),
            "physical_pass": physical_pass(run),
        }
    rows = result["variants"]
    for row in rows.values():
        row["qualifies"] = (row["physical_pass"] and row["public_delta"] >= .01
                            and row["industry_delta"] >= .05
                            and min(row["regression_deltas"].values()) >= -.01)
    one, tied = rows["one"], rows["tied"]
    public_gain = tied["public_ndcg"] - one["public_ndcg"]
    industry_gain = tied["industry_accuracy"] - one["industry_accuracy"]
    result["decision"] = "baseline"
    if tied["qualifies"] and public_gain >= 0 and industry_gain >= 0 and (public_gain >= .005 or industry_gain >= .03):
        result["decision"] = "tied"
    elif one["qualifies"]:
        result["decision"] = "one"
    write(root / "context-summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("static_assets", type=Path)
    parser.add_argument("--variant", choices=VARIANT_NAMES, default="baseline")
    parser.add_argument("--backend", choices=("cpu", "myriad"), default="myriad")
    parser.add_argument("--shaves", type=int, default=16)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--retime", action="store_true", help="Refresh latency and industry checks of a saved report")
    args = parser.parse_args()
    root, static = args.directory, args.static_assets
    if args.summarize:
        summarize(root, static)
        return
    adapter = root / f"context-{args.variant}.npz" if args.variant != "baseline" else None
    report_path = root / f"context-{args.variant}-{args.backend}.json"
    if args.retime:
        report = read(report_path)
        fixture, texts = fixtures(root, static, "nfcorpus")
        encoder = make_encoder(static, root / "nfcorpus-lexical.npy", adapter, args.backend,
                               args.shaves, profile=True)
        primary = report["datasets"]["nfcorpus"]
        identity = {key: value for key, value in encoder.identity.items() if key != "compiled_blob_bytes"}
        if any(primary["runtime"].get(key) != value for key, value in identity.items()):
            raise ValueError("Retime requires the same model and device allocation as the saved report")
        primary["latency_ms"] = timings(encoder, fixture, texts)
        if args.backend == "myriad":
            reference = make_encoder(static, root / "nfcorpus-lexical.npy", adapter, "cpu", args.shaves)
            primary["edge_checks"] = edge_checks(encoder, reference, adapter is not None)
        del encoder
        report["industry"] = run_industry(root, static, adapter, args.shaves, args.backend)
        write(report_path, report)
        print(args.variant, [row["p95"] for row in primary["latency_ms"]], flush=True)
        return
    report = {"variant": args.variant, "backend": args.backend, "platform": platform.platform(),
              "adapter_bytes": adapter.stat().st_size if adapter else 0, "datasets": {}}
    for dataset in DATASETS:
        values = run_dataset(root, static, dataset, adapter, args.shaves, backend=args.backend)
        report["datasets"][dataset] = values
        print(args.variant, dataset, values["retrieval"]["ndcg_at_10"], flush=True)
    report["industry"] = run_industry(root, static, adapter, args.shaves, args.backend)
    print(args.variant, "industry", report["industry"]["pair_accuracy"], flush=True)
    write(report_path, report)


if __name__ == "__main__":
    main()
