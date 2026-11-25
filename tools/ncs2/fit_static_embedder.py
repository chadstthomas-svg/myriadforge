"""Fit the compact projection and prepare the pinned retrieval comparison."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import pyarrow.parquet as pq
from safetensors import safe_open
from tokenizers import Tokenizer

from apps.embeddinggemma2.__main__ import Encoder, normalize, prompt
from apps.static_embedder import TOKEN_LIMIT

DIM = 256


def rows(directory, split):
    return pq.read_table(directory / (split + ".parquet")).to_pylist()


def to_row(row, kind):
    return {"id": row["_id"], "kind": kind, "text": row["text"]}


def seed_key(row):
    return sha256(row["_id"].encode()).digest()


def nonempty(rows):
    return [row for row in rows if isinstance(row["text"], str) and row["text"].strip()]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("teacher", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dataset", choices=("scifact", "hotpot"), default="scifact")
    parser.add_argument("--skip-fit", action="store_true")
    args = parser.parse_args()
    root = args.directory
    teacher = Encoder(args.teacher, args.threads)
    if not args.skip_fit:
        fit_student(root, teacher)
    prepare_evaluation(root, teacher, args.dataset, args.threads)


def fit_student(root, teacher):
    fit = sorted(nonempty(rows(root, "fiqa-corpus")), key=seed_key)[:256]
    fit_text = [row["text"] for row in fit]
    fit_prompts = [prompt(to_row(row, "document")) for row in fit]
    print("Encoding 256 FiQA training texts", flush=True)
    targets = normalize(teacher.encode(fit_prompts, TOKEN_LIMIT, 8), DIM // 2)

    with safe_open(root / "static-table.safetensors", framework="numpy") as source:
        pretrained = source.get_tensor("embedding.weight")
    base = pretrained[:, :DIM]
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=TOKEN_LIMIT)
    ids = [e.ids or [tokenizer.token_to_id("[UNK]")]
           for e in tokenizer.encode_batch(fit_text, add_special_tokens=False)]
    raw = np.stack([base[tokens].mean(axis=0) for tokens in ids])
    x = normalize(raw, DIM).astype(np.float64)
    projection = np.linalg.solve(x.T @ x + 0.01 * np.eye(DIM), x.T @ targets)
    projected_norm = np.linalg.norm(raw @ projection, axis=1)
    base_norm = np.linalg.norm(raw[:, :DIM // 2], axis=1)
    scale = np.median(base_norm) / np.median(projected_norm)
    table = np.concatenate([0.8 * base[:, :DIM // 2], 0.6 * scale * (base @ projection)], axis=1)
    np.save(root / "base256.npy", base.astype(np.float32), allow_pickle=False)
    np.save(root / "student256.npy", table.astype(np.float32), allow_pickle=False)
    np.save(root / "student256-fp16.npy", table.astype(np.float16), allow_pickle=False)
    np.save(root / "base1024-fp16.npy", pretrained.astype(np.float16), allow_pickle=False)
    (root / "fit.json").write_text(json.dumps({"ids": [row["_id"] for row in fit],
                                              "projection_scale": float(scale)}), encoding="utf8")
    print(f"Saved {table.size:,} parameters with calibrated projection scale {scale:.5f}", flush=True)


def prepare_evaluation(root, teacher, dataset, threads):
    qrels = rows(root, dataset + "-qrels")
    judged = {row["corpus-id"] for row in qrels}
    corpus = nonempty(rows(root, dataset + "-corpus"))
    positives = [row for row in corpus if row["_id"] in judged]
    if {row["_id"] for row in positives} != judged:
        raise ValueError("A judged document is missing")
    if len(positives) > 384:
        raise ValueError("The corpus sample cannot contain every judged document")
    negatives = sorted((row for row in corpus if row["_id"] not in judged), key=seed_key)
    docs = positives + negatives[:384 - len(positives)]
    queries = rows(root, dataset + "-queries")
    fixture = [to_row(row, "document") for row in docs] + [to_row(row, "query") for row in queries]
    prompts = [prompt(row) for row in fixture]
    print(f"Encoding {len(fixture)} held-out {dataset} texts", flush=True)
    baseline = teacher.encode(prompts, TOKEN_LIMIT, 8)
    np.save(root / (dataset + "-teacher.npy"), normalize(baseline, DIM), allow_pickle=False)
    np.save(root / (dataset + "-teacher768.npy"), normalize(baseline, 768), allow_pickle=False)
    timing_rows = list(range(12)) + list(range(len(docs), len(docs) + 12))
    teacher.encode([prompts[0]], TOKEN_LIMIT, 1)
    latencies = []
    for i in timing_rows:
        start = perf_counter_ns()
        normalize(teacher.encode([prompts[i]], TOKEN_LIMIT, 1), 768)
        latencies.append((perf_counter_ns() - start) / 1e6)
    context = {"rows": fixture, "qrels": qrels, "teacher_latencies_ms": latencies,
               "timing_rows": timing_rows, "teacher_threads": threads}
    (root / (dataset + "-evaluation.json")).write_text(json.dumps(context, ensure_ascii=False),
                                                      encoding="utf8")


if __name__ == "__main__":
    main()
