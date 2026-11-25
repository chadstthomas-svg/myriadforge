"""Encode text or score a catalog with a CPU or hybrid EmbeddingGemma 2 lane."""

from argparse import ArgumentParser
import base64
from contextlib import contextmanager, nullcontext
from hashlib import sha256
import json
import os
from pathlib import Path
from queue import Empty, Queue
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
from time import perf_counter_ns

import numpy as np

from tools.ncs2.worker_process import stop_worker

MARKER = "MYRIAD_EMBEDDING "
DIMENSIONS = (128, 256, 512, 768)
ROOT = Path(__file__).resolve().parents[2]


class Myriad:
    def __init__(self, python, weights, block, shaves):
        self.block = block
        self.requests = []
        self.outputs = {name: value.shape[0] for name, value in weights.items()}
        self.work = TemporaryDirectory(prefix="embeddinggemma2-")
        bundle = Path(self.work.name) / "projections.npz"
        self.process = None
        self.messages = Queue()
        try:
            np.savez(bundle, **weights)
            command = [str(python), "-u", str(ROOT / "tools/ncs2/embeddinggemma2_worker.py"),
                       str(bundle), "--block", str(block), "--shaves", str(shaves)]
            # Keep host-only packages out of the Python 3.10 OpenVINO worker.
            env = {key: value for key, value in os.environ.items() if key.upper() != "PYTHONPATH"}
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            text=True, encoding="utf8", errors="replace", env=env)
            self.reader = Thread(target=self.read_replies, daemon=True)
            self.reader.start()
            self.identity = self.receive(120)
            if self.identity["status"] != "ready":
                raise RuntimeError("NCS2 worker did not become ready")
        except BaseException:
            self.close()
            raise

    def read_replies(self):
        for line in self.process.stdout:
            if line.startswith(MARKER):
                self.messages.put(line[len(MARKER):])
        self.messages.put(None)

    def receive(self, timeout):
        try:
            message = self.messages.get(timeout=timeout)
        except Empty as error:
            raise RuntimeError("NCS2 projection worker timed out") from error
        if message is None:
            raise RuntimeError("NCS2 projection worker exited")
        result = json.loads(message)
        if result["status"] == "error":
            raise RuntimeError(result["detail"])
        return result

    def project(self, name, matrix):
        start = perf_counter_ns()
        output = []
        for offset in range(0, len(matrix), self.block):
            block = np.ascontiguousarray(matrix[offset:offset + self.block], dtype="<f4")
            body = {"stage": name, "rows": len(block), "data": base64.b64encode(block).decode("ascii")}
            self.process.stdin.write(json.dumps(body) + "\n")
            self.process.stdin.flush()
            response = self.receive(30)
            values = np.frombuffer(base64.b64decode(response["data"], validate=True), dtype="<f2").astype(np.float32)
            output.append(values.reshape(len(block), self.outputs[name]))
            self.requests.append({"stage": name, "rows": len(block), "device_ms": response["device_ms"]})
        self.requests[-1]["call_wall_ms"] = (perf_counter_ns() - start) / 1e6
        return np.concatenate(output)

    def close(self):
        if self.process is not None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=4)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                stop_worker(self.process)
                self.process.wait()
            if hasattr(self, "reader"):
                self.reader.join(timeout=2)
            self.process.stdout.close()
        self.work.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def prompt(row):
    if not isinstance(row.get("text"), str) or not row["text"].strip():
        raise ValueError("Every input row needs nonempty text")
    role = row.get("kind", "document")
    if role == "query":
        return "task: search result | query: " + row["text"]
    if role == "document":
        title = row.get("title") or "none"
        return f"title: {title} | text: {row['text']}"
    raise ValueError("kind must be query or document")


def normalize(values, dim):
    values = np.asarray(values[:, :dim], dtype=np.float32)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all() or not np.isfinite(norm).all() or (norm == 0).any():
        raise ArithmeticError("Cannot normalize zero or non-finite embeddings")
    return values / norm


def rank(rows, embeddings, top_k):
    docs = [i for i, row in enumerate(rows) if row.get("kind", "document") == "document"]
    results = {}
    for i, row in enumerate(rows):
        if row.get("kind") != "query":
            continue
        scores = embeddings[docs] @ embeddings[i]
        order = np.argsort(-scores, kind="stable")[:top_k]
        results[row["id"]] = [{"id": rows[docs[j]]["id"], "cosine": float(scores[j])} for j in order]
    return results


class Encoder:
    def __init__(self, checkpoint, threads):
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(threads)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        self.model = AutoModel.from_pretrained(checkpoint, vision_config=None, audio_config=None,
                                               dtype=torch.float32, local_files_only=True).eval()
        if self.model.config.model_type != "embedding_gemma2":
            raise ValueError("Supply a Google EmbeddingGemma 2 checkpoint")
        self.backbone = self.model.language_model

    def weights(self, mode):
        head = self.backbone.embedding_projection
        if mode == "cpu":
            return {}
        stages = {"head": head}
        if mode == "ple-head":
            stages["ple"] = self.backbone.ple.per_layer_model_projection
        if any(module.bias is not None for module in stages.values()):
            raise ValueError("Only bias-free projection offload is supported")
        return {name: module.weight.detach().cpu().float().numpy() for name, module in stages.items()}

    @contextmanager
    def offload(self, worker):
        torch = self.torch
        head = self.backbone.embedding_projection
        ple = self.backbone.ple.per_layer_model_projection

        class PerLayerProjection(torch.nn.Module):
            def forward(self, values):
                shape = values.shape[:-1]
                array = values.detach().float().numpy().reshape(-1, values.shape[-1])
                result = worker.project("ple", array)
                return torch.from_numpy(result).reshape(*shape, result.shape[-1]).to(values.dtype)

        self.backbone.embedding_projection = torch.nn.Identity()
        if "ple" in worker.outputs:
            self.backbone.ple.per_layer_model_projection = PerLayerProjection()
        try:
            yield
        finally:
            self.backbone.embedding_projection = head
            self.backbone.ple.per_layer_model_projection = ple

    def encode(self, texts, max_tokens, batch, worker=None):
        result = []
        with self.torch.inference_mode(), self.offload(worker) if worker else nullcontext():
            for offset in range(0, len(texts), batch):
                inputs = self.tokenizer(texts[offset:offset + batch], padding=True, truncation=True,
                                        max_length=max_tokens, return_tensors="pt")
                mask = inputs["attention_mask"].unsqueeze(-1)
                values = self.model(input_ids=inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"]).last_hidden_state.float()
                pooled = ((values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)).numpy()
                result.append(worker.project("head", pooled) if worker else pooled)
        return np.concatenate(result)


def timed(encoder, texts, args, worker=None):
    encoder.encode(texts, args.max_tokens, args.batch, worker)
    durations = []
    for _ in range(args.repeats):
        start = perf_counter_ns()
        values = encoder.encode(texts, args.max_tokens, args.batch, worker)
        durations.append((perf_counter_ns() - start) / 1e6)
    return values, durations


def compare(rows, values, control):
    evidence = {}
    labels = {row["id"]: row["expected"] for row in rows if row.get("kind") == "query" and "expected" in row}
    for dim in DIMENSIONS:
        a, b = normalize(values, dim), normalize(control, dim)
        scores = np.clip(np.sum(a * b, axis=1), -1, 1)
        hybrid, reference = rank(rows, a, 1), rank(rows, b, 1)
        matched = sum([hit["id"] for hit in hybrid[key]] == [hit["id"] for hit in hits]
                      for key, hits in reference.items() if hits)
        evidence[dim] = {"min_embedding_cosine": float(scores.min()), "mean_embedding_cosine": float(scores.mean()),
                         "ranking_top1_matches": matched, "queries_with_catalog": sum(bool(hits) for hits in reference.values()),
                         "labeled_top1_matches": sum(bool(hybrid[key]) and hybrid[key][0]["id"] == label
                                                     for key, label in labels.items()),
                         "labeled_queries": len(labels),
                         "cpu_top1": reference, "hybrid_top1": hybrid}
    return evidence


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Local EmbeddingGemma 2 snapshot")
    parser.add_argument("input", type=Path, help="JSONL of ID, text, and optional kind and title")
    parser.add_argument("--mode", choices=("cpu", "head", "ple-head"), default="cpu")
    parser.add_argument("--ncs2-python", type=Path)
    parser.add_argument("--dimension", type=int, choices=DIMENSIONS, default=256)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--block", type=int, default=16, help="Static VPU projection row count")
    parser.add_argument("--shaves", type=int, choices=range(1, 17), default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch, args.block, args.threads, args.repeats, args.top_k) < 1 or not 1 <= args.max_tokens <= 8192:
        parser.error("Use positive batch, block, threads, repeats and top-k with 1 to 8192 max tokens")
    if args.mode != "cpu" and not args.ncs2_python:
        parser.error("Projection offload requires --ncs2-python")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        parser.error("JSONL rows must be objects")
    ids = [row.get("id") for row in rows]
    if not ids or any(not isinstance(key, str) or not key for key in ids) or len(ids) != len(set(ids)):
        parser.error("Supply rows with unique nonempty string IDs")
    texts = [prompt(row) for row in rows]
    encoder = Encoder(args.checkpoint, args.threads)
    benchmark = {}
    if args.compare and args.mode != "cpu":
        control, benchmark["cpu_samples_ms"] = timed(encoder, texts, args)
    else:
        control = None
    weights = encoder.weights(args.mode)
    with Myriad(args.ncs2_python, weights, args.block, args.shaves) if weights else nullcontext() as worker:
        if args.compare or args.repeats > 1:
            values, benchmark["selected_samples_ms"] = timed(encoder, texts, args, worker)
        else:
            start = perf_counter_ns()
            values = encoder.encode(texts, args.max_tokens, args.batch, worker)
            benchmark["selected_samples_ms"] = [(perf_counter_ns() - start) / 1e6]
        if worker:
            benchmark["ncs2"] = worker.identity
            benchmark["stages"] = {}
            for name in weights:
                records = [row for row in worker.requests if row["stage"] == name]
                benchmark["stages"][name] = {"learned_parameters": weights[name].size, "requests": len(records),
                                             "float32_sha256": sha256(weights[name].tobytes()).hexdigest(),
                                             "device_median_ms": median(row["device_ms"] for row in records),
                                             "call_wall_median_ms": median(row["call_wall_ms"] for row in records
                                                                           if "call_wall_ms" in row)}
    embedding = normalize(values, args.dimension)
    agreement = compare(rows, values, control) if control is not None else {}
    qualified = all(row["min_embedding_cosine"] >= .999 and row["ranking_top1_matches"] == row["queries_with_catalog"]
                    for row in agreement.values())
    report = {"schema": "myriadforge.embeddinggemma2/1", "mode": args.mode,
              "dimension": args.dimension, "text_count": len(rows),
              "host_text_parameters": sum(parameter.numel() for parameter in encoder.model.parameters()),
              "cpu_threads": args.threads, "batch": args.batch, "max_tokens": args.max_tokens,
              "device_block": args.block if worker else None, "device_shaves": args.shaves if worker else None,
              "benchmark": benchmark, "parity": agreement,
              "embeddings": dict(zip(ids, embedding.tolist())), "ranking": rank(rows, embedding, args.top_k)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")
    print(f"Saved {len(rows)} embeddings and retrieval results to {args.out}")
    return 0 if qualified else 1


if __name__ == "__main__":
    sys.exit(main())
