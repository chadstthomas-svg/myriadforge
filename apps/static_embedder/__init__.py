"""Static text embeddings for a physical Intel NCS2 or a CPU reference."""

from collections import Counter
from hashlib import blake2s
import re
from pathlib import Path

import numpy as np

from .contextual import (CONTEXT_WEIGHT, WIDTH, graph_context, grouped_features, load_adapter,
                         model_fingerprint, numpy_context, unit)

TOKEN_LIMIT = 128
LEXICAL_BUCKETS = 4096
CHANNEL_SCALE = np.float32(1 / np.sqrt(2))
WORDS = re.compile(r"\b\w\w+\b")


def hashed_words(text):
    words = WORDS.findall(text.lower())[:TOKEN_LIMIT]
    for word, count in Counter(words).items():
        yield int.from_bytes(blake2s(word.encode(), digest_size=4).digest(), "little"), count


def lexical_counts(text):
    values = np.zeros(LEXICAL_BUCKETS, dtype=np.float32)
    for key, count in hashed_words(text):
        values[key % LEXICAL_BUCKETS] += np.sqrt(count) * (-1 if key >> 31 else 1)
    return values


def fit_lexical_weights(texts):
    df = np.zeros(LEXICAL_BUCKETS, dtype=np.int64)
    for text in texts:
        used = {key % LEXICAL_BUCKETS for key, _ in hashed_words(text)}
        df[list(used)] += 1
    return np.log1p(len(texts) / (df + 1)).astype(np.float32)


class StaticEncoder:
    def __init__(self, tokenizer_path, table_path, *, backend="cpu", shaves=16,
                 lexical_path=None, context_path=None, profile=False):
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=TOKEN_LIMIT)
        self.table = np.load(table_path, allow_pickle=False)
        if (self.table.ndim != 2 or self.table.shape[0] != self.tokenizer.get_vocab_size()
                or self.table.dtype not in (np.float16, np.float32) or not self.table.shape[1]
                or not np.isfinite(self.table).all()):
            raise ValueError("A finite embedding table matching the tokenizer is required")
        self.lexical = np.load(lexical_path, allow_pickle=False) if lexical_path else None
        if self.lexical is not None and (self.lexical.shape != (LEXICAL_BUCKETS,)
                or not np.isfinite(self.lexical).all() or np.any(self.lexical <= 0)):
            raise ValueError("Lexical weights must contain one positive finite value per bucket")
        if context_path and (self.lexical is None or self.table.shape[1] != 1024):
            raise ValueError("The contextual adapter requires the 1024-dimensional table and keyword weights")
        self.context = load_adapter(context_path) if context_path else None
        self.identity = {"device": "CPU (NumPy)"}
        if backend == "myriad":
            self.infer, self.identity = compile_myriad(self.table, shaves, self.lexical,
                                                       self.context, profile=profile)
        elif backend == "cpu":
            self.infer = self.encode_cpu_context if self.context else self.encode_cpu
        else:
            raise ValueError("Expected backend cpu or myriad")
        if self.context:
            self.identity["model_fingerprint"] = model_fingerprint(
                *(Path(p) for p in (tokenizer_path, table_path, lexical_path, context_path)))

    def encode(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if not ids:
            raise ValueError("Text must contain at least one token")
        lexical = lexical_counts(text) if self.lexical is not None else None
        values = (self.infer(ids, lexical, grouped_features(ids)) if self.context
                  else self.infer(ids, lexical))
        if not np.isfinite(values).all() or not np.any(values):
            raise ArithmeticError("The encoder produced an invalid vector")
        return values

    def encode_cpu(self, ids, lexical):
        values = self.table[ids].sum(axis=0, dtype=np.float32)
        norm = np.linalg.norm(values)
        if not np.isfinite(norm) or norm == 0:
            raise ArithmeticError("The token weights do not define a cosine vector")
        semantic = values / norm
        if lexical is None:
            return semantic
        weighted = lexical * self.lexical
        lexical_norm = np.linalg.norm(weighted)
        if lexical_norm == 0:
            return np.concatenate([semantic, weighted])
        return np.concatenate([semantic, weighted / lexical_norm]) * CHANNEL_SCALE

    def encode_cpu_context(self, ids, lexical, grouped):
        base = self.encode_cpu(ids, lexical)
        context = numpy_context(*grouped, self.context)
        return unit(np.concatenate([np.float32(np.sqrt(1 - CONTEXT_WEIGHT)) * base,
                                    np.float32(np.sqrt(CONTEXT_WEIGHT)) * context]))


def compile_myriad(table, shaves, lexical_weights, context_adapter=None, *, profile=False):
    from openvino.runtime import Core, Model, get_version, opset8 as op

    from myriadforge.myriad_x import allocation_properties, require_myriad_x

    core = Core()
    identity = {"device": require_myriad_x(core), "runtime": get_version(), "shaves": shaves}
    # A power-of-two rescaling conditions the pooling arithmetic in FP16.
    weights = np.ascontiguousarray(table.T / 1024, dtype=np.float32)
    counts = op.parameter([1, len(table)], np.float32, name="token_counts")
    pooled = op.matmul(counts, op.constant(weights), False, True)
    axes = op.constant(np.array([1], dtype=np.int32))
    output = op.normalize_l2(pooled, axes, 1e-8, "max")
    inputs = [counts]
    if lexical_weights is not None:
        lexical = op.parameter([1, LEXICAL_BUCKETS], np.float32, name="lexical_counts")
        weighted = op.multiply(lexical, op.constant(lexical_weights.astype(np.float32)))
        normalized = op.normalize_l2(weighted, axes, 1e-8, "max")
        output = op.normalize_l2(op.concat([output, normalized], 1), axes, 1e-8, "max")
        inputs.append(lexical)
    if context_adapter:
        context, group_inputs = graph_context(op, context_adapter)
        # Legacy MYRIAD lowers a scalar ScaleShift on rank-two vectors incorrectly.
        base = op.multiply(output, np.full((1, table.shape[1] + LEXICAL_BUCKETS),
                                           np.sqrt(1 - CONTEXT_WEIGHT), dtype=np.float32))
        context = op.multiply(context, np.full((1, WIDTH), np.sqrt(CONTEXT_WEIGHT), dtype=np.float32))
        output = op.normalize_l2(op.concat([base, context], 1), axes, 1e-8, "max")
        inputs.extend(group_inputs)
    model = Model([output], inputs, "static_embedder")
    executable = core.compile_model(model, "MYRIAD", allocation_properties(shaves, shaves, 1))
    if profile:
        identity["compiled_blob_bytes"] = len(executable.export_model())
    request = executable.create_infer_request()

    def infer(ids, lexical, grouped=None):
        values = np.bincount(ids, minlength=len(table)).astype(np.float32)[None, :]
        inputs = {executable.input(0): values}
        if lexical is not None:
            inputs[executable.input(1)] = lexical[None, :]
        if grouped is not None:
            inputs.update(zip(executable.inputs[2:], grouped))
        result = request.infer(inputs)[executable.output(0)]
        return result[0].astype(np.float32, copy=True)

    return infer, identity
