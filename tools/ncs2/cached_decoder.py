"""Static, one-token Llama decoder with host-managed KV storage on Myriad X."""

import math
from hashlib import sha256
from copy import copy
from pathlib import Path
import sys
from time import perf_counter_ns

import numpy as np

from decoder_checkpoint import Checkpoint, load_prepared

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from myriadforge.myriad_x import allocation_properties, require_myriad_x


class ContextLimit(ValueError):
    """The static decoder window has no space for another token."""


def load_weights(path):
    return Checkpoint(path) if path.is_file() else load_prepared(path)


def decoder_fingerprint():
    return sha256(Path(__file__).read_text(encoding="utf8").encode("utf8")).hexdigest()


def weight_bytes(weights, start, end, logits):
    arrays = ("att_norm", "wq", "wk", "wv", "wo", "ffn_norm", "w1", "w2", "w3")
    count = sum(getattr(weights, name)[start:end].size for name in arrays)
    if logits:
        count += weights.final_norm.size + weights.output.size
    return count * 2


def partitions(weights, count):
    if count == 1:
        cut = weights.layers
        return [(0, cut)]
    if count != 2 or weights.layers < 2:
        raise ValueError("Use one graph or two nonempty contiguous graphs")
    cut = min(range(1, weights.layers), key=lambda i: abs(
        weight_bytes(weights, 0, i, False) - weight_bytes(weights, i, weights.layers, True)
    ))
    return [(0, cut), (cut, weights.layers)]


def rotary(weights, position):
    width = weights.head_dim
    phase = position * weights.theta ** (-np.arange(0, width, 2, dtype=np.float32) / width)
    phase = np.repeat(phase, 2) if weights.rope_layout == "interleaved" else np.tile(phase, 2)
    return np.cos(phase)[None, :], np.sin(phase)[None, :]


def build_graph(weights, context, start, end, *, logits, diagnostic=False,
                attention="reduce", layout="packed", norm="standard"):
    """Compile static 2-D operations. Cache rows use device-friendly FP16 storage."""
    from openvino.runtime import Model, opset8 as op

    def constant(value):
        return op.constant(np.asarray(value, dtype=np.float32))

    def dense(x, weight):
        return op.matmul(x, constant(weight), False, True)

    def graph_norm(x, weight, epsilon):
        width = weight.size
        if norm == "scaled":
            return scaled_norm(x, weight, epsilon)
        power = op.reduce_sum(op.multiply(x, x), np.array([1], np.int32), True)
        divisor = op.sqrt(op.add(power, constant(epsilon * width)))
        return op.multiply(op.divide(x, divisor), constant(weight[None, :] * math.sqrt(width)))

    def scaled_norm(x, weight, epsilon):
        axes = np.array([1], np.int32)
        width = weight.size
        peak = op.maximum(op.reduce_max(op.absolute(x), axes, True), constant(.001))
        scaled = op.divide(x, peak)
        power = op.reduce_sum(op.multiply(scaled, scaled), axes, True)
        offset = op.divide(op.divide(constant(epsilon * width), peak), peak)
        return op.multiply(op.divide(scaled, op.sqrt(op.add(power, offset))),
                           constant(weight[None, :] * math.sqrt(width)))

    def graph_parameter(name, shape, dtype=np.float32):
        parameter = op.parameter(shape, dtype, name=name)
        parameter.output(0).get_tensor().set_names({name})
        return parameter

    heads, kv_heads, width = weights.heads, weights.kv_heads, weights.head_dim
    if not 0 <= start < end <= weights.layers or context < 2:
        raise ValueError("Invalid graph range or context")
    aligned = layout == "aligned"
    rows = (end - start) * kv_heads * (context if aligned else context - 1)
    hidden = graph_parameter("hidden", [1, weights.dim])
    cos = graph_parameter("cos", [1, width])
    sin = graph_parameter("sin", [1, width])
    mask = graph_parameter("mask", [1, context])
    graph_inputs = [hidden, cos, sin, mask]
    row_axis = np.int32(0)
    col_axis = np.int32(1)

    def history_ports(name):
        if layout != "layer":
            dtype = np.float32 if layout == "float32" else np.float16
            parameter = graph_parameter(name, [rows, width], dtype)
            graph_inputs.append(parameter)
            groups = op.split(op.convert(parameter, np.float32), row_axis, (end - start) * kv_heads).outputs()
            if aligned:
                return [op.slice(group, [0], [context - 1], [1], [0]) for group in groups]
            return groups
        outputs = []
        for layer in range(start, end):
            parameter = graph_parameter(f"{name}_{layer}", [kv_heads * (context - 1), width], np.float16)
            graph_inputs.append(parameter)
            outputs.extend(op.split(op.convert(parameter, np.float32), row_axis, kv_heads).outputs())
        return outputs

    past_keys, past_values = history_ports("keys"), history_ports("values")
    rotation = np.zeros((width, width), np.float32)
    left = np.arange(0, width, 2) if weights.rope_layout == "interleaved" else np.arange(width // 2)
    right = left + (1 if weights.rope_layout == "interleaved" else width // 2)
    rotation[left, right] = 1
    rotation[right, left] = -1
    rotation = constant(rotation)
    scale = constant(width ** -.5)
    sum_rows = np.array([0], np.int32)
    sum_columns = np.array([1], np.int32)
    transpose = np.array([1, 0], np.int32)

    def rotate(x):
        return op.add(op.multiply(x, cos), op.multiply(op.matmul(x, rotation, False, False), sin))

    def fused_split(x, matrices):
        lengths = np.array([matrix.shape[0] for matrix in matrices], np.int32)
        return op.variadic_split(dense(x, np.concatenate(matrices)), col_axis, lengths).outputs()

    x = hidden
    keys_out, values_out, taps = [], [], []
    for index in range(start, end):
        normalized = graph_norm(x, weights.att_norm[index], weights.epsilon)
        if attention == "grouped":
            q, k, v = fused_split(normalized, (weights.wq[index], weights.wk[index], weights.wv[index]))
            q = op.split(op.reshape(q, [heads, width], False), row_axis, kv_heads)
        else:
            q = op.split(dense(normalized, weights.wq[index]), col_axis, heads)
            k = dense(normalized, weights.wk[index])
            v = dense(normalized, weights.wv[index])
        k = op.split(k, col_axis, kv_heads)
        v = op.split(v, col_axis, kv_heads)
        new_keys = [rotate(k.output(i)) for i in range(kv_heads)]
        new_values = [v.output(i) for i in range(kv_heads)]
        offset = (index - start) * kv_heads
        histories = [
            (op.concat([past_keys[offset + i], new_keys[i]], 0),
             op.concat([past_values[offset + i], new_values[i]], 0))
            for i in range(kv_heads)
        ]
        outputs = []
        groups = kv_heads if attention == "grouped" else heads
        for index_head in range(groups):
            query = rotate(q.output(index_head))
            key, value = histories[index_head // (groups // kv_heads)]
            if attention in ("gemm", "grouped"):
                scores = op.matmul(query, key, False, True)
            else:
                scores = op.transpose(op.reduce_sum(op.multiply(key, query), sum_columns, True), transpose)
            probability = op.softmax(op.add(op.multiply(scores, scale), mask), 1)
            if attention in ("gemm", "grouped"):
                attended = op.matmul(probability, value, False, False)
            else:
                weighted = op.multiply(value, op.transpose(probability, transpose))
                attended = op.reduce_sum(weighted, sum_rows, True)
            if diagnostic is True and not index_head:
                taps.extend([normalized, query, new_keys[0], scores, probability, attended])
            outputs.append(attended)
        joined = (op.reshape(op.concat(outputs, 0), [1, weights.dim], False) if attention == "grouped"
                  else op.concat(outputs, 1))
        x = op.add(x, dense(joined, weights.wo[index]))
        normalized = graph_norm(x, weights.ffn_norm[index], weights.epsilon)
        if attention == "grouped":
            gate, up = fused_split(normalized, (weights.w1[index], weights.w3[index]))
        else:
            gate, up = dense(normalized, weights.w1[index]), dense(normalized, weights.w3[index])
        x = op.add(x, dense(op.multiply(op.multiply(gate, op.sigmoid(gate)), up), weights.w2[index]))
        if diagnostic == "layers":
            taps.append(x)
        keys_out.extend(new_keys)
        values_out.extend(new_values)
    result = dense(graph_norm(x, weights.final_norm, weights.epsilon), weights.output) if logits else x
    outputs = [
        result, op.convert(op.concat(keys_out, 0), np.float16), op.convert(op.concat(values_out, 0), np.float16),
    ]
    return Model(outputs + taps, graph_inputs, f"Cached Llama blocks {start}-{end}")


class Cache:
    def __init__(self, weights, context, aligned=False):
        self.context = context
        self.keys = np.zeros((weights.layers, weights.kv_heads, context - 1 + aligned, weights.head_dim),
                             dtype=np.float16)
        self.values = np.zeros_like(self.keys)
        self.position = 0

    def reset(self):
        self.keys.fill(0)
        self.values.fill(0)
        self.position = 0

    def capacity(self):
        if self.position >= self.context:
            raise ContextLimit(f"context limit of {self.context} tokens reached")

    def mask(self):
        mask = np.full((1, self.context), -10000, np.float32)
        mask[:, :self.position] = 0
        mask[:, -1] = 0
        return mask

    def commit(self, keys, values):
        if self.position < self.context - 1:
            self.keys[:, :, self.position, :] = keys
            self.values[:, :, self.position, :] = values
        self.position += 1


class DeviceDecoder:
    def __init__(self, weights, context, graph_count, allocation, *, attention="reduce",
                 layout="packed", device="MYRIAD", norm="standard", hardware=True):
        from openvino.runtime import Core

        self.weights = weights
        self.cache = Cache(weights, context, aligned=layout == "aligned")
        self.parts = []
        self.load_ms = []
        self.layout = layout
        self.core = Core()
        self.device = require_myriad_x(self.core) if device == "MYRIAD" else "CPU"
        options = allocation_properties(*allocation) if device == "MYRIAD" else {"INFERENCE_NUM_THREADS": 1}
        if device == "MYRIAD" and not hardware:
            options["MYRIAD_ENABLE_HW_ACCELERATION"] = "NO"
        self.graphs = partitions(weights, graph_count)
        for start, end in self.graphs:
            start_time = perf_counter_ns()
            print(f"BUILD {start}:{end}", flush=True)
            model = build_graph(weights, context, start, end, logits=end == weights.layers,
                                attention=attention, layout=layout, norm=norm)
            print(f"COMPILE {start}:{end}", flush=True)
            compiled = self.core.compile_model(model, device, options)
            self.parts.append((compiled, compiled.create_infer_request()))
            self.load_ms.append((perf_counter_ns() - start_time) / 1e6)
            print(f"LOADED {start}:{end}", flush=True)

    def reset(self):
        self.cache.reset()

    def new_session(self):
        session = copy(self)
        session.cache = Cache(self.weights, self.cache.context, aligned=self.layout == "aligned")
        session.parts = [(model, model.create_infer_request()) for model, _ in self.parts]
        return session

    def step(self, token):
        self.cache.capacity()
        if not 0 <= token < self.weights.embed.shape[0]:
            raise ValueError("Token is outside the checkpoint vocabulary")
        cos, sin = rotary(self.weights, self.cache.position)
        mask = self.cache.mask()
        x = np.asarray(self.weights.embed[token:token + 1], dtype=np.float32)
        width = self.weights.head_dim
        updates_k, updates_v = [], []
        for (start, end), (compiled, request) in zip(self.graphs, self.parts):
            inputs = {"hidden": x, "cos": cos, "sin": sin, "mask": mask}
            for name, cache in (("keys", self.cache.keys), ("values", self.cache.values)):
                if self.layout != "layer":
                    data = cache[start:end].reshape(-1, width)
                    inputs[name] = data.astype(np.float32) if self.layout == "float32" else data
                else:
                    inputs.update((f"{name}_{layer}", cache[layer].reshape(-1, width))
                                  for layer in range(start, end))
            output = request.infer(inputs)
            x = np.array(output[compiled.output(0)], dtype=np.float32, copy=True)
            shape = (end - start, self.weights.kv_heads, width)
            updates_k.append(np.array(output[compiled.output(1)], dtype=np.float16, copy=True).reshape(shape))
            updates_v.append(np.array(output[compiled.output(2)], dtype=np.float16, copy=True).reshape(shape))
        if not np.isfinite(x).all() or any(not np.isfinite(v).all() for v in updates_k + updates_v):
            raise ArithmeticError("Non-finite logits or KV update")
        self.cache.commit(np.concatenate(updates_k), np.concatenate(updates_v))
        return x.reshape(-1)


class ReferenceDecoder:
    """Independent NumPy decoder with FP32 operations and FP16 saved K/V."""

    def __init__(self, weights, context):
        self.weights = weights
        self.cache = Cache(weights, context)

    def reset(self):
        self.cache.reset()

    def step(self, token, *, stop=None, trace=None):
        self.cache.capacity()
        w = self.weights
        if not 0 <= token < w.embed.shape[0]:
            raise ValueError("Token is outside the checkpoint vocabulary")
        pos = self.cache.position
        cos, sin = rotary(w, pos)
        group = np.arange(w.heads) // (w.heads // w.kv_heads)
        left = np.arange(0, w.head_dim, 2) if w.rope_layout == "interleaved" else np.arange(w.head_dim // 2)
        right = left + (1 if w.rope_layout == "interleaved" else w.head_dim // 2)
        keys_out, values_out = [], []

        def norm(x, scale):
            return x * scale / np.sqrt(np.mean(x * x) + w.epsilon)

        def rotate(x):
            cross = np.empty_like(x)
            cross[:, left], cross[:, right] = -x[:, right], x[:, left]
            return x * cos + cross * sin

        x = w.embed[token]
        for layer in range(w.layers if stop is None else stop):
            normalized = norm(x, w.att_norm[layer])
            q = rotate((w.wq[layer] @ normalized).reshape(w.heads, w.head_dim))
            k = rotate((w.wk[layer] @ normalized).reshape(w.kv_heads, w.head_dim))
            v = (w.wv[layer] @ normalized).reshape(w.kv_heads, w.head_dim)
            arr_k = np.concatenate([self.cache.keys[layer, :, :pos].astype(np.float32), k[:, None, :]], axis=1)
            arr_v = np.concatenate([self.cache.values[layer, :, :pos].astype(np.float32), v[:, None, :]], axis=1)
            scores = np.einsum("hd,htd->ht", q, arr_k[group])
            scores *= w.head_dim ** -.5
            probabilities = np.exp(scores - scores.max(axis=1, keepdims=True))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            attended = np.einsum("ht,htd->hd", probabilities, arr_v[group])
            if trace is not None:
                trace.update(norm=normalized, query=q[0], key=k[0], scores=scores[0],
                             softmax=probabilities[0], attention=attended[0])
            x = x + w.wo[layer] @ attended.reshape(-1)
            normalized = norm(x, w.ffn_norm[layer])
            gate, up = w.w1[layer] @ normalized, w.w3[layer] @ normalized
            x = x + w.w2[layer] @ (up * (gate / (1 + np.exp(-gate))))
            if trace is not None:
                trace[f"block_{layer}"] = x.copy()
            keys_out.append(k)
            values_out.append(v)
        if stop is not None:
            return x
        self.cache.commit(np.asarray(keys_out), np.asarray(values_out))
        return w.output @ norm(x, w.final_norm)
