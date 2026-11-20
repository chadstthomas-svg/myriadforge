"""Compile exact Boolean policies to static graphs for Myriad 2."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper as h, numpy_helper

from policy import Policy


class Graph:
    def __init__(self):
        self.nodes = []
        self.constants = []

    def const(self, data):
        key = "k%d" % len(self.constants)
        self.constants.append(numpy_helper.from_array(np.asarray(data, dtype=np.float32), key))
        return key

    def op(self, op, inputs, **attributes):
        value = "v%d" % len(self.nodes)
        self.nodes.append(h.make_node(op, inputs, [value], name=value, **attributes))
        return value

    def conv(self, inputs, weights, bias):
        return self.op("Conv", [inputs, self.const(weights), self.const(bias)],
                       kernel_shape=list(weights.shape[2:]))


def dense_head(g, policy, features):
    # Inverted edges allow both AND and OR to use one affine/ReLU gate each.
    n = len(policy.literals)
    gates = []
    levels = [0] * n
    def lower(node):
        if node[0] == "literal":
            return policy.literals.index(node[1]), False
        if node[0] == "not":
            i, invert = lower(node[1])
            return i, not invert
        disjunction = node[0] == "or"
        args = tuple(sorted((i, inv ^ disjunction) for i, inv in map(lower, node[1:])))
        if args not in gates:
            gates.append(args)
            levels.append(1 + max(levels[i] for i, _ in args))
        return n + gates.index(args), disjunction
    ends = [lower(node) for node in policy.outputs]
    live = list(range(n))
    for depth in range(1, max(levels) + 1):
        needed = {i for i, _ in ends}
        for j, pair in enumerate(gates, n):
            if levels[j] > depth:
                needed.update(i for i, _ in pair)
        layer = [i for i in live if i in needed] + [i for i, d in enumerate(levels) if d == depth]
        matrix = np.zeros((len(layer), len(live), 1, 1), dtype=np.float32)
        bias = np.zeros(len(layer), dtype=np.float32)
        for row, key in enumerate(layer):
            if key in live:
                matrix[row, live.index(key)] = 1
            else:
                bias[row] = -1
                for i, inv in gates[key-n]:
                    matrix[row, live.index(i)] += -1 if inv else 1
                    bias[row] += int(inv)
        features = g.op("Relu", [g.conv(features, matrix, bias)])
        live = layer
    matrix = np.zeros((len(ends), len(live), 1, 1), dtype=np.float32)
    bias = []
    for row, (i, inv) in enumerate(ends):
        matrix[row, live.index(i)] = -1 if inv else 1
        bias.append(int(inv))
    return g.conv(features, matrix, bias)


def gates_head(g, policy, features):
    slices = ["atom%d" % i for i in range(len(policy.literals))]
    g.nodes.append(h.make_node("Split", [features], slices, axis=1, split=[1]*len(slices)))
    one = g.const(np.ones((1, 1, 1, 1)))
    seen = {}
    def emit(node):
        if node not in seen:
            if node[0] == "literal":
                v = slices[policy.literals.index(node[1])]
            elif node[0] == "not":
                v = g.op("Sub", [one, emit(node[1])])
            else:
                v = g.op("Mul" if node[0] == "and" else "Max", [emit(n) for n in node[1:]])
            seen[node] = v
        return seen[node]
    return g.op("Concat", [emit(node) for node in policy.outputs], axis=1)


def scanner(g, policy, side):
    # Exact equality on small integer word IDs using adjacent ReLU hinges.
    vocab = len(policy.word_ids)
    hinges = np.ones((vocab+2, 1, 1, 1), dtype=np.float32)
    cuts = g.op("Relu", [g.conv("lexemes", hinges, -np.arange(vocab+2))])
    tents = np.zeros((vocab, vocab+2, 1, 1), dtype=np.float32)
    for i in range(vocab):
        tents[i, i:i+3, 0, 0] = [1, -2, 1]
    equal = g.conv(cuts, tents, np.zeros(vocab))
    width = max(map(len, policy.patterns))
    w = np.zeros((len(policy.patterns), vocab, 1, width), dtype=np.float32)
    for row, phrase in enumerate(policy.patterns):
        for j, token in enumerate(phrase):
            w[row, token-1, 0, width - len(phrase) + j] = 1
    scores = g.op("Relu", [g.conv(equal, w, [1-len(p) for p in policy.patterns])])
    remaining = side
    while remaining > 1:
        size = 4 if remaining % 4 == 0 else 2
        scores = g.op("MaxPool", [scores], kernel_shape=[size,size], strides=[size,size])
        remaining //= size
    return g.op("Max", [scores, "prior"])


def packed_table(policy):
    """Encode reachable output vectors using signed, exact FP16 integers."""
    facts = len(policy.literals)
    if facts > 10:
        raise ValueError("Packed lookup supports up to 10 literal predicates")
    rows = [tuple(map(int, policy.reference([(mask >> j) & 1 for j in range(facts)])))
            for mask in range(1 << facts)]
    codebook = sorted(set(rows))
    code_width = max(1, (len(codebook) - 1).bit_length())
    lane_bits = min(facts, (12 // code_width).bit_length() - 1)
    lanes = 1 << lane_bits
    # FP16 represents every integer in [-2048, 2047] exactly. Centering
    # the word at zero fits 12 bits where an unsigned word would fit 11.
    offset = 1 << (code_width * lanes - 1)
    indices = {row: i for i, row in enumerate(codebook)}
    words = [sum(indices[row] << (lane * code_width)
                 for lane, row in enumerate(rows[first:first + lanes])) - offset
             for first in range(0, len(rows), lanes)]
    table = np.asarray(words, dtype=np.float32).reshape(1, -1, 1, 1)
    return table, dict(codebook=codebook, code_width=code_width, lane_bits=lane_bits, offset=offset)


def compile_policy(source, directory, variant, side=64):
    if side < 2 or side & (side-1):
        raise ValueError("Tile side must be a power of two")
    spec = json.loads(source.read_text(encoding="utf8"))
    policy = Policy(spec)
    g = Graph()
    nf = len(policy.literals)
    shapes = {"features": [1, nf, 1, 1]}
    outputs = {}
    metadata = {}
    if variant == "scan":
        shapes = {"lexemes": [1, 1, side, side + max(map(len,policy.patterns)) - 1],
                  "prior": [1, nf, 1, 1]}
        features = scanner(g, policy, side)
    else:
        features = "features"
    if variant in ("packed", "gather"):
        table, metadata = packed_table(policy)
        if variant == "gather":
            shapes = {"features": [1]}
            decisions = g.op("Gather", [g.const(table.reshape(-1)), "features"], axis=0)
            outputs[decisions] = [1]
        else:
            shapes = {"features": [1, table.shape[1], 1, 1]}
            decisions = g.conv("features", table, [0])
            outputs[decisions] = [1, 1, 1, 1]
    elif variant == "lookup":
        if nf > 10:
            raise ValueError("The lookup form is intended for at most 10 literal predicates")
        count = 1 << nf
        words = (len(policy.outputs)+9)//10
        table = np.zeros((words,count,1,1),dtype=np.float32)
        for index in range(count):
            facts = [(index >> j) & 1 for j in range(nf)]
            for k,bit in enumerate(policy.reference(facts)):
                table[k//10,index,0,0] += int(bit) << (k % 10)
        shapes = {"features":[1,count,1,1]}
        decisions = g.conv("features",table,np.zeros(words))
        outputs[decisions] = [1,words,1,1]
    else:
        head = gates_head if variant == "gates" else dense_head
        decisions = head(g, policy, features)
        if variant == "scan":
            decisions = g.op("Concat", [features, decisions], axis=1)
        outputs[decisions] = [1, len(policy.outputs) + (nf if variant == "scan" else 0), 1, 1]
    def ports(mapping):
        return [h.make_tensor_value_info(
            name, TensorProto.INT32 if variant == "gather" and name == "features" else TensorProto.FLOAT, shape
        ) for name, shape in mapping.items()]
    # OpenVINO uses producer node names when naming outputs, so keep those names in the manifest.
    graph = h.make_graph(g.nodes, "boolean_" + variant, ports(shapes), ports(outputs), g.constants)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 9)], ir_version=6)
    onnx.checker.check_model(model)
    directory.mkdir(exist_ok=True, parents=True)
    blob = model.SerializeToString()
    (directory/"model.onnx").write_bytes(blob)
    manifest = dict(policy=spec, variant=variant, decisions=decisions,
                    state_channels=nf if variant=="scan" else 0, tile_side=side,
                    input_shapes=shapes, onnx_sha256=hashlib.sha256(blob).hexdigest(),
                    fp16_coefficient_bytes=2*sum(np.prod(v.dims, dtype=int).item() for v in g.constants),
                    learned_parameters=0, **metadata)
    (directory/"model.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("policy", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--variant", choices=["gates", "dense", "scan", "lookup", "packed", "gather"], required=True)
    p.add_argument("--side", type=int, default=64)
    args = p.parse_args()
    compile_policy(args.policy, args.output, args.variant, args.side)
