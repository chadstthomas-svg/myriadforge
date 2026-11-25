"""Fixed-shape contextual adapter shared by the NumPy and MYRIAD runtimes."""

from hashlib import sha256

import numpy as np

SCHEMA = "myriadforge.context-adapter/1"
GROUPS, GROUP_WIDTH, BUCKETS = 8, 16, 256
WIDTH, HIDDEN, EPSILON = 32, 64, 1e-4
CONTEXT_WEIGHT = 0.2
VARIANTS = {"features": (), "one": (0,), "tied": (0, 0), "untied": (0, 1)}
BLOCK_SHAPES = {"q": (WIDTH, WIDTH), "k": (WIDTH, WIDTH), "v": (WIDTH, WIDTH),
                "o": (WIDTH, WIDTH), "up": (WIDTH, HIDDEN), "down": (HIDDEN, WIDTH),
                "gate": (WIDTH, 1)}


def adapter_shapes(variant):
    schedule = VARIANTS[variant]
    return {"token": (BUCKETS, WIDTH), "position": (GROUPS, WIDTH), "bias": (len(schedule),),
            **{f"{kind}{block}": shape for block in sorted(set(schedule))
               for kind, shape in BLOCK_SHAPES.items()}}


def load_adapter(path):
    with np.load(path, allow_pickle=False) as source:
        if "variant" not in source or "schema" not in source:
            raise ValueError("A context adapter needs schema and variant fields")
        variant = str(source["variant"].item())
        if source["schema"].item() != SCHEMA or variant not in VARIANTS:
            raise ValueError("Unsupported context adapter schema or variant")
        shapes = adapter_shapes(variant)
        if not shapes.keys() <= set(source.files):
            raise ValueError("A context adapter is missing required weights")
        arrays = {name: source[name] for name in shapes}
    for name, value in arrays.items():
        if value.shape != shapes[name] or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid context weight {name}")
    return variant, arrays


def model_fingerprint(*paths):
    digest = sha256()
    for path in paths:
        digest.update(sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def grouped_features(ids):
    features = np.zeros((GROUPS, BUCKETS), dtype=np.float32)
    mask = np.zeros((GROUPS, 1), dtype=np.float32)
    for pos in range(GROUPS):
        group = ids[pos * GROUP_WIDTH:(pos + 1) * GROUP_WIDTH]
        if not len(group):
            break
        for token in group:
            key = ((int(token) + 1) * 2654435761) & 0xFFFFFFFF
            features[pos, (key >> 12) % (BUCKETS // 2)] += -1 if key >> 31 else 1
        for a, b in zip(group, group[1:]):
            key = ((int(a) + 1) * 2246822519 + (int(b) + 1) * 3266489917) & 0xFFFFFFFF
            features[pos, BUCKETS // 2 + (key >> 12) % (BUCKETS // 2)] += -1 if key >> 31 else 1
        features[pos] /= np.sqrt(len(group))
        mask[pos] = 1
    return features, mask


def unit(vector):
    return vector / max(float(np.linalg.norm(vector)), 1e-4)


def numpy_context(features, mask, adapter):
    variant, w = adapter

    def norm(x):
        x = x - x.mean(axis=1, keepdims=True)
        return x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + EPSILON)

    state = features @ w["token"] + w["position"]
    for step, block in enumerate(VARIANTS[variant]):
        x = norm(state)
        scores = (x @ w[f"q{block}"]) @ (x @ w[f"k{block}"]).T
        scores = scores / np.float32(np.sqrt(WIDTH)) + (mask.T - 1) * 64
        scores = np.exp(scores - scores.max(axis=1, keepdims=True))
        attn = scores / scores.sum(axis=1, keepdims=True)
        logit = np.clip(x @ w[f"gate{block}"] + w["bias"][step], -30, 30)
        gate = 1 / (1 + np.exp(-logit))
        state += gate * (attn @ (x @ w[f"v{block}"]) @ w[f"o{block}"])
        state += gate * (np.maximum(norm(state) @ w[f"up{block}"], 0) @ w[f"down{block}"])
    return unit((state * mask).sum(axis=0))


def graph_context(op, adapter):
    variant, w = adapter
    features = op.parameter([GROUPS, BUCKETS], np.float32, name="group_features")
    mask = op.parameter([GROUPS, 1], np.float32, name="valid_groups")
    c = {key: op.constant(value, name=f"context_{key}") for key, value in w.items() if key != "bias"}
    channel_axis = op.constant(np.array([2], dtype=np.int32))

    def linear(x, key):
        return op.matmul(x, c[key], False, False)

    def norm(x):
        shaped = op.reshape(x, [GROUPS, 1, WIDTH], False)
        normalized = op.mvn(shaped, channel_axis, True, EPSILON, "inside_sqrt")
        return op.reshape(normalized, [GROUPS, WIDTH], False)

    state = op.add(linear(features, "token"), c["position"])
    key_mask = op.multiply(op.transpose(op.subtract(mask, np.float32(1)), [1, 0]), np.float32(64))
    for step, block in enumerate(VARIANTS[variant]):
        x = norm(state)
        scores = op.matmul(linear(x, f"q{block}"), linear(x, f"k{block}"), False, True)
        scores = op.multiply(scores, np.float32(1 / np.sqrt(WIDTH)))
        attn = op.softmax(op.add(scores, key_mask), 1)
        values = linear(op.matmul(attn, linear(x, f"v{block}"), False, False), f"o{block}")
        gate = op.sigmoid(op.add(linear(x, f"gate{block}"), w["bias"][step]))
        state = op.add(state, op.multiply(gate, values))
        values = linear(op.relu(linear(norm(state), f"up{block}")), f"down{block}")
        state = op.add(state, op.multiply(gate, values))
    pooled = op.reduce_sum(op.multiply(state, mask), op.constant(np.array([0], dtype=np.int32)), True)
    context = op.normalize_l2(pooled, op.constant(np.array([1], dtype=np.int32)), 1e-8, "max")
    return context, (features, mask)
