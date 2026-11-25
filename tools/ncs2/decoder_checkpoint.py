"""Prepare pinned Hugging Face Llama weights for the legacy NCS2 process."""

from argparse import ArgumentParser
from contextlib import nullcontext
from hashlib import sha256
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np


SINGLES = {"embed": "model.embed_tokens.weight", "final_norm": "model.norm.weight"}
BLOCKS = {
    "att_norm": "input_layernorm.weight", "ffn_norm": "post_attention_layernorm.weight",
    "wq": "self_attn.q_proj.weight", "wk": "self_attn.k_proj.weight",
    "wv": "self_attn.v_proj.weight", "wo": "self_attn.o_proj.weight",
    "w1": "mlp.gate_proj.weight", "w2": "mlp.down_proj.weight", "w3": "mlp.up_proj.weight",
}


class Checkpoint:
    """Legacy llama2.c float32 checkpoint layout."""

    def __init__(self, file):
        with file.open("rb") as stream:
            d, h, layers, heads, kv_heads, vocab, limit = struct.unpack("<7i", stream.read(28))
        if (min(d, h, layers, heads, kv_heads, abs(vocab), limit) <= 0 or d % heads
                or heads % kv_heads or (d // heads) % 2):
            raise ValueError("Invalid llama2.c checkpoint header")
        self.dim, self.layers, self.heads = d, layers, heads
        self.kv_heads, self.head_dim, self.vocab, self.limit = kv_heads, d // heads, abs(vocab), limit
        self.epsilon, self.theta, self.rope_layout = 1e-5, 10_000.0, "interleaved"
        self.tied = vocab > 0
        k = self.head_dim * kv_heads
        data = np.memmap(file, mode="r", dtype="<f4", offset=28)
        offset = 0

        def take(*shape):
            nonlocal offset
            size = int(np.prod(shape))
            array = data[offset:offset + size].reshape(shape)
            offset += size
            return array

        self.embed = take(self.vocab, d)
        self.att_norm = take(layers, d)
        self.wq = take(layers, d, d)
        self.wk = take(layers, k, d)
        self.wv = take(layers, k, d)
        self.wo = take(layers, d, d)
        self.ffn_norm = take(layers, d)
        self.w1 = take(layers, h, d)
        self.w2 = take(layers, d, h)
        self.w3 = take(layers, h, d)
        self.final_norm = take(d)
        self.parameters = offset + (0 if self.tied else self.vocab * d)
        take(limit, self.head_dim)  # Legacy stored RoPE tables.
        self.output = self.embed if self.tied else take(self.vocab, d)
        if offset != len(data):
            raise ValueError("Unexpected llama2.c checkpoint length")


def digest(path):
    result = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
    return result.hexdigest()


def checkpoint_spec(config):
    if (config["model_type"] != "llama" or config["hidden_act"] != "silu"
            or config.get("attention_bias") or config.get("mlp_bias") or config.get("rope_scaling")):
        raise ValueError("Expected a bias-free Llama with the standard RoPE and SiLU")
    hidden, heads = config["hidden_size"], config["num_attention_heads"]
    kv = config.get("num_key_value_heads", heads)
    width = config.get("head_dim", hidden // heads)
    if width * heads != hidden or width % 2 or heads % kv:
        raise ValueError("Unexpected Llama projection dimensions")
    return dict(dim=hidden, inner=config["intermediate_size"], heads=heads, kv_heads=kv, head_dim=width,
                layers=config["num_hidden_layers"], vocab=config["vocab_size"],
                limit=config["max_position_embeddings"], epsilon=config["rms_norm_eps"],
                theta=config.get("rope_theta", 10_000.0), rope_layout="split",
                tied=config.get("tie_word_embeddings", False), eos=config["eos_token_id"])


def expected_shapes(w):
    d, i, k, n = w.dim, w.inner, w.kv_heads * w.head_dim, w.layers
    shapes = {
        "embed": (w.vocab, d), "final_norm": (d,), "att_norm": (n, d), "ffn_norm": (n, d),
        "wq": (n, d, d), "wk": (n, k, d), "wv": (n, k, d), "wo": (n, d, d),
        "w1": (n, i, d), "w2": (n, d, i), "w3": (n, i, d),
    }
    if not w.tied:
        shapes["output"] = shapes["embed"]
    return shapes


def prepare(source, destination, revision):
    if destination.exists():
        raise FileExistsError(f"Prepared checkpoint already exists at {destination}")
    config = source / "config.json"
    spec = checkpoint_spec(json.loads(config.read_text(encoding="utf8")))
    weights = SimpleNamespace(**spec)
    shapes = expected_shapes(weights)
    path = source / "model.safetensors"
    if path.exists():
        from safetensors import safe_open

        opened = safe_open(str(path), framework="pt", device="cpu")
    else:
        import torch

        path = source / "pytorch_model.bin"
        opened = nullcontext(torch.load(path, map_location="cpu", weights_only=True))
    destination.parent.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with TemporaryDirectory(prefix=".ncs2-checkpoint-", dir=destination.parent) as directory, opened as tensors:
        staging = Path(directory) / "weights"
        staging.mkdir()
        get = tensors.get_tensor if hasattr(tensors, "get_tensor") else tensors.__getitem__

        def read_tensor(key, shape):
            value = get(key)
            if tuple(value.shape) != shape:
                raise ValueError(f"Tensor {key} has shape {tuple(value.shape)}, expected {shape}")
            return value.float().numpy()

        for name, shape in shapes.items():
            file = staging / (name + ".npy")
            stored = np.lib.format.open_memmap(file, mode="w+", shape=shape, dtype=np.float32)
            try:
                if name in BLOCKS:
                    for layer in range(weights.layers):
                        key = f"model.layers.{layer}.{BLOCKS[name]}"
                        stored[layer] = read_tensor(key, shape[1:])
                else:
                    stored[:] = read_tensor(SINGLES.get(name, "lm_head.weight"), shape)
                stored.flush()
            finally:
                del stored
            hashes[name] = digest(file)
        manifest = {"schema": "myriadforge.llama-weights/1", "revision": revision, "spec": spec,
                    "source_sha256": {path.name: digest(path), config.name: digest(config)},
                    "weights_sha256": hashes}
        (staging / "checkpoint.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
        staging.rename(destination)


def load_prepared(directory):
    manifest = json.loads((directory / "checkpoint.json").read_text(encoding="utf8"))
    if manifest["schema"] != "myriadforge.llama-weights/1":
        raise ValueError("Unexpected prepared checkpoint format")
    result = SimpleNamespace(**manifest["spec"])
    result.parameters = 0
    for name, shape in expected_shapes(result).items():
        file = directory / (name + ".npy")
        if digest(file) != manifest["weights_sha256"][name]:
            raise ValueError(f"Weight hash disagrees for {name}")
        array = np.load(file, mmap_mode="r", allow_pickle=False)
        if array.shape != shape or array.dtype != np.float32:
            raise ValueError(f"Unexpected tensor layout for {name}")
        setattr(result, name, array)
        result.parameters += array.size
    if result.tied:
        result.output = result.embed
    return result


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    prepare(args.source, args.destination, args.revision)
