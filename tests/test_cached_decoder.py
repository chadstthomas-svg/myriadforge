import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def decoder(monkeypatch):
    location = Path(__file__).resolve().parents[1] / "tools/ncs2"
    monkeypatch.syspath_prepend(str(location))
    return importlib.import_module("cached_decoder")


def test_cache_window_and_reset(decoder):
    weights = SimpleNamespace(layers=2, kv_heads=1, head_dim=2)
    cache = decoder.Cache(weights, 3)
    assert cache.keys.dtype == np.float16
    np.testing.assert_array_equal(cache.mask(), [[-10000, -10000, 0]])
    keys = np.array([[[1, 2]], [[3, 4]]], dtype=np.float32)
    for token in range(3):
        cache.capacity()
        cache.commit(keys + token, keys - token)
    with pytest.raises(decoder.ContextLimit, match="3 tokens"):
        cache.capacity()
    np.testing.assert_array_equal(cache.keys[:, :, 1, :], keys + 1)
    np.testing.assert_array_equal(cache.mask(), [[0, 0, 0]])
    cache.reset()
    assert cache.position == 0 and not cache.keys.any() and not cache.values.any()


def test_partition_includes_projection_weight(decoder):
    weights = SimpleNamespace(layers=4, final_norm=np.ones(2), output=np.ones((20, 2)))
    block = np.ones((4, 1))
    for name in ("att_norm", "wq", "wk", "wv", "wo", "ffn_norm", "w1", "w2", "w3"):
        setattr(weights, name, block)
    assert decoder.partitions(weights, 2) == [(0, 3), (3, 4)]
    assert sum(decoder.weight_bytes(weights, a, b, b == 4)
               for a, b in decoder.partitions(weights, 2)) == decoder.weight_bytes(weights, 0, 4, True)


def test_reference_normalization_and_token_projection(decoder):
    dimension, vocab = 2, 3
    weights = SimpleNamespace(
        layers=1, kv_heads=1, heads=1, head_dim=2, dim=dimension,
        epsilon=1e-5, theta=10000, rope_layout="interleaved",
        embed=np.array([[1., 2.], [3., 4.], [-1., -2.]], dtype=np.float32),
        output=np.array([[1., 2.], [3., 4.], [-1., -2.]], dtype=np.float32),
        final_norm=np.ones(dimension), att_norm=np.ones((1, dimension)),
        ffn_norm=np.ones((1, dimension)),
    )
    for name in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
        setattr(weights, name, np.zeros((1, dimension, dimension), dtype=np.float32))
    runtime = decoder.ReferenceDecoder(weights, 4)
    character = weights.embed[0]
    expected = weights.output @ (character / np.sqrt(np.mean(character**2) + weights.epsilon))
    result = runtime.step(0)
    assert result.shape == (vocab,)
    np.testing.assert_allclose(result, expected, rtol=1e-6)
    assert runtime.cache.position == 1


def test_checkpoint_preparation_is_atomic_and_verifies_weights(decoder, monkeypatch, tmp_path):
    parser = importlib.import_module("decoder_checkpoint")
    source = tmp_path / "model"
    source.mkdir()
    config = {
        "model_type": "llama", "hidden_act": "silu", "hidden_size": 4, "intermediate_size": 8,
        "num_attention_heads": 2, "num_key_value_heads": 1, "num_hidden_layers": 1,
        "vocab_size": 4, "max_position_embeddings": 128, "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True, "eos_token_id": 2,
    }
    (source / "config.json").write_text(json.dumps(config))
    (source / "pytorch_model.bin").write_bytes(b"test checkpoint")
    spec = SimpleNamespace(**parser.checkpoint_spec(config))
    shapes = parser.expected_shapes(spec)
    tensors = {}
    for name, shape in shapes.items():
        key = f"model.layers.0.{parser.BLOCKS[name]}" if name in parser.BLOCKS else parser.SINGLES[name]
        array = np.ones(shape[1:] if name in parser.BLOCKS else shape, dtype=np.float32)
        tensors[key] = SimpleNamespace(shape=array.shape, float=lambda array=array: SimpleNamespace(numpy=lambda: array))

    def load(path, *, map_location, weights_only):
        assert path.name == "pytorch_model.bin" and map_location == "cpu" and weights_only
        return tensors

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=load))
    expected = tensors.pop("model.norm.weight")
    destination = tmp_path / "prepared"
    with pytest.raises(KeyError):
        parser.prepare(source, destination, "fixture")
    assert not destination.exists()
    tensors["model.norm.weight"] = expected
    parser.prepare(source, destination, "fixture")
    weights = parser.load_prepared(destination)
    assert weights.output is weights.embed
    assert weights.parameters == sum(int(np.prod(shape)) for shape in shapes.values())
    del weights
    with (destination / "wq.npy").open("ab") as stream:
        stream.write(b"trailing bytes")
    with pytest.raises(ValueError, match="hash disagrees for wq"):
        parser.load_prepared(destination)
