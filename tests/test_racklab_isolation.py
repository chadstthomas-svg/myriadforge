import importlib.util
from pathlib import Path

import numpy as np
import pytest


MODULE = Path(__file__).resolve().parents[1] / "labs" / "racklab" / "isolation.py"
SPEC = importlib.util.spec_from_file_location("racklab_isolation", MODULE)
ISOLATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ISOLATION)


def render(samples, block):
    stream = ISOLATION.SpectralIsolation()
    parts = [stream.push(samples[i:i + block]) for i in range(0, len(samples), block)]
    parts.append(stream.flush())
    assert len(stream.flush()) == 0
    return np.concatenate(parts)


def test_arbitrary_chunks_are_aligned_and_match():
    samples = np.random.default_rng(5).normal(0, 0.1, 1879).astype(np.float32)
    whole = render(samples, len(samples))
    assert whole.shape == samples.shape
    for block in (1, 123, 320, 512):
        np.testing.assert_array_equal(render(samples, block), whole)


def test_flush_preserves_short_and_empty_inputs():
    np.testing.assert_array_equal(render(np.zeros(99, dtype=np.float32), 17), np.zeros(99))
    assert len(render(np.empty(0, dtype=np.float32), 1)) == 0


def test_rejects_bad_audio_and_append_after_end():
    stream = ISOLATION.SpectralIsolation()
    with pytest.raises(ValueError, match="finite mono"):
        stream.push(np.array([np.nan]))
    stream.flush()
    with pytest.raises(ValueError, match="after flushing"):
        stream.push(np.zeros(1))
