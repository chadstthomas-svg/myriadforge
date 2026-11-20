"""Deterministic host preprocessing shared by training and single-request inference."""
import hashlib
import re
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def mel_filters():
    hz = np.linspace(0, 8000, 257)
    edges = 700 * (10 ** (np.linspace(0, 2595 * np.log10(1 + 8000 / 700), 42) / 2595) - 1)
    left = (hz[:, None] - edges[:-2]) / (edges[1:-1] - edges[:-2])
    right = (edges[2:] - hz[:, None]) / (edges[2:] - edges[1:-1])
    return np.maximum(0, np.minimum(left, right)).astype(np.float32)


def audio_features(samples, rate):
    if rate != 16000 or samples.ndim != 1 or len(samples) > 16000 or not len(samples):
        raise ValueError("Expected a nonempty mono 16 kHz command clip of at most one second")
    samples = np.pad(samples.astype(np.float32), (0, 16000 - len(samples)))
    offsets = np.arange(98)[:, None] * 160 + np.arange(400)
    frames = samples[offsets] * np.hanning(400)
    power = np.abs(np.fft.rfft(frames, n=512, axis=1)) ** 2
    mel = np.log(np.maximum(power @ mel_filters(), 1e-8))
    return ((mel.T - mel.mean()) / max(mel.std(), 0.1))[None].astype(np.float32)


def text_features(sentence):
    words = re.findall(r"[\w']+", sentence.casefold())
    grams = words + [" ".join(pair) for pair in zip(words, words[1:])]
    vector = np.zeros(4096, dtype=np.float32)
    for gram in grams:
        key = hashlib.blake2s(gram.encode(), digest_size=4).digest()
        vector[int.from_bytes(key, "little") % len(vector)] += 1
    return vector / max(np.linalg.norm(vector), 1)
