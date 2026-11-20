"""Lossless integer words stored in the normal finite FP16 encodings."""
import numpy as np


SIGN_BIT = 0x8000
FIRST_NORMAL = 0x0400
NORMAL_PER_SIGN = 0x7800
CAPACITY = 2 * NORMAL_PER_SIGN


def encode_ranks(ranks):
    ranks = np.asarray(ranks, dtype=np.int32)
    if np.any((ranks < 0) | (ranks >= CAPACITY)):
        raise ValueError("Radix word is outside the finite FP16 code space")
    bits = FIRST_NORMAL + ranks + np.where(ranks < NORMAL_PER_SIGN, 0, SIGN_BIT - NORMAL_PER_SIGN)
    return bits.astype(np.uint16).view(np.float16)


def decode_ranks(words):
    halves = words
    if words.dtype != np.float16:
        if not np.isfinite(words).all() or np.any(np.abs(words) > 65504):
            raise RuntimeError("Radix head returned an invalid FP16 word")
        halves = words.astype(np.float16)
        if not np.array_equal(halves, words):
            raise RuntimeError("Radix head returned an inexact FP16 word")
    bits = halves.view(np.uint16).astype(np.int32)
    magnitude = bits & (SIGN_BIT - 1)
    if np.any((magnitude < FIRST_NORMAL) | (magnitude >= FIRST_NORMAL + NORMAL_PER_SIGN)):
        raise RuntimeError("Radix head returned a non-normal FP16 word")
    return magnitude - FIRST_NORMAL + np.where(bits & SIGN_BIT, NORMAL_PER_SIGN, 0)
