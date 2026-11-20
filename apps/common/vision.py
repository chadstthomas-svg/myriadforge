"""Image/frame preprocessing shared by the gate, footage, and cascade pilots.

Zoo-classifier convention (mobilenetv2-12 et al.): resize the short side to
`scale`, center-crop to the model's square input, RGB, /255 -> (x-0.5)/0.5,
HWC -> NCHW. Input size is read from the compiled model where possible.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    ex = np.exp(shifted)
    return ex / ex.sum()


def _dim_len(dim: Any, fallback: int) -> int:
    """Dimension -> int, honoring the OpenVINO 2022.3 API (get_length)."""
    if not dim.is_static:
        return fallback
    try:
        return int(dim.get_length())
    except AttributeError:
        return int(dim)


def input_hw(compiled: Any, default: int = 224) -> tuple[int, int, int]:
    """(channels, h, w) from the compiled model's first input, with fallbacks."""
    try:
        sig = next(iter(compiled.inputs))
        dims = sig.get_partial_shape()
        c = _dim_len(dims[1], 3)
        h = _dim_len(dims[2], default)
        w = _dim_len(dims[3], default)
        return c, h, w
    except Exception:
        return 3, default, default


def frame_to_nchw(
    frame_bgr: np.ndarray,
    channels: int = 3,
    size: int = 224,
    scale: int = 256,
) -> np.ndarray:
    """BGR frame (H, W, 3 uint8) -> model input (1, C, size, size) float32."""
    import cv2

    h, w = frame_bgr.shape[:2]
    short = min(h, w)
    if short != scale:
        if h < w:
            new_h, new_w = scale, max(scale, int(round(w * scale / h)))
        else:
            new_h, new_w = max(scale, int(round(h * scale / w))), scale
        frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    h, w = frame_bgr.shape[:2]
    y0 = max(0, (h - size) // 2)
    x0 = max(0, (w - size) // 2)
    crop = frame_bgr[y0 : y0 + size, x0 : x0 + size]

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - 0.5) / 0.5
    chw = rgb.transpose(2, 0, 1)
    if channels != 3:  # degenerate single-channel inputs stay honest
        chw = chw[:1]
    return chw[None].astype(np.float32)
