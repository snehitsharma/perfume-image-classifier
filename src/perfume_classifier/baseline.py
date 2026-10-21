"""Baseline image features: an HSV colour histogram of the non-white pixels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

H_BINS, S_BINS, V_BINS = 16, 4, 4
DIM = H_BINS * S_BINS * V_BINS


def colour_histogram(img: Image.Image, white_threshold: int = 235) -> np.ndarray:
    """Square-rooted, L2-normalised HSV histogram, so a dot product is the Bhattacharyya coefficient.

    Near-white pixels are dropped so the studio background does not dominate every histogram.
    If an image is entirely near-white, all pixels are used.
    """
    img = img.convert("RGB")
    img.thumbnail((128, 128))
    rgb = np.asarray(img)
    hsv = np.asarray(img.convert("HSV")).reshape(-1, 3)
    keep = rgb.reshape(-1, 3).min(axis=1) <= white_threshold
    if keep.any():
        hsv = hsv[keep]
    h = (hsv[:, 0].astype(int) * H_BINS) // 256
    s = (hsv[:, 1].astype(int) * S_BINS) // 256
    v = (hsv[:, 2].astype(int) * V_BINS) // 256
    hist = np.bincount((h * S_BINS + s) * V_BINS + v, minlength=DIM).astype(np.float32)
    hist = np.sqrt(hist / hist.sum())
    return hist / np.linalg.norm(hist)


def histogram_from_path(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return colour_histogram(img)
