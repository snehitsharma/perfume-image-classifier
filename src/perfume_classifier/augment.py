"""Turn clean catalog shots into photo-like queries at three difficulty levels.

Every query is generated from a seeded RNG, so the same seed always gives the same query set.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage

LEVELS = ("easy", "medium", "hard")


def background_mask(img: Image.Image, threshold: int = 235) -> np.ndarray:
    """Near-white pixels connected to the image border (the studio background, not a white label)."""
    a = np.asarray(img.convert("RGB"))
    white = a.min(axis=2) > threshold
    labels, _ = ndimage.label(white)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    border = border[border != 0]
    return np.isin(labels, border)


def random_background(size: tuple[int, int], rng: np.random.Generator) -> Image.Image:
    """A solid colour, a gradient or a noisy texture, standing in for a table or a wall."""
    w, h = size
    kind = rng.integers(3)
    c1, c2 = rng.integers(0, 256, 3), rng.integers(0, 256, 3)
    if kind == 0:
        arr = np.broadcast_to(c1, (h, w, 3)).astype(np.float32)
    elif kind == 1:
        t = np.linspace(0, 1, h)[:, None, None]
        arr = np.broadcast_to(c1 * (1 - t) + c2 * t, (h, w, 3)).astype(np.float32)
    else:
        small = rng.integers(0, 256, (max(h // 16, 2), max(w // 16, 2), 3)).astype(np.uint8)
        arr = np.asarray(Image.fromarray(small).resize((w, h), Image.BILINEAR), dtype=np.float32)
        arr = arr * 0.5 + c1 * 0.5
    arr = arr + rng.normal(0, 8, (h, w, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def replace_background(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    img = img.convert("RGB")
    mask = Image.fromarray((background_mask(img) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1))
    return Image.composite(random_background(img.size, rng), img, mask)


def random_crop(img: Image.Image, rng: np.random.Generator, min_scale: float, max_scale: float) -> Image.Image:
    w, h = img.size
    s = rng.uniform(min_scale, max_scale)
    cw, ch = int(w * s), int(h * s)
    x, y = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
    return img.crop((x, y, x + cw, y + ch))


def jitter(img: Image.Image, rng: np.random.Generator, strength: float) -> Image.Image:
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        img = enhancer(img).enhance(1 + rng.uniform(-strength, strength))
    return img


def colour_cast(img: Image.Image, rng: np.random.Generator, strength: float) -> Image.Image:
    a = np.asarray(img, dtype=np.float32) * (1 + rng.uniform(-strength, strength, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def make_query(img: Image.Image, level: str, rng: np.random.Generator) -> Image.Image:
    """Apply the augmentations for one difficulty level."""
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}, expected one of {LEVELS}")
    img = img.convert("RGB")
    if level == "easy":
        img = random_crop(img, rng, 0.85, 1.0)
        img = jitter(img, rng, 0.1)
        return jpeg(img, 85)
    if level == "medium":
        img = img.rotate(rng.uniform(-15, 15), resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
        img = random_crop(img, rng, 0.7, 1.0)
        img = jitter(img, rng, 0.25)
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.5)))
        return jpeg(img, int(rng.integers(60, 81)))
    # hard: new background, stronger geometry and lighting, low resolution
    img = replace_background(img, rng)
    img = img.rotate(rng.uniform(-25, 25), resample=Image.BICUBIC, expand=True,
                     fillcolor=tuple(int(c) for c in rng.integers(0, 256, 3)))
    img = random_crop(img, rng, 0.6, 0.9)
    img = jitter(img, rng, 0.35)
    img = colour_cast(img, rng, 0.2)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.5)))
    w, h = img.size
    small = 160 / max(w, h)
    img = img.resize((max(int(w * small), 1), max(int(h * small), 1)), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    return jpeg(img, int(rng.integers(40, 71)))
