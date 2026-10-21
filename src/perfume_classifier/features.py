"""Cache feature matrices on disk, keyed by the input files so a stale cache is never reused."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np


def cache_key(paths: list[Path], extra: str = "") -> str:
    text = extra + "\n" + "\n".join(f"{p.name}:{p.stat().st_size}" for p in paths)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def cached_features(
    paths: list[Path], cache_dir: Path, name: str, compute: Callable[[list[Path]], np.ndarray], extra: str = ""
) -> np.ndarray:
    """Load features for paths from cache_dir, or compute and save them."""
    cache = cache_dir / f"{name}_{cache_key(paths, extra)}.npy"
    if cache.exists():
        return np.load(cache)
    feats = compute(paths)
    if len(feats) != len(paths):
        raise ValueError(f"got {len(feats)} feature rows for {len(paths)} files")
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, feats)
    return feats


def histogram_features(paths: list[Path]) -> np.ndarray:
    from multiprocessing import Pool

    from perfume_classifier.baseline import histogram_from_path

    with Pool() as pool:
        return np.stack(pool.map(histogram_from_path, paths, chunksize=64))


MODELS = ("histogram", "clip")


def model_features(kind: str, paths: list[Path], cfg, name: str) -> np.ndarray:
    """Features for paths from the named model, cached in cfg.features_dir."""
    if kind == "histogram":
        return cached_features(paths, cfg.features_dir, f"histogram_{name}", histogram_features)
    if kind == "clip":
        from perfume_classifier.embed import Embedder

        model_id = f"{cfg.embed_model}/{cfg.embed_pretrained}"

        def compute(ps: list[Path]) -> np.ndarray:
            return Embedder(cfg.embed_model, cfg.embed_pretrained).embed_paths(ps, cfg.embed_batch_size)

        tag = model_id.replace("/", "_")
        return cached_features(paths, cfg.features_dir, f"clip_{tag}_{name}", compute, extra=model_id)
    raise ValueError(f"unknown model {kind!r}, expected one of {MODELS}")
