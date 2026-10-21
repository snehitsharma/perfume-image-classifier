"""Nearest-neighbour search over normalised feature vectors and retrieval scoring."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def search(queries: np.ndarray, index: np.ndarray, k: int, batch: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Top-k catalog indices and scores for each query, by dot product (cosine for normalised rows)."""
    if queries.ndim != 2 or index.ndim != 2 or queries.shape[1] != index.shape[1]:
        raise ValueError(f"shape mismatch: queries {queries.shape}, index {index.shape}")
    k = min(k, index.shape[0])
    all_idx, all_scores = [], []
    for start in range(0, len(queries), batch):
        sims = queries[start:start + batch] @ index.T
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        top_scores = np.take_along_axis(sims, top, axis=1)
        order = np.argsort(-top_scores, axis=1, kind="stable")
        all_idx.append(np.take_along_axis(top, order, axis=1))
        all_scores.append(np.take_along_axis(top_scores, order, axis=1))
    return np.vstack(all_idx), np.vstack(all_scores)


def hit_rank(top_idx: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """1-based rank of the target in each row of top_idx, or 0 if it is not there."""
    if len(top_idx) != len(targets):
        raise ValueError("one target per query is required")
    hits = top_idx == np.asarray(targets)[:, None]
    return np.where(hits.any(axis=1), hits.argmax(axis=1) + 1, 0)


def topk_accuracy(ranks: np.ndarray, ks: Sequence[int] = (1, 5, 10)) -> dict[str, float]:
    """Share of queries whose target is within the top k."""
    ranks = np.asarray(ranks)
    if ranks.size == 0:
        raise ValueError("no queries to score")
    return {f"top{k}": float(((ranks >= 1) & (ranks <= k)).mean()) for k in ks}


def same_label_rank(top_idx: np.ndarray, targets: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Like hit_rank, but a hit is any catalog item with the same label as the target (e.g. brand + name)."""
    labels = np.asarray(labels)
    hits = labels[top_idx] == labels[np.asarray(targets)][:, None]
    return np.where(hits.any(axis=1), hits.argmax(axis=1) + 1, 0)
