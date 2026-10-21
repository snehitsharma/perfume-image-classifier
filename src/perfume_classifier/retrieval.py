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


def open_set_rates(
    in_scores: np.ndarray, in_correct: np.ndarray, out_scores: np.ndarray, threshold: float
) -> dict[str, float]:
    """Outcomes when a top-1 score below threshold means "not in the catalog".

    in_*: queries whose item is in the index (in_correct: was top-1 the right item).
    out_scores: top-1 scores for queries whose item was removed from the index.
    """
    in_scores, in_correct, out_scores = map(np.asarray, (in_scores, in_correct, out_scores))
    if in_scores.size == 0 or out_scores.size == 0:
        raise ValueError("need both in-catalog and held-out queries")
    accepted = in_scores >= threshold
    right_in = float((accepted & in_correct).mean())
    right_out = float((out_scores < threshold).mean())
    return {
        "in_catalog_correct": right_in,
        "in_catalog_wrong_match": float((accepted & ~in_correct).mean()),
        "in_catalog_rejected": float((~accepted).mean()),
        "held_out_rejected": right_out,
        "balanced_correct": (right_in + right_out) / 2,
    }


def choose_threshold(in_scores: np.ndarray, in_correct: np.ndarray, out_scores: np.ndarray) -> float:
    """Threshold that maximises balanced_correct, searched over every observed score."""
    candidates = np.unique(np.concatenate([in_scores, out_scores, [np.inf]]))
    best = max(candidates, key=lambda t: open_set_rates(in_scores, in_correct, out_scores, t)["balanced_correct"])
    return float(best)


def same_label_rank(top_idx: np.ndarray, targets: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Like hit_rank, but a hit is any catalog item with the same label as the target (e.g. brand + name)."""
    labels = np.asarray(labels)
    hits = labels[top_idx] == labels[np.asarray(targets)][:, None]
    return np.where(hits.any(axis=1), hits.argmax(axis=1) + 1, 0)
