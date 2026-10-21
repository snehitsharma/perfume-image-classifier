"""Fragrance-family classifier on image features, and its evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def split(labels: np.ndarray, groups: np.ndarray, test_size: float, seed: int, by_group: bool) -> tuple[np.ndarray, np.ndarray]:
    """Train and test row indices. With by_group, no group (brand) appears on both sides."""
    rows = np.arange(len(labels))
    if by_group:
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(rows, groups=groups))
        return train, test
    train, test = train_test_split(rows, test_size=test_size, random_state=seed, stratify=labels)
    return train, test


def logistic(C: float, balanced: bool, seed: int) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, max_iter=3000, class_weight="balanced" if balanced else None, random_state=seed),
    )


def majority() -> DummyClassifier:
    return DummyClassifier(strategy="most_frequent")


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        "per_class_f1": dict(zip(classes, f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0).round(4).tolist())),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
    }
