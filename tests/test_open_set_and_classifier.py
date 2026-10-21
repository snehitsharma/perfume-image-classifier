from pathlib import Path

import numpy as np
import pytest

from perfume_classifier.classifier import evaluate, split
from perfume_classifier.features import cached_features, model_features
from perfume_classifier.retrieval import choose_threshold, open_set_rates


def test_open_set_rates_counts_each_outcome() -> None:
    in_scores = np.array([0.9, 0.8, 0.3, 0.95])
    in_correct = np.array([True, False, True, True])
    out_scores = np.array([0.2, 0.85])
    r = open_set_rates(in_scores, in_correct, out_scores, threshold=0.5)
    assert r["in_catalog_correct"] == pytest.approx(0.5)
    assert r["in_catalog_wrong_match"] == pytest.approx(0.25)
    assert r["in_catalog_rejected"] == pytest.approx(0.25)
    assert r["held_out_rejected"] == pytest.approx(0.5)
    assert r["balanced_correct"] == pytest.approx(0.5)


def test_open_set_rates_rejects_empty() -> None:
    with pytest.raises(ValueError):
        open_set_rates(np.array([0.5]), np.array([True]), np.array([]), 0.5)


def test_choose_threshold_separates_clean_scores() -> None:
    in_scores = np.array([0.9, 0.8, 0.85])
    out_scores = np.array([0.3, 0.4])
    t = choose_threshold(in_scores, np.array([True, True, True]), out_scores)
    assert 0.4 < t <= 0.8
    assert open_set_rates(in_scores, np.ones(3, bool), out_scores, t)["balanced_correct"] == 1.0


def test_brand_split_keeps_brands_on_one_side() -> None:
    rng = np.random.default_rng(0)
    groups = np.repeat([f"b{i}" for i in range(50)], 10)
    labels = rng.choice(["A", "B"], size=len(groups))
    train, test = split(labels, groups, 0.2, seed=1, by_group=True)
    assert not set(groups[train]) & set(groups[test])
    assert len(train) + len(test) == len(groups)


def test_random_split_is_stratified_and_seeded() -> None:
    labels = np.array(["A"] * 80 + ["B"] * 20)
    groups = np.arange(100).astype(str)
    a = split(labels, groups, 0.2, seed=3, by_group=False)
    b = split(labels, groups, 0.2, seed=3, by_group=False)
    assert np.array_equal(a[1], b[1])
    assert (labels[a[1]] == "B").sum() == 4


def test_evaluate_handles_missing_class_in_predictions() -> None:
    r = evaluate(np.array(["A", "B", "C"]), np.array(["A", "A", "A"]), ["A", "B", "C"])
    assert r["accuracy"] == pytest.approx(1 / 3)
    assert r["per_class_f1"]["B"] == 0.0
    assert r["confusion_matrix"] == [[1, 0, 0], [1, 0, 0], [1, 0, 0]]


def test_cache_recomputes_when_files_change(tmp_path: Path) -> None:
    f = tmp_path / "a.jpg"
    f.write_bytes(b"x")
    calls = []

    def compute(paths: list[Path]) -> np.ndarray:
        calls.append(1)
        return np.ones((len(paths), 2))

    cached_features([f], tmp_path / "c", "t", compute)
    cached_features([f], tmp_path / "c", "t", compute)
    assert len(calls) == 1
    f.write_bytes(b"xy")
    cached_features([f], tmp_path / "c", "t", compute)
    assert len(calls) == 2


def test_cache_rejects_wrong_row_count(tmp_path: Path) -> None:
    f = tmp_path / "a.jpg"
    f.write_bytes(b"x")
    with pytest.raises(ValueError):
        cached_features([f], tmp_path, "t", lambda ps: np.ones((2, 2)))


def test_model_features_rejects_unknown_model() -> None:
    with pytest.raises(ValueError):
        model_features("resnet", [], cfg=None, name="x")
