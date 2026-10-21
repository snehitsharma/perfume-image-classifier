import numpy as np
import pytest
from PIL import Image

from perfume_classifier.augment import LEVELS, background_mask, make_query
from perfume_classifier.baseline import DIM, colour_histogram
from perfume_classifier.retrieval import hit_rank, same_label_rank, search, topk_accuracy


def bottle(colour: tuple[int, int, int] = (200, 30, 30)) -> Image.Image:
    img = Image.new("RGB", (100, 140), (255, 255, 255))
    img.paste(Image.new("RGB", (40, 80), colour), (30, 30))
    img.paste(Image.new("RGB", (10, 10), (255, 255, 255)), (45, 60))  # white label inside the bottle
    return img


def test_search_returns_sorted_top_k() -> None:
    index = np.eye(4, dtype=np.float32)
    queries = np.array([[0.1, 0.9, 0.0, 0.4]], dtype=np.float32)
    idx, scores = search(queries, index, k=3)
    assert idx.tolist() == [[1, 3, 0]]
    assert np.all(np.diff(scores[0]) <= 0)


def test_search_caps_k_at_index_size_and_batches() -> None:
    rng = np.random.default_rng(0)
    index = rng.normal(size=(5, 3))
    queries = rng.normal(size=(7, 3))
    idx, _ = search(queries, index, k=50, batch=2)
    assert idx.shape == (7, 5)
    assert idx[:, 0].tolist() == (queries @ index.T).argmax(axis=1).tolist()


def test_search_rejects_mismatched_dims() -> None:
    with pytest.raises(ValueError):
        search(np.zeros((2, 3)), np.zeros((4, 5)), k=1)


def test_hit_rank_and_topk() -> None:
    top = np.array([[3, 1, 2], [0, 2, 1], [5, 6, 7]])
    ranks = hit_rank(top, np.array([3, 1, 9]))
    assert ranks.tolist() == [1, 3, 0]
    assert topk_accuracy(ranks, ks=(1, 3)) == pytest.approx({"top1": 1 / 3, "top3": 2 / 3})


def test_topk_rejects_empty() -> None:
    with pytest.raises(ValueError):
        topk_accuracy(np.array([]))


def test_hit_rank_needs_one_target_per_query() -> None:
    with pytest.raises(ValueError):
        hit_rank(np.zeros((2, 3), dtype=int), np.array([1]))


def test_same_label_rank_counts_namesakes() -> None:
    labels = np.array(["a", "b", "a", "c"])
    top = np.array([[2, 1], [3, 1]])
    assert same_label_rank(top, np.array([0, 0]), labels).tolist() == [1, 0]


def test_background_mask_keeps_white_label_inside_bottle() -> None:
    mask = background_mask(bottle())
    assert mask[0, 0] and mask[-1, -1]
    assert not mask[65, 50]  # the white label is not connected to the border
    assert not mask[40, 40]


@pytest.mark.parametrize("level", LEVELS)
def test_make_query_is_deterministic(level: str) -> None:
    a = make_query(bottle(), level, np.random.default_rng(1))
    b = make_query(bottle(), level, np.random.default_rng(1))
    assert a.mode == "RGB"
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_make_query_rejects_unknown_level() -> None:
    with pytest.raises(ValueError):
        make_query(bottle(), "extreme", np.random.default_rng(0))


def test_make_query_accepts_rgba() -> None:
    assert make_query(bottle().convert("RGBA"), "hard", np.random.default_rng(0)).mode == "RGB"


def test_histogram_is_unit_norm_and_ignores_white_background() -> None:
    h = colour_histogram(bottle())
    assert h.shape == (DIM,)
    assert np.linalg.norm(h) == pytest.approx(1.0)
    wider = Image.new("RGB", (300, 300), (255, 255, 255))
    wider.paste(bottle(), (100, 80))
    assert float(h @ colour_histogram(wider)) > 0.95
    assert float(h @ colour_histogram(bottle((30, 30, 200)))) < 0.5


def test_histogram_of_all_white_image_is_valid() -> None:
    h = colour_histogram(Image.new("RGB", (20, 20), (255, 255, 255)))
    assert np.isfinite(h).all() and np.linalg.norm(h) == pytest.approx(1.0)
