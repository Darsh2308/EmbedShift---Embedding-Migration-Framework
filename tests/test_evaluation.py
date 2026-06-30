"""Tests for retrieval, recall@k, and the mapper evaluation."""

import numpy as np
import pytest

from app.core.evaluation import evaluate_mapper, recall_at_k, retrieve
from app.core.mapper import LinearMapper
from app.stores.synthetic import make_related_spaces


# --------------------------------------------------------------------------- #
# recall_at_k
# --------------------------------------------------------------------------- #
def test_recall_perfect():
    gold = np.array([[1, 2, 3], [4, 5, 6]])
    assert recall_at_k(gold, gold) == pytest.approx(1.0)


def test_recall_zero_when_disjoint():
    retrieved = np.array([[7, 8, 9]])
    gold = np.array([[1, 2, 3]])
    assert recall_at_k(retrieved, gold) == pytest.approx(0.0)


def test_recall_partial():
    retrieved = np.array([[1, 2, 99]])  # 2 of 3 correct
    gold = np.array([[1, 2, 3]])
    assert recall_at_k(retrieved, gold) == pytest.approx(2 / 3)


def test_recall_mismatched_query_count_raises():
    with pytest.raises(ValueError):
        recall_at_k(np.zeros((2, 3), int), np.zeros((3, 3), int))


# --------------------------------------------------------------------------- #
# retrieve
# --------------------------------------------------------------------------- #
def test_retrieve_finds_nearest():
    corpus = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    queries = np.array([[1.0, 0.05]], dtype=np.float32)  # closest to row 0
    top = retrieve(queries, corpus, k=1)
    assert top[0, 0] == 0


def test_retrieve_excludes_self():
    corpus = np.eye(4, dtype=np.float32)
    queries = corpus.copy()
    exclude = np.arange(4)
    top = retrieve(queries, corpus, k=1, exclude_idx=exclude)
    # the nearest (itself) is excluded, so it must never return its own index
    for i in range(4):
        assert top[i, 0] != i


# --------------------------------------------------------------------------- #
# evaluate_mapper — the three-way comparison
# --------------------------------------------------------------------------- #
def test_evaluate_good_mapper_beats_do_nothing():
    """A mapper trained on related spaces should retain most quality and beat old."""
    old, new = make_related_spaces(
        n=2500, d_old=48, d_new=64, noise=0.02, nonlinearity=0.0, seed=0
    )
    tr, te = slice(0, 2000), slice(2000, 2500)
    mapper = LinearMapper(normalize_output=True).fit_cv(old[tr], new[tr], metric="cosine")

    res = evaluate_mapper(old[te], new[te], mapper, k=10, max_queries=None, seed=1)

    assert res.recall_at_k_max == pytest.approx(1.0)        # ceiling is the gold itself
    assert res.recall_at_k_mapped > res.recall_at_k_old     # beats do-nothing
    assert res.quality_retained > 0.5                       # retains substantial quality
    assert -1.0 <= res.mean_cosine_mapped_vs_true <= 1.0
    assert res.n_queries == 500 and res.n_corpus == 500
    assert res.mapper.d_old == 48 and res.mapper.d_new == 64


def test_evaluate_bad_mapper_low_quality():
    """Unrelated old/new spaces: the mapper can't learn a useful map -> low recall."""
    rng = np.random.default_rng(3)
    old = rng.standard_normal((1500, 32)).astype(np.float32)
    new = rng.standard_normal((1500, 32)).astype(np.float32)  # independent of old
    tr, te = slice(0, 1000), slice(1000, 1500)
    mapper = LinearMapper().fit(old[tr], new[tr], lambda_=1.0)

    res = evaluate_mapper(old[te], new[te], mapper, k=10, max_queries=None, seed=2)
    assert res.quality_retained < 0.2  # near chance


def test_evaluate_rejects_bad_shapes():
    old, new = make_related_spaces(n=50, d_old=8, d_new=8, seed=0)
    mapper = LinearMapper().fit(old, new, lambda_=1.0)
    with pytest.raises(ValueError):
        evaluate_mapper(old[:30], new[:20], mapper)  # row mismatch
    with pytest.raises(ValueError):
        evaluate_mapper(old[:5], new[:5], mapper, k=10)  # n <= k


def test_max_queries_caps_query_count():
    old, new = make_related_spaces(n=800, d_old=16, d_new=16, seed=0)
    mapper = LinearMapper().fit(old, new, lambda_=1.0)
    res = evaluate_mapper(old, new, mapper, k=5, max_queries=100, seed=0)
    assert res.n_queries == 100
    assert res.n_corpus == 800
