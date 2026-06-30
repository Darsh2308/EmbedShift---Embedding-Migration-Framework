"""Tests for the numeric metric helpers."""

import numpy as np
import pytest

from app.core.metrics import l2_normalize, mean_cosine_similarity, mse


def test_l2_normalize_unit_rows():
    X = np.array([[3.0, 4.0], [1.0, 0.0]])
    out = l2_normalize(X)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(out[0], [0.6, 0.8], atol=1e-6)


def test_l2_normalize_handles_zero_row():
    X = np.array([[0.0, 0.0]])
    out = l2_normalize(X)
    assert np.all(np.isfinite(out))


def test_mse_zero_for_identical():
    A = np.random.default_rng(0).standard_normal((10, 5))
    assert mse(A, A) == pytest.approx(0.0)


def test_mse_known_value():
    a = np.array([[0.0, 0.0]])
    b = np.array([[1.0, 1.0]])
    assert mse(a, b) == pytest.approx(1.0)  # mean of [1,1]


def test_cosine_identical_is_one():
    A = np.random.default_rng(1).standard_normal((8, 6))
    assert mean_cosine_similarity(A, A) == pytest.approx(1.0, abs=1e-6)


def test_cosine_opposite_is_minus_one():
    A = np.array([[1.0, 2.0, 3.0]])
    assert mean_cosine_similarity(A, -A) == pytest.approx(-1.0, abs=1e-6)


def test_cosine_orthogonal_is_zero():
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    assert mean_cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mse(np.zeros((2, 3)), np.zeros((2, 4)))
