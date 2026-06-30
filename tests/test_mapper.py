"""Tests for the LinearMapper — the mathematical heart of the framework.

These verify the math is correct, generalizes (not just memorizes), handles the
tricky details (mean offset, dimension mismatch, normalization), and persists
losslessly.
"""

import numpy as np
import pytest

from app.core import LinearMapper, load_mapper, mean_cosine_similarity, mse
from app.stores.synthetic import make_linear_pair
from app.utils.numerics import safe_matmul


# --------------------------------------------------------------------------- #
# Correctness: recover a known linear relationship
# --------------------------------------------------------------------------- #
def test_recovers_noiseless_linear_map():
    old, W_true, new = make_linear_pair(n=500, d_old=16, d_new=24, seed=0, noise=0.0)
    m = LinearMapper().fit(old, new, lambda_=1e-6)
    pred = m.transform(old)
    assert mse(pred, new) < 1e-4
    assert mean_cosine_similarity(pred, new) > 0.9999


def test_generalizes_to_unseen_vectors():
    """The whole premise: learn on a sample, map vectors it never trained on."""
    old, W_true, new = make_linear_pair(n=2000, d_old=24, d_new=40, seed=1, noise=0.01)
    tr, te = slice(0, 1500), slice(1500, 2000)
    m = LinearMapper().fit(old[tr], new[tr], lambda_=1e-3)
    pred = m.transform(old[te])  # held-out vectors
    assert mean_cosine_similarity(pred, new[te]) > 0.99


def test_mean_centering_captures_offset():
    """new = old @ W + b: the constant offset must be absorbed by mu_y."""
    rng = np.random.default_rng(2)
    old = rng.standard_normal((800, 12)).astype(np.float32)
    W_true = rng.standard_normal((12, 18)).astype(np.float32)
    b = rng.standard_normal(18).astype(np.float32) * 5.0  # large offset
    with safe_matmul():
        new = (old @ W_true + b).astype(np.float32)

    m = LinearMapper().fit(old, new, lambda_=1e-6)
    pred = m.transform(old)
    assert mse(pred, new) < 1e-3
    # mu_y should be close to the true mean of new (= mean(old)@W + b)
    np.testing.assert_allclose(m.mu_y, new.mean(axis=0), rtol=1e-2, atol=1e-2)


# --------------------------------------------------------------------------- #
# Dimension handling
# --------------------------------------------------------------------------- #
def test_handles_dimension_mismatch():
    old, _, new = make_linear_pair(n=300, d_old=8, d_new=32, seed=3, noise=0.0)
    m = LinearMapper().fit(old, new, lambda_=1e-4)
    assert m.d_old == 8 and m.d_new == 32
    assert m.transform(old).shape == (300, 32)


def test_transform_single_vector_returns_1d():
    old, _, new = make_linear_pair(n=100, d_old=10, d_new=10, seed=4, noise=0.0)
    m = LinearMapper().fit(old, new, lambda_=1e-4)
    out = m.transform(old[0])
    assert out.ndim == 1
    assert out.shape == (10,)


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def test_normalize_output_gives_unit_vectors():
    old, _, new = make_linear_pair(n=200, d_old=16, d_new=16, seed=5, noise=0.1)
    m = LinearMapper(normalize_output=True).fit(old, new, lambda_=1.0)
    out = m.transform(old)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, np.ones(len(norms)), atol=1e-5)


def test_normalize_output_off_by_default():
    old, _, new = make_linear_pair(n=100, d_old=8, d_new=8, seed=6, noise=0.1)
    m = LinearMapper().fit(old, new, lambda_=1.0)
    assert m.normalize_output is False


# --------------------------------------------------------------------------- #
# Cross-validated lambda selection
# --------------------------------------------------------------------------- #
def test_fit_cv_selects_lambda_and_fits_well():
    old, _, new = make_linear_pair(n=1000, d_old=20, d_new=20, seed=7, noise=0.05)
    m = LinearMapper().fit_cv(old, new, folds=5, seed=0, metric="mse")
    assert m.is_fitted
    assert m.lambda_ in m.cv_results_
    assert len(m.cv_results_) == 8  # default grid size
    # the chosen lambda should be the grid minimum for MSE
    assert m.cv_results_[m.lambda_] == min(m.cv_results_.values())
    assert mean_cosine_similarity(m.transform(old), new) > 0.99


def test_fit_cv_cosine_metric_picks_max():
    old, _, new = make_linear_pair(n=600, d_old=12, d_new=12, seed=8, noise=0.05)
    m = LinearMapper().fit_cv(old, new, folds=4, seed=1, metric="cosine")
    assert m.cv_results_[m.lambda_] == max(m.cv_results_.values())


def test_fit_cv_rejects_bad_args():
    old, _, new = make_linear_pair(n=10, d_old=4, d_new=4, seed=9, noise=0.0)
    with pytest.raises(ValueError):
        LinearMapper().fit_cv(old, new, folds=1)  # need >= 2 folds
    with pytest.raises(ValueError):
        LinearMapper().fit_cv(old, new, folds=100)  # more folds than samples
    with pytest.raises(ValueError):
        LinearMapper().fit_cv(old, new, metric="recall")  # unsupported here


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_save_load_roundtrip_identical_predictions(tmp_path):
    old, _, new = make_linear_pair(n=400, d_old=16, d_new=24, seed=10, noise=0.05)
    m = LinearMapper(normalize_output=True).fit(old, new, lambda_=0.5)
    path = m.save(tmp_path / "mapper")  # .npz appended
    assert path.exists()

    loaded = load_mapper(path)
    assert isinstance(loaded, LinearMapper)
    assert loaded.normalize_output is True
    assert loaded.lambda_ == pytest.approx(0.5)
    np.testing.assert_array_equal(m.transform(old), loaded.transform(old))


def test_linearmapper_load_classmethod(tmp_path):
    old, _, new = make_linear_pair(n=100, d_old=8, d_new=8, seed=11, noise=0.0)
    m = LinearMapper().fit(old, new, lambda_=1.0)
    p = m.save(tmp_path / "m.npz")
    loaded = LinearMapper.load(p)
    np.testing.assert_array_equal(m.W, loaded.W)


def test_callable_alias_matches_transform():
    old, _, new = make_linear_pair(n=50, d_old=6, d_new=6, seed=12, noise=0.0)
    m = LinearMapper().fit(old, new, lambda_=1.0)
    np.testing.assert_array_equal(m(old), m.transform(old))


# --------------------------------------------------------------------------- #
# Guard rails
# --------------------------------------------------------------------------- #
def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        LinearMapper().transform(np.zeros((2, 4)))


def test_transform_wrong_dim_raises():
    old, _, new = make_linear_pair(n=50, d_old=8, d_new=8, seed=13, noise=0.0)
    m = LinearMapper().fit(old, new, lambda_=1.0)
    with pytest.raises(ValueError):
        m.transform(np.zeros((3, 5)))  # 5 != d_old 8


def test_fit_mismatched_rows_raises():
    with pytest.raises(ValueError):
        LinearMapper().fit(np.zeros((10, 4)), np.zeros((8, 4)))


def test_fit_negative_lambda_raises():
    old, _, new = make_linear_pair(n=20, d_old=4, d_new=4, seed=14, noise=0.0)
    with pytest.raises(ValueError):
        LinearMapper().fit(old, new, lambda_=-1.0)


def test_more_dims_than_samples_is_stable():
    """d_old > n (few samples, high dim): ridge must stay finite and sane."""
    old, _, new = make_linear_pair(n=20, d_old=100, d_new=50, seed=15, noise=0.0)
    m = LinearMapper().fit(old, new, lambda_=1.0)
    pred = m.transform(old)
    assert np.all(np.isfinite(pred))
