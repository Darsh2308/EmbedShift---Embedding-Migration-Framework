"""Tests for the MLP mapper and the auto linear->MLP upgrade."""

import numpy as np
import pytest

from app.core.evaluation import evaluate_mapper
from app.core.mapper import LinearMapper, load_mapper
from app.core.mlp import MLPMapper
from app.core.pipeline import select_mapper
from app.models.migration import MigrationConfig
from app.stores.synthetic import make_related_spaces


# --------------------------------------------------------------------------- #
# MLP basics
# --------------------------------------------------------------------------- #
def test_mlp_fits_and_transforms_shapes():
    old, new = make_related_spaces(800, 24, 40, noise=0.02, seed=0)
    m = MLPMapper(hidden=64, epochs=100, seed=0).fit(old, new)
    assert m.d_old == 24 and m.d_new == 40
    out = m.transform(old)
    assert out.shape == (800, 40)
    assert np.all(np.isfinite(out))


def test_mlp_single_vector_and_normalize():
    old, new = make_related_spaces(400, 16, 16, noise=0.05, seed=1)
    m = MLPMapper(hidden=32, epochs=80, normalize_output=True, seed=0).fit(old, new)
    out = m.transform(old[0])
    assert out.ndim == 1 and out.shape == (16,)
    batch = m.transform(old[:10])
    np.testing.assert_allclose(np.linalg.norm(batch, axis=1), np.ones(10), atol=1e-5)


def test_mlp_save_load_roundtrip(tmp_path):
    old, new = make_related_spaces(300, 16, 24, noise=0.05, seed=2)
    m = MLPMapper(hidden=32, n_layers=2, epochs=60, normalize_output=True, seed=0).fit(old, new)
    path = m.save(tmp_path / "mlp")
    loaded = load_mapper(path)
    assert isinstance(loaded, MLPMapper)
    np.testing.assert_allclose(m.transform(old), loaded.transform(old), rtol=1e-5, atol=1e-5)


def test_mlp_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        MLPMapper().transform(np.zeros((2, 4)))


def test_mlp_bad_activation_raises():
    with pytest.raises(ValueError):
        MLPMapper(activation="sigmoid")


# --------------------------------------------------------------------------- #
# The key property: MLP beats linear on a nonlinear relationship
# --------------------------------------------------------------------------- #
def test_mlp_beats_linear_on_nonlinear_data():
    old, new = make_related_spaces(
        3000, 32, 48, noise=0.02, nonlinearity=0.8, seed=0
    )
    tr, te = slice(0, 2400), slice(2400, 3000)

    lin = LinearMapper(normalize_output=True).fit(old[tr], new[tr], lambda_=1.0)
    mlp = MLPMapper(hidden=128, epochs=300, normalize_output=True, seed=0).fit(old[tr], new[tr])

    q_lin = evaluate_mapper(old[te], new[te], lin, k=10, max_queries=None).quality_retained
    q_mlp = evaluate_mapper(old[te], new[te], mlp, k=10, max_queries=None).quality_retained

    assert q_mlp > q_lin


# --------------------------------------------------------------------------- #
# Auto-selection: linear chosen on easy data, MLP on hard data
# --------------------------------------------------------------------------- #
def test_select_mapper_keeps_linear_on_easy_data():
    old, new = make_related_spaces(1500, 32, 32, noise=0.02, nonlinearity=0.0, seed=0)

    cfg = MigrationConfig(sample_fraction=1.0, validation_fraction=0.3, k=10, confidence_threshold=0.8)
    from app.core.pipeline import SampleSplit

    split = SampleSplit(old[:1000], new[:1000], old[1000:1500], new[1000:1500], 1500, 1000, 500)
    mapper, kind, attempts = select_mapper(split, cfg)
    assert kind == "linear"


def test_select_mapper_upgrades_to_mlp_on_hard_data():
    old, new = make_related_spaces(3000, 32, 48, noise=0.02, nonlinearity=0.8, seed=0)
    from app.core.pipeline import SampleSplit

    cfg = MigrationConfig(
        validation_fraction=0.3, k=10, confidence_threshold=0.95,
        mlp_hidden=128, mlp_epochs=250,
    )
    split = SampleSplit(old[:2000], new[:2000], old[2000:3000], new[2000:3000], 3000, 2000, 1000)
    mapper, kind, attempts = select_mapper(split, cfg)
    assert kind == "mlp"
    assert [a["kind"] for a in attempts] == ["linear", "mlp"]
    assert attempts[1]["quality_retained"] > attempts[0]["quality_retained"]
