"""End-to-end tests for the migration pipeline (sample -> train -> gate -> transform)."""

from pathlib import Path

import numpy as np
import pytest

from app.core.mapper import load_mapper
from app.core.pipeline import run_migration
from app.models.migration import MigrationConfig
from app.stores import FileStore, load_vectors, save_vectors
from app.stores.synthetic import make_related_spaces


def _build_case(tmp_path, n=1500, d_old=48, d_new=64, noise=0.02, nonlinearity=0.0, seed=0):
    """A store of old vectors + a synthetic 'new model' that embeds text(=id) -> new vec."""
    old, new = make_related_spaces(n, d_old, d_new, noise=noise, nonlinearity=nonlinearity, seed=seed)
    ids = [f"doc-{i}" for i in range(n)]
    save_vectors(tmp_path / "old.npz", ids, old)
    store = FileStore(tmp_path / "old.npz")

    id_to_new = {ids[i]: new[i] for i in range(n)}
    texts = {i: i for i in ids}  # the "source text" of a doc is just its id here

    def embed(batch_texts):
        return np.array([id_to_new[t] for t in batch_texts], dtype=np.float32)

    return store, embed, texts, ids, old


def _cfg(tmp_path, **overrides):
    base = dict(
        sample_fraction=0.3,
        validation_fraction=0.3,
        k=10,
        output_dir=str(tmp_path),
        artifacts_dir=str(tmp_path / "artifacts"),
        output_collection="corpus_v2",
        batch_size=200,
    )
    base.update(overrides)
    return MigrationConfig(**base)


def test_pipeline_passes_and_transforms(tmp_path):
    store, embed, texts, ids, old = _build_case(tmp_path, n=1500, noise=0.02)
    cfg = _cfg(tmp_path, confidence_threshold=0.5)

    result = run_migration(store, embed, texts, cfg)

    assert result.report.verdict.passed is True
    assert result.transformed is True
    assert result.n_transformed == store.count()
    assert result.sample_size == 450 and result.val_size == 135 and result.train_size == 315

    # all artifacts exist
    assert Path(result.output_path).exists()
    assert Path(result.mapper_path).exists()
    assert Path(result.report_path).exists()

    # transformed output equals the saved mapper applied to the old vectors
    out_ids, out_vecs = load_vectors(result.output_path)
    assert len(out_ids) == store.count()
    mapper = load_mapper(result.mapper_path)
    id_to_out = {out_ids[i]: out_vecs[i] for i in range(len(out_ids))}
    for j in (0, 250, 900, 1499):
        np.testing.assert_allclose(
            id_to_out[ids[j]], mapper.transform(old[j]), rtol=1e-4, atol=1e-4
        )


def test_pipeline_fails_gate_and_skips_transform(tmp_path):
    store, embed, texts, ids, old = _build_case(tmp_path, n=1500, noise=0.05, nonlinearity=0.9)
    cfg = _cfg(tmp_path, confidence_threshold=0.95, mapper_kind="linear")

    result = run_migration(store, embed, texts, cfg)

    assert result.report.verdict.passed is False
    assert result.transformed is False
    assert result.skipped_reason is not None
    assert result.output_path is None
    assert not (tmp_path / "corpus_v2.jsonl").exists()
    # artifacts (mapper + report) are still saved for inspection
    assert Path(result.mapper_path).exists()
    assert Path(result.report_path).exists()


def test_pipeline_force_transforms_despite_failed_gate(tmp_path):
    store, embed, texts, ids, old = _build_case(tmp_path, n=1500, noise=0.05, nonlinearity=0.9)
    cfg = _cfg(tmp_path, confidence_threshold=0.95, force=True, mapper_kind="linear")

    result = run_migration(store, embed, texts, cfg)

    assert result.report.verdict.passed is False
    assert result.transformed is True  # forced
    assert result.n_transformed == store.count()
    assert Path(result.output_path).exists()


def test_pipeline_validation_slice_too_small_raises(tmp_path):
    store, embed, texts, ids, old = _build_case(tmp_path, n=400)
    cfg = _cfg(tmp_path, sample_size=30, validation_fraction=0.2, k=10)  # val=6 <= k
    with pytest.raises(ValueError):
        run_migration(store, embed, texts, cfg)


def test_pipeline_empty_store_raises(tmp_path):
    save_vectors(tmp_path / "empty.jsonl", [], np.empty((0, 8), dtype=np.float32))
    store = FileStore(tmp_path / "empty.jsonl")
    with pytest.raises(ValueError):
        run_migration(store, lambda t: np.zeros((len(t), 8), np.float32), {}, _cfg(tmp_path))
