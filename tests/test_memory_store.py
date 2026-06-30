"""Tests for the in-memory (DB-like) store, store-to-store transform, and DB pipeline path."""

import time

import numpy as np

from app import services
from app.core.mapper import LinearMapper
from app.core.pipeline import run_migration
from app.core.transform import transform_to_store
from app.embedders import register_embedder
from app.models.migration import MigrationConfig
from app.stores import InMemoryBackend, InMemoryVectorStore
from app.stores.synthetic import make_related_spaces


def _seed(n=1500, d_old=48, d_new=64, noise=0.02, nonlinearity=0.0, seed=0):
    old, new = make_related_spaces(n, d_old, d_new, noise=noise, nonlinearity=nonlinearity, seed=seed)
    ids = [f"doc-{i}" for i in range(n)]
    backend = InMemoryBackend()
    source = InMemoryVectorStore("old", backend, ids, old)
    id_to_new = {ids[i]: new[i] for i in range(n)}
    embed = lambda texts: np.array([id_to_new[t] for t in texts], dtype=np.float32)
    return backend, source, ids, old, embed


# --------------------------------------------------------------------------- #
# Interface conformance
# --------------------------------------------------------------------------- #
def test_memory_store_interface():
    backend, source, ids, old, _ = _seed(n=200, d_old=8, d_new=8)
    assert source.count() == 200
    assert source.dim == 8

    seen = []
    for batch in source.iter_vectors(batch_size=64):
        seen.extend(batch.ids)
    assert seen == ids

    sample = source.fetch_sample(20, seed=1)
    assert len(sample) == 20
    assert set(sample.ids).issubset(set(ids))


def test_memory_store_upsert_creates_and_merges():
    backend = InMemoryBackend()
    store = InMemoryVectorStore("a", backend, ["x"], np.ones((1, 3), np.float32))
    store.upsert("b", ["p", "q"], np.zeros((2, 3), np.float32))
    assert "b" in backend.collections
    # upsert again: replace p, add r
    store.upsert("b", ["p", "r"], np.ones((2, 3), np.float32))
    b = InMemoryVectorStore("b", backend)
    assert b.count() == 3  # p, q, r


# --------------------------------------------------------------------------- #
# Store-to-store transform
# --------------------------------------------------------------------------- #
def test_transform_to_store():
    backend, source, ids, old, _ = _seed(n=300, d_old=16, d_new=24)
    new = make_related_spaces(300, 16, 24, seed=0)[1]
    mapper = LinearMapper().fit(old, new, lambda_=1.0)

    dest = InMemoryVectorStore("ignored", backend)
    summary = transform_to_store(source, mapper, dest, "corpus_v2", batch_size=64)
    assert summary.n_written == 300

    migrated = InMemoryVectorStore("corpus_v2", backend)
    assert migrated.count() == 300
    # spot-check a few ids against the mapper output
    out_map = {migrated._data()[0][i]: migrated._data()[1][i] for i in range(migrated.count())}
    id_to_old = {ids[i]: old[i] for i in range(len(ids))}
    for vid in ("doc-0", "doc-150", "doc-299"):
        np.testing.assert_allclose(out_map[vid], mapper.transform(id_to_old[vid]), rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------- #
# Full pipeline against a DB (programmatic, dest_store)
# --------------------------------------------------------------------------- #
def test_run_migration_with_dest_store(tmp_path):
    backend, source, ids, old, embed = _seed(n=1500, noise=0.02)
    cfg = MigrationConfig(
        sample_fraction=0.3, validation_fraction=0.3, k=10, confidence_threshold=0.5,
        output_collection="corpus_v2", output_dir=str(tmp_path), artifacts_dir=str(tmp_path / "art"),
    )
    result = run_migration(source, embed, None, cfg, dest_store=source)

    assert result.report.verdict.passed is True
    assert result.transformed is True
    assert result.output_path == "corpus_v2"
    assert result.n_transformed == 1500

    migrated = InMemoryVectorStore("corpus_v2", backend)
    assert migrated.count() == 1500
    assert migrated.dim == 64
    # no file was written (DB path)
    assert not (tmp_path / "corpus_v2.jsonl").exists()


# --------------------------------------------------------------------------- #
# Full pipeline against a DB through the service layer (the API's transform path)
# --------------------------------------------------------------------------- #
def test_service_db_transform_path(tmp_path):
    backend, source, ids, old, embed = _seed(n=1500, noise=0.02, seed=5)
    register_embedder("mem-lookup", embed)

    session = services.connect(
        store=source, backend="memory",
        output_dir=str(tmp_path), artifacts_dir=str(tmp_path / "art"),
    )
    services.sample(session, "mem-lookup", sample_fraction=0.3, validation_fraction=0.3)
    services.train(session)
    report = services.evaluate(session, k=10, confidence_threshold=0.5)
    assert report.verdict.passed is True

    job = services.transform(session)
    deadline = time.time() + 10
    while time.time() < deadline:
        j = services.JOBS.get(job.id)
        if j.status in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert j.status == "completed"
    assert j.result["n_transformed"] == 1500

    migrated = InMemoryVectorStore(session.config.output_collection, backend)
    assert migrated.count() == 1500
