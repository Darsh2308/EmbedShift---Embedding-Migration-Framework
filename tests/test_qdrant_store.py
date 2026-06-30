"""End-to-end migration against a local (in-memory) Qdrant instance.

Skipped automatically if qdrant-client isn't installed. Uses Qdrant's no-server
``location=":memory:"`` mode, so it runs anywhere without Docker.
"""

import numpy as np
import pytest

pytest.importorskip("qdrant_client")

from qdrant_client import QdrantClient  # noqa: E402

from app.core.pipeline import run_migration  # noqa: E402
from app.models.migration import MigrationConfig  # noqa: E402
from app.stores.qdrant_store import QdrantStore  # noqa: E402
from app.stores.synthetic import make_related_spaces  # noqa: E402


def test_qdrant_store_crud():
    client = QdrantClient(location=":memory:")
    store = QdrantStore("c", client=client, distance="dot")
    ids = [f"doc-{i}" for i in range(50)]
    vecs = np.random.default_rng(0).standard_normal((50, 8)).astype(np.float32)
    store.upsert("c", ids, vecs)

    assert store.count() == 50
    assert store.dim == 8

    seen = set()
    for batch in store.iter_vectors(batch_size=16):
        seen.update(batch.ids)
    assert seen == set(ids)

    sample = store.fetch_sample(10, seed=1)
    assert len(sample) == 10
    assert set(sample.ids).issubset(set(ids))


def test_qdrant_full_migration_end_to_end(tmp_path):
    client = QdrantClient(location=":memory:")
    N, D_OLD, D_NEW = 1500, 48, 64
    old, new = make_related_spaces(N, D_OLD, D_NEW, noise=0.02, seed=0)
    ids = [f"doc-{i}" for i in range(N)]
    id_to_new = {ids[i]: new[i] for i in range(N)}

    # Seed the source collection (DOT distance preserves raw vectors).
    QdrantStore("old", client=client, distance="dot").upsert("old", ids, old)

    source = QdrantStore("old", client=client)
    dest = QdrantStore("corpus_v2", client=client, distance="cosine")
    embed = lambda texts: np.array([id_to_new[t] for t in texts], dtype=np.float32)

    cfg = MigrationConfig(
        sample_fraction=0.3, validation_fraction=0.3, k=10, confidence_threshold=0.5,
        output_collection="corpus_v2", output_dir=str(tmp_path), artifacts_dir=str(tmp_path / "art"),
    )
    result = run_migration(source, embed, {i: i for i in ids}, cfg, dest_store=dest)

    assert result.report.verdict.passed is True
    assert result.transformed is True
    assert result.output_path == "corpus_v2"

    # The migrated collection exists in the same Qdrant instance with all vectors.
    migrated = QdrantStore("corpus_v2", client=client)
    assert migrated.count() == N
    assert migrated.dim == D_NEW

    # Retrieval works: querying corpus_v2 with the NEW-model vector of a doc
    # returns that doc near the top (queries are embedded with the new model).
    hits = 0
    probe = ids[:50]
    for vid in probe:
        res = client.query_points("corpus_v2", query=id_to_new[vid].tolist(), limit=5).points
        found = {(p.payload or {}).get("_id") for p in res}
        if vid in found:
            hits += 1
    assert hits / len(probe) > 0.8  # mapped index reproduces the new model's retrieval
