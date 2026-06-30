"""Tests for the full-corpus transform, including resumability."""

import numpy as np
import pytest

from app.core.mapper import LinearMapper
from app.core.transform import transform_corpus
from app.stores import FileStore, load_vectors, save_vectors
from app.stores.synthetic import make_related_spaces


def _store_and_mapper(tmp_path, n=500, d_old=24, d_new=32, seed=0):
    old, new = make_related_spaces(n, d_old, d_new, noise=0.02, seed=seed)
    ids = [f"doc-{i}" for i in range(n)]
    save_vectors(tmp_path / "old.npz", ids, old)
    store = FileStore(tmp_path / "old.npz")
    mapper = LinearMapper().fit(old, new, lambda_=1.0)
    return store, mapper, ids, old


def test_transform_writes_all_in_order(tmp_path):
    store, mapper, ids, old = _store_and_mapper(tmp_path, n=300)
    out = tmp_path / "mapped.jsonl"
    summary = transform_corpus(store, mapper, out, batch_size=64)

    assert summary.n_written == 300
    assert summary.n_total == 300
    assert summary.done is True

    out_ids, out_vecs = load_vectors(out)
    assert out_ids == ids  # order preserved
    np.testing.assert_allclose(out_vecs, mapper.transform(old), rtol=1e-4, atol=1e-5)


def test_transform_fresh_run_clears_stale_state(tmp_path):
    store, mapper, ids, old = _store_and_mapper(tmp_path, n=120)
    out = tmp_path / "mapped.jsonl"
    # leave a stale, wrong file + checkpoint behind
    out.write_text('{"id": "stale", "vector": [1,2,3]}\n')
    out.with_suffix(".ckpt.json").write_text('{"written": 1, "bytes": 10}')

    summary = transform_corpus(store, mapper, out, batch_size=50, resume=False)
    out_ids, _ = load_vectors(out)
    assert summary.n_written == 120
    assert "stale" not in out_ids


def test_transform_resume_completes_after_crash(tmp_path):
    store, mapper, ids, old = _store_and_mapper(tmp_path, n=500)

    # Full reference run.
    full = tmp_path / "full.jsonl"
    transform_corpus(store, mapper, full, batch_size=50)
    full_ids, full_vecs = load_vectors(full)

    # Simulated crash: progress callback raises after 3 batches.
    class Boom(Exception):
        pass

    calls = {"n": 0}

    def cb(_p):
        calls["n"] += 1
        if calls["n"] == 3:
            raise Boom()

    part = tmp_path / "part.jsonl"
    with pytest.raises(Boom):
        transform_corpus(store, mapper, part, batch_size=50, progress_cb=cb)

    partial_ids, _ = load_vectors(part)
    assert 0 < len(partial_ids) < 500  # genuinely partial

    # Resume to completion.
    summary = transform_corpus(store, mapper, part, batch_size=50, resume=True)
    assert summary.n_written == 500
    assert summary.resumed is True

    res_ids, res_vecs = load_vectors(part)
    assert res_ids == full_ids
    np.testing.assert_allclose(res_vecs, full_vecs, rtol=1e-5)


def test_resume_discards_partial_trailing_write(tmp_path):
    store, mapper, ids, old = _store_and_mapper(tmp_path, n=200)
    out = tmp_path / "mapped.jsonl"

    # Crash after 2 batches.
    class Boom(Exception):
        pass

    calls = {"n": 0}

    def cb(_p):
        calls["n"] += 1
        if calls["n"] == 2:
            raise Boom()

    with pytest.raises(Boom):
        transform_corpus(store, mapper, out, batch_size=40, progress_cb=cb)

    # Simulate a torn write after the last checkpoint: append a partial line.
    with out.open("a", encoding="utf-8") as f:
        f.write('{"id": "torn", "vec')

    summary = transform_corpus(store, mapper, out, batch_size=40, resume=True)
    assert summary.n_written == 200
    out_ids, out_vecs = load_vectors(out)  # would raise if a torn line survived
    assert "torn" not in out_ids
    np.testing.assert_allclose(out_vecs, mapper.transform(old), rtol=1e-4, atol=1e-5)
