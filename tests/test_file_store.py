"""Tests for the file-based VectorStore and format round-trips."""

import json

import numpy as np
import pytest

from app.stores import FileStore, load_vectors, save_vectors
from app.stores.base import make_sample_pairs
from app.stores.synthetic import make_old_vectors

FORMATS = [".jsonl", ".npz", ".npy"]


def _write(tmp_path, fmt, n=50, dim=8, with_text=False):
    ids, vectors, texts = make_old_vectors(n, dim, seed=1, with_text=with_text)
    path = tmp_path / f"vecs{fmt}"
    save_vectors(path, ids, vectors, texts=texts)
    return path, ids, vectors, texts


# --------------------------------------------------------------------------- #
# Format round-trips
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt", FORMATS)
def test_save_load_roundtrip(tmp_path, fmt):
    path, ids, vectors, _ = _write(tmp_path, fmt)
    got_ids, got_vecs = load_vectors(path)
    assert got_ids == ids
    assert got_vecs.shape == vectors.shape
    np.testing.assert_allclose(got_vecs, vectors, rtol=1e-5)
    assert got_vecs.dtype == np.float32


def test_unsupported_format_rejected(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("nope")
    with pytest.raises(ValueError):
        FileStore(p)


# --------------------------------------------------------------------------- #
# Core VectorStore interface
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fmt", FORMATS)
def test_count_and_dim(tmp_path, fmt):
    path, ids, vectors, _ = _write(tmp_path, fmt, n=37, dim=16)
    store = FileStore(path)
    assert store.count() == 37
    assert store.dim == 16


@pytest.mark.parametrize("fmt", FORMATS)
def test_iter_vectors_covers_everything(tmp_path, fmt):
    path, ids, vectors, _ = _write(tmp_path, fmt, n=50, dim=8)
    store = FileStore(path)

    seen_ids = []
    seen_vecs = []
    n_batches = 0
    for batch in store.iter_vectors(batch_size=7):
        n_batches += 1
        assert len(batch) <= 7
        seen_ids.extend(batch.ids)
        seen_vecs.append(batch.vectors)

    assert n_batches == 8  # 50 / 7 -> 7 full + 1 partial
    assert seen_ids == ids
    np.testing.assert_allclose(np.vstack(seen_vecs), vectors, rtol=1e-5)


@pytest.mark.parametrize("fmt", FORMATS)
def test_fetch_sample_size_and_membership(tmp_path, fmt):
    path, ids, vectors, _ = _write(tmp_path, fmt, n=100, dim=8)
    store = FileStore(path)

    sample = store.fetch_sample(10, seed=42)
    assert len(sample) == 10
    assert sample.vectors.shape == (10, 8)
    assert set(sample.ids).issubset(set(ids))
    assert len(set(sample.ids)) == 10  # no duplicates


@pytest.mark.parametrize("fmt", FORMATS)
def test_fetch_sample_is_reproducible(tmp_path, fmt):
    path, *_ = _write(tmp_path, fmt, n=100, dim=8)
    store = FileStore(path)
    a = store.fetch_sample(15, seed=7)
    b = store.fetch_sample(15, seed=7)
    assert a.ids == b.ids
    np.testing.assert_array_equal(a.vectors, b.vectors)


def test_fetch_sample_caps_at_count(tmp_path):
    path, ids, *_ = _write(tmp_path, ".npz", n=5, dim=4)
    store = FileStore(path)
    sample = store.fetch_sample(100, seed=0)
    assert len(sample) == 5


@pytest.mark.parametrize("fmt", FORMATS)
def test_sampled_vectors_match_source(tmp_path, fmt):
    """A sampled id's vector must equal that id's vector in the source."""
    path, ids, vectors, _ = _write(tmp_path, fmt, n=60, dim=8)
    id_to_vec = {i: vectors[k] for k, i in enumerate(ids)}
    store = FileStore(path)
    sample = store.fetch_sample(12, seed=3)
    for k, sid in enumerate(sample.ids):
        np.testing.assert_allclose(sample.vectors[k], id_to_vec[sid], rtol=1e-5)


# --------------------------------------------------------------------------- #
# Upsert (write-back)
# --------------------------------------------------------------------------- #
def test_upsert_by_name_writes_npz(tmp_path):
    path, ids, vectors, _ = _write(tmp_path, ".npz", n=20, dim=8)
    store = FileStore(path, output_dir=tmp_path)
    store.upsert("corpus_v2", ids, vectors)

    out = tmp_path / "corpus_v2.npz"
    assert out.exists()
    got_ids, got_vecs = load_vectors(out)
    assert got_ids == ids
    np.testing.assert_allclose(got_vecs, vectors, rtol=1e-5)


def test_upsert_by_path_respects_extension(tmp_path):
    path, ids, vectors, _ = _write(tmp_path, ".npz", n=10, dim=4)
    store = FileStore(path)
    dest = tmp_path / "mapped.jsonl"
    store.upsert(str(dest), ids, vectors)
    assert dest.exists()
    got_ids, _ = load_vectors(dest)
    assert got_ids == ids


# --------------------------------------------------------------------------- #
# Sample pairs (old vector + source text)
# --------------------------------------------------------------------------- #
def test_fetch_sample_pairs_aligns_text(tmp_path):
    path, ids, vectors, texts = _write(tmp_path, ".jsonl", n=40, dim=8, with_text=True)
    text_map = {i: t for i, t in zip(ids, texts)}

    store = FileStore(path)
    pairs = store.fetch_sample_pairs(10, text_map, seed=5)

    assert len(pairs) == 10
    assert pairs.old_vectors.shape == (10, 8)
    for sid, txt in zip(pairs.ids, pairs.texts):
        assert text_map[sid] == txt


def test_make_sample_pairs_missing_text_raises(tmp_path):
    path, ids, vectors, _ = _write(tmp_path, ".npz", n=10, dim=4)
    store = FileStore(path)
    batch = store.fetch_sample(5, seed=0)
    with pytest.raises(KeyError):
        make_sample_pairs(batch, {})  # no texts at all


def test_fetch_sample_pairs_from_jsonl_text_file(tmp_path):
    path, ids, vectors, texts = _write(tmp_path, ".npz", n=30, dim=8)
    # Source text provided as a separate jsonl file (only for sampled ids in practice).
    text_path = tmp_path / "texts.jsonl"
    with text_path.open("w") as f:
        for i in ids:
            f.write(json.dumps({"id": i, "text": f"text for {i}"}) + "\n")

    store = FileStore(path)
    pairs = store.fetch_sample_pairs(8, text_path, seed=1)
    assert len(pairs) == 8
    for sid, txt in zip(pairs.ids, pairs.texts):
        assert txt == f"text for {sid}"
