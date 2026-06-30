"""Tests for PineconeStore against a fake in-memory Pinecone client (no network).

The fake mimics the pinecone v5 surface the connector uses: describe_index,
Index(...).{describe_index_stats, list_paginated, fetch, upsert}, list_indexes,
create_index, and inference.embed.
"""

import sys
import types

import numpy as np
import pytest

from app.stores.pinecone_store import PineconeStore


# --------------------------------------------------------------------------- #
# Fake Pinecone SDK
# --------------------------------------------------------------------------- #
class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _FakeIndex:
    def __init__(self, backend, name):
        self.backend = backend
        self.name = name

    def _data(self):
        return self.backend.indexes[self.name]

    def describe_index_stats(self):
        return _Obj(total_vector_count=len(self._data()), namespaces={})

    def list_paginated(self, namespace="", limit=100, pagination_token=None):
        ids = list(self._data().keys())
        start = int(pagination_token or 0)
        page = ids[start:start + limit]
        nxt = str(start + limit) if start + limit < len(ids) else None
        return _Obj(vectors=[_Obj(id=i) for i in page], pagination=_Obj(next=nxt))

    def fetch(self, ids, namespace=""):
        data = self._data()
        vectors = {i: _Obj(values=data[i]["values"], metadata=data[i]["metadata"]) for i in ids if i in data}
        return _Obj(vectors=vectors)

    def upsert(self, vectors, namespace=""):
        data = self._data()
        for vid, values, metadata in vectors:
            data[vid] = {"values": list(values), "metadata": dict(metadata or {})}


class _FakeInference:
    def embed(self, model=None, inputs=None, parameters=None):
        # deterministic 1024-dim "llama" vectors derived from text length
        data = [_Obj(values=[float(len(t) % 7 + 1)] * 1024) for t in inputs]
        return _Obj(data=data)


class FakePinecone:
    def __init__(self, indexes, dims):
        self.indexes = indexes            # {name: {id: {"values","metadata"}}}
        self.dims = dims                  # {name: dim}
        self.inference = _FakeInference()
        self.created = []

    def describe_index(self, name):
        return _Obj(dimension=self.dims.get(name, 0), status=_Obj(ready=True))

    def Index(self, name=None, host=None):
        return _FakeIndex(self, name)

    def list_indexes(self):
        return _Obj(names=lambda: list(self.indexes.keys()))

    def create_index(self, name, dimension, metric, spec=None):
        self.created.append((name, dimension, metric))
        self.indexes.setdefault(name, {})
        self.dims[name] = dimension


def _install_fake_pinecone_module(monkeypatch):
    """Provide `from pinecone import ServerlessSpec` for _ensure_index."""
    fake = types.ModuleType("pinecone")
    fake.ServerlessSpec = lambda **kw: _Obj(**kw)
    fake.Pinecone = FakePinecone
    monkeypatch.setitem(sys.modules, "pinecone", fake)


def _seed_source(n=50, dim=1024, with_values=True):
    rng = np.random.default_rng(0)
    src = {}
    for i in range(n):
        vid = f"doc-{i}"
        values = rng.standard_normal(dim).tolist() if with_values else None
        src[vid] = {"values": values, "metadata": {"page_content": f"text {i}", "section_number": str(i)}}
    return src


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #
def test_count_and_dim():
    indexes = {"indian-law": _seed_source(40)}
    client = FakePinecone(indexes, {"indian-law": 1024})
    store = PineconeStore("indian-law", client=client)
    assert store.count() == 40
    assert store.dim == 1024


def test_iter_vectors_paginates_and_returns_all():
    indexes = {"indian-law": _seed_source(250)}
    client = FakePinecone(indexes, {"indian-law": 1024})
    store = PineconeStore("indian-law", client=client)

    seen = []
    for batch in store.iter_vectors(batch_size=64):
        seen.extend(batch.ids)
        assert batch.vectors.shape[1] == 1024
    assert sorted(seen) == sorted(indexes["indian-law"].keys())
    assert len(seen) == 250


def test_fetch_sample_reservoir_inherited():
    indexes = {"indian-law": _seed_source(100)}
    client = FakePinecone(indexes, {"indian-law": 1024})
    store = PineconeStore("indian-law", client=client)
    sample = store.fetch_sample(15, seed=1)
    assert len(sample) == 15
    assert set(sample.ids).issubset(set(indexes["indian-law"].keys()))


def test_iter_vectors_regenerates_when_no_values():
    # Integrated-inference index: fetch returns metadata but no raw values.
    indexes = {"indian-law": _seed_source(30, with_values=False)}
    client = FakePinecone(indexes, {"indian-law": 1024})
    store = PineconeStore("indian-law", client=client, regenerate_model="llama-text-embed-v2")

    batches = list(store.iter_vectors(batch_size=16))
    total = sum(len(b) for b in batches)
    assert total == 30
    # regenerated vectors are the fake inference output (1024-dim, finite)
    assert all(b.vectors.shape[1] == 1024 and np.all(np.isfinite(b.vectors)) for b in batches)


# --------------------------------------------------------------------------- #
# Writes
# --------------------------------------------------------------------------- #
def test_upsert_creates_index_and_copies_metadata(monkeypatch):
    _install_fake_pinecone_module(monkeypatch)  # for ServerlessSpec
    indexes = {"indian-law": _seed_source(20)}
    client = FakePinecone(indexes, {"indian-law": 1024})
    store = PineconeStore("indian-law", client=client, copy_metadata=True)

    ids = [f"doc-{i}" for i in range(20)]
    mapped = np.random.default_rng(1).standard_normal((20, 2048)).astype(np.float32)
    store.upsert("indian-law-voyage", ids, mapped)

    assert ("indian-law-voyage", 2048, "cosine") in client.created
    new = indexes["indian-law-voyage"]
    assert len(new) == 20
    # vectors are 2048-dim and original metadata was copied across
    assert len(new["doc-0"]["values"]) == 2048
    assert new["doc-0"]["metadata"]["page_content"] == "text 0"
    assert new["doc-5"]["metadata"]["section_number"] == "5"


def test_upsert_without_metadata_copy(monkeypatch):
    _install_fake_pinecone_module(monkeypatch)
    indexes = {"indian-law": _seed_source(10)}
    client = FakePinecone(indexes, {"indian-law": 1024})
    store = PineconeStore("indian-law", client=client, copy_metadata=False)
    store.upsert("dest", [f"doc-{i}" for i in range(10)], np.zeros((10, 2048), np.float32))
    assert indexes["dest"]["doc-0"]["metadata"] == {}


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def test_make_store_pinecone_requires_collection():
    from app.stores import make_store

    with pytest.raises(ValueError):
        make_store("pinecone")  # missing collection
