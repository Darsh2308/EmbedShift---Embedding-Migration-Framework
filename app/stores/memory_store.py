"""In-memory VectorStore — a DB-like backend with named collections.

Mirrors how a real vector DB works (one client, many collections) but holds
everything in process memory. Useful as a reference backend, for tests, and as a
write-back destination when migrating between collections.
"""

from __future__ import annotations

from typing import Iterator, Optional, Sequence

import numpy as np

from app.stores.base import VECTOR_DTYPE, VectorBatch, VectorStore


class InMemoryBackend:
    """Holds multiple collections, like a vector-DB instance."""

    def __init__(self) -> None:
        self.collections: dict[str, tuple[list[str], np.ndarray]] = {}


class InMemoryVectorStore(VectorStore):
    def __init__(
        self,
        collection: str,
        backend: Optional[InMemoryBackend] = None,
        ids: Optional[Sequence[str]] = None,
        vectors: Optional[np.ndarray] = None,
    ) -> None:
        self.collection = collection
        self.backend = backend if backend is not None else InMemoryBackend()
        if ids is not None and vectors is not None:
            self.backend.collections[collection] = (
                list(ids),
                np.asarray(vectors, dtype=VECTOR_DTYPE),
            )

    def _data(self) -> tuple[list[str], np.ndarray]:
        if self.collection not in self.backend.collections:
            raise KeyError(f"collection '{self.collection}' not found")
        return self.backend.collections[self.collection]

    @property
    def dim(self) -> int:
        _, v = self._data()
        return int(v.shape[1]) if v.ndim == 2 and v.size else 0

    def count(self) -> int:
        return len(self._data()[0])

    def iter_vectors(self, batch_size: int = 1000) -> Iterator[VectorBatch]:
        ids, vectors = self._data()
        for start in range(0, len(ids), batch_size):
            end = min(start + batch_size, len(ids))
            yield VectorBatch(ids[start:end], np.asarray(vectors[start:end], dtype=VECTOR_DTYPE))

    def upsert(self, collection: str, ids: Sequence[str], vectors: np.ndarray) -> None:
        ids = [str(i) for i in ids]
        vectors = np.asarray(vectors, dtype=VECTOR_DTYPE)
        if collection not in self.backend.collections:
            self.backend.collections[collection] = (ids, vectors)
            return
        # Merge with existing (upsert semantics: replace by id, append new).
        ex_ids, ex_vecs = self.backend.collections[collection]
        index = {vid: k for k, vid in enumerate(ex_ids)}
        ex_ids = list(ex_ids)
        ex_vecs = list(ex_vecs)
        for i, vid in enumerate(ids):
            if vid in index:
                ex_vecs[index[vid]] = vectors[i]
            else:
                index[vid] = len(ex_ids)
                ex_ids.append(vid)
                ex_vecs.append(vectors[i])
        self.backend.collections[collection] = (ex_ids, np.array(ex_vecs, dtype=VECTOR_DTYPE))
