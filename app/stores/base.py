"""VectorStore interface and the data types it moves around.

A ``VectorStore`` is the framework's single contract for getting old vectors *out*
and writing mapped vectors *in*. Every backend (files now; Pinecone/Qdrant later)
implements this same interface, so the rest of the pipeline never knows or cares
which database is underneath.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

import numpy as np

#: Canonical on-disk / in-memory dtype for vectors.
VECTOR_DTYPE = np.float32


@dataclass
class VectorBatch:
    """A chunk of ``(id, vector)`` records.

    ``vectors`` has shape ``(n, d)`` and ``ids`` has length ``n``.
    """

    ids: list[str]
    vectors: np.ndarray

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError(f"vectors must be 2D (n, d); got shape {self.vectors.shape}")
        if len(self.ids) != self.vectors.shape[0]:
            raise ValueError(
                f"ids ({len(self.ids)}) and vectors ({self.vectors.shape[0]}) length mismatch"
            )
        if self.vectors.dtype != VECTOR_DTYPE:
            self.vectors = self.vectors.astype(VECTOR_DTYPE, copy=False)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])


@dataclass
class SamplePairs:
    """Sampled old vectors aligned with their source text.

    This is the input to Step 1 of the migration: the source text is run through
    the *new* model to produce the matched ``(old_vector, new_vector)`` pairs the
    mapper trains on.
    """

    ids: list[str]
    old_vectors: np.ndarray
    texts: list[str]

    def __post_init__(self) -> None:
        if not (len(self.ids) == self.old_vectors.shape[0] == len(self.texts)):
            raise ValueError("ids, old_vectors, and texts must all be the same length")

    def __len__(self) -> int:
        return len(self.ids)


def make_sample_pairs(batch: VectorBatch, texts: Mapping[str, str]) -> SamplePairs:
    """Attach source text to a sampled batch, preserving id order.

    Raises ``KeyError`` if any sampled id has no text — we never want to train on
    misaligned pairs.
    """
    missing = [i for i in batch.ids if i not in texts]
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(
            f"{len(missing)} sampled id(s) have no source text (e.g. {preview}). "
            "Source text is required for the sample to build training pairs."
        )
    return SamplePairs(
        ids=list(batch.ids),
        old_vectors=batch.vectors,
        texts=[texts[i] for i in batch.ids],
    )


class VectorStore(ABC):
    """Read old vectors out, write mapped vectors in — backend-agnostic."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the stored vectors."""

    @abstractmethod
    def count(self) -> int:
        """Total number of vectors available."""

    @abstractmethod
    def iter_vectors(self, batch_size: int = 1000) -> Iterator[VectorBatch]:
        """Stream every vector in batches of ``batch_size`` (for the full transform)."""

    @abstractmethod
    def upsert(self, collection: str, ids: Sequence[str], vectors: np.ndarray) -> None:
        """Write mapped vectors to ``collection`` (a new destination, never the source)."""

    def fetch_sample(self, n: int, seed: int | None = None) -> VectorBatch:
        """Return a random sample of ``n`` records (the 1-5% we re-embed).

        Default implementation uses reservoir sampling over ``iter_vectors`` — a
        single streaming pass with O(n) memory, so it works on any backend
        (including DBs with no random access). Backends with cheap random access
        (e.g. FileStore) override this for speed.
        """
        if n <= 0:
            raise ValueError("sample size n must be positive")
        rng = np.random.default_rng(seed)
        res_ids: list[str] = []
        res_vecs: list[np.ndarray] = []
        i = 0
        for batch in self.iter_vectors(batch_size=max(256, n)):
            for j in range(len(batch)):
                vec = np.asarray(batch.vectors[j], dtype=VECTOR_DTYPE)
                if i < n:
                    res_ids.append(batch.ids[j])
                    res_vecs.append(vec)
                else:
                    k = int(rng.integers(0, i + 1))
                    if k < n:
                        res_ids[k] = batch.ids[j]
                        res_vecs[k] = vec
                i += 1
        if not res_ids:
            return VectorBatch([], np.empty((0, self.dim), dtype=VECTOR_DTYPE))
        return VectorBatch(res_ids, np.array(res_vecs, dtype=VECTOR_DTYPE))
