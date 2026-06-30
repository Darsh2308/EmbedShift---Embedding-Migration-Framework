"""File-based VectorStore — the default backend for v1.

The user exports their old vectors to a file (``.jsonl`` / ``.npz`` / ``.npy``) and
points us at it. This is universal (every vector DB can export), needs no auth or
network, and is correct because the back-catalog is frozen during migration.

Reading strategy:
  - ``.npy``  is memory-mapped, so iterating doesn't pull the whole array into RAM.
  - ``.npz``  is loaded once into memory (numpy archives don't memory-map cleanly).
  - ``.jsonl`` is streamed line by line; sampling uses reservoir sampling so memory
              stays at O(sample) rather than O(corpus).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np

from app.stores.base import (
    VECTOR_DTYPE,
    SamplePairs,
    VectorBatch,
    VectorStore,
    make_sample_pairs,
)
from app.stores.formats import (
    detect_format,
    load_texts,
    load_vectors,
    save_vectors,
    stream_jsonl,
)


class FileStore(VectorStore):
    def __init__(self, path: str | Path, *, output_dir: str | Path | None = None) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Vector file not found: {self.path}")
        self._fmt = detect_format(self.path)
        # Where upsert() writes if given a bare collection name.
        self.output_dir = Path(output_dir) if output_dir is not None else self.path.parent

        # jsonl is streamed; npy/npz support random access (mmap / in-memory).
        self._random_access = self._fmt != ".jsonl"
        self._ids: list[str] | None = None
        self._vectors: np.ndarray | None = None
        self._count: int | None = None
        self._dim: int | None = None

        if self._random_access:
            self._load_random_access()
        else:
            self._scan_jsonl()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _load_random_access(self) -> None:
        if self._fmt == ".npy":
            # Memory-map so we don't hold the full corpus in RAM.
            self._vectors = np.load(self.path, mmap_mode="r")
            sidecar = self.path.with_suffix(".ids.json")
            if sidecar.exists():
                import json

                self._ids = [str(i) for i in json.loads(sidecar.read_text(encoding="utf-8"))]
            else:
                self._ids = [str(i) for i in range(self._vectors.shape[0])]
        else:  # .npz
            self._ids, self._vectors = load_vectors(self.path)
        self._count = int(self._vectors.shape[0])
        self._dim = int(self._vectors.shape[1]) if self._vectors.ndim == 2 else 0

    def _scan_jsonl(self) -> None:
        count = 0
        dim = 0
        for _id, vec in stream_jsonl(self.path):
            if count == 0:
                dim = int(vec.shape[0])
            count += 1
        self._count = count
        self._dim = dim

    # ------------------------------------------------------------------ #
    # VectorStore interface
    # ------------------------------------------------------------------ #
    @property
    def dim(self) -> int:
        return int(self._dim or 0)

    def count(self) -> int:
        return int(self._count or 0)

    def iter_vectors(self, batch_size: int = 1000) -> Iterator[VectorBatch]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self._random_access:
            assert self._ids is not None and self._vectors is not None
            for start in range(0, self._count or 0, batch_size):
                end = min(start + batch_size, self._count)
                yield VectorBatch(
                    ids=self._ids[start:end],
                    vectors=np.asarray(self._vectors[start:end], dtype=VECTOR_DTYPE),
                )
            return

        # jsonl: stream and chunk
        batch_ids: list[str] = []
        batch_vecs: list[np.ndarray] = []
        for vid, vec in stream_jsonl(self.path):
            batch_ids.append(vid)
            batch_vecs.append(vec)
            if len(batch_ids) >= batch_size:
                yield VectorBatch(batch_ids, np.array(batch_vecs, dtype=VECTOR_DTYPE))
                batch_ids, batch_vecs = [], []
        if batch_ids:
            yield VectorBatch(batch_ids, np.array(batch_vecs, dtype=VECTOR_DTYPE))

    def fetch_sample(self, n: int, seed: int | None = None) -> VectorBatch:
        if n <= 0:
            raise ValueError("sample size n must be positive")
        total = self.count()
        n = min(n, total)
        rng = np.random.default_rng(seed)

        if self._random_access:
            assert self._ids is not None and self._vectors is not None
            idx = np.sort(rng.choice(total, size=n, replace=False))
            ids = [self._ids[i] for i in idx]
            vectors = np.asarray(self._vectors[idx], dtype=VECTOR_DTYPE)
            return VectorBatch(ids, vectors)

        # jsonl: reservoir sampling in a single streaming pass (O(n) memory)
        res_ids: list[str] = []
        res_vecs: list[np.ndarray] = []
        for i, (vid, vec) in enumerate(stream_jsonl(self.path)):
            if i < n:
                res_ids.append(vid)
                res_vecs.append(vec)
            else:
                j = int(rng.integers(0, i + 1))
                if j < n:
                    res_ids[j] = vid
                    res_vecs[j] = vec
        return VectorBatch(res_ids, np.array(res_vecs, dtype=VECTOR_DTYPE))

    def upsert(self, collection: str, ids: Sequence[str], vectors: np.ndarray) -> None:
        """Write mapped vectors to ``collection``.

        ``collection`` may be a full path or a bare name (resolved under
        ``output_dir``, defaulting to ``.npz``).
        """
        dest = Path(collection)
        if dest.suffix.lower() not in (".jsonl", ".npz", ".npy"):
            dest = self.output_dir / f"{collection}.npz"
        save_vectors(dest, list(ids), np.asarray(vectors, dtype=VECTOR_DTYPE))

    # ------------------------------------------------------------------ #
    # Convenience: sample + attach source text -> training pairs
    # ------------------------------------------------------------------ #
    def fetch_sample_pairs(
        self,
        n: int,
        texts: str | Path | Mapping[str, str],
        seed: int | None = None,
    ) -> SamplePairs:
        """Sample ``n`` records and attach their source text (from a path or mapping)."""
        text_map = load_texts(texts) if isinstance(texts, (str, Path)) else texts
        batch = self.fetch_sample(n, seed=seed)
        return make_sample_pairs(batch, text_map)
