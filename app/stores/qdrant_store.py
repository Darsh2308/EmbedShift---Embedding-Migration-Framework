"""Qdrant VectorStore connector.

Reads old vectors from a Qdrant collection and writes mapped vectors back to a
new collection — behind the same VectorStore interface as everything else, so the
pipeline is unchanged.

``qdrant-client`` is an optional dependency (``pip install qdrant-client``),
imported lazily so the rest of the framework works without it. Supports a real
server (``url``/``api_key``), an on-disk local DB (``location="path"``), and an
in-process DB (``location=":memory:"``) handy for tests.

Original (possibly non-integer) ids are preserved in each point's payload under
``_id``; the Qdrant point id is a deterministic UUID derived from it.
"""

from __future__ import annotations

import uuid
from typing import Iterator, Optional, Sequence

import numpy as np

from app.stores.base import VECTOR_DTYPE, VectorBatch, VectorStore

_ID_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-0000000000ed")


def _point_id(original_id: str) -> str:
    return str(uuid.uuid5(_ID_NAMESPACE, str(original_id)))


class QdrantStore(VectorStore):
    def __init__(
        self,
        collection: str,
        location: Optional[str] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        client: object | None = None,
        distance: str = "cosine",
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "QdrantStore requires qdrant-client. Install it: pip install qdrant-client"
            ) from exc

        if client is not None:
            self.client = client
        elif url is not None:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(location=location or ":memory:")
        self.collection = collection
        self._distance = distance

    # ------------------------------------------------------------------ #
    def _exists(self, name: str) -> bool:
        try:
            return bool(self.client.collection_exists(name))
        except AttributeError:  # older clients
            try:
                self.client.get_collection(name)
                return True
            except Exception:
                return False

    @property
    def dim(self) -> int:
        info = self.client.get_collection(self.collection)
        vectors = info.config.params.vectors
        if hasattr(vectors, "size"):
            return int(vectors.size)
        # named vectors: take the first
        return int(next(iter(vectors.values())).size)

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def iter_vectors(self, batch_size: int = 1000) -> Iterator[VectorBatch]:
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                with_vectors=True,
                with_payload=True,
                limit=batch_size,
                offset=offset,
            )
            if not points:
                break
            ids = [
                (p.payload or {}).get("_id", str(p.id)) for p in points
            ]
            vecs = np.array([p.vector for p in points], dtype=VECTOR_DTYPE)
            yield VectorBatch(ids, vecs)
            if offset is None:
                break

    def upsert(self, collection: str, ids: Sequence[str], vectors: np.ndarray) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        vectors = np.asarray(vectors, dtype=VECTOR_DTYPE)
        dim = int(vectors.shape[1])
        if not self._exists(collection):
            dist = Distance.COSINE if self._distance == "cosine" else Distance.DOT
            self.client.create_collection(
                collection, vectors_config=VectorParams(size=dim, distance=dist)
            )
        points = [
            PointStruct(
                id=_point_id(oid), vector=vectors[i].tolist(), payload={"_id": str(oid)}
            )
            for i, oid in enumerate(ids)
        ]
        self.client.upsert(collection, points=points)
