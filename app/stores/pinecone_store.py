"""Pinecone VectorStore connector.

Reads old vectors from an existing Pinecone index and writes mapped vectors to a
NEW index, behind the same VectorStore interface as everything else.

``pinecone`` (v5+) is imported lazily so the rest of the framework runs without it.

Built for the real-world case (e.g. a Pinecone *integrated-inference* index whose
`fetch` may not return raw dense values):
  - reads page through `list` + `fetch`;
  - if a fetched record has no dense `values`, the old vector is regenerated from
    the chunk text in metadata via Pinecone inference (deterministic, complete);
  - `upsert` creates the destination index on first write and (by default) copies
    each id's original metadata from the source index, so downstream retrieval that
    relies on metadata (text, section numbers, citations) keeps working.

The source index is only ever read — never modified.
"""

from __future__ import annotations

import time
from typing import Iterator, Sequence

import numpy as np

from app.stores.base import VECTOR_DTYPE, VectorBatch, VectorStore
from app.utils.logging import get_logger

logger = get_logger(__name__)


class PineconeStore(VectorStore):
    def __init__(
        self,
        index_name: str,
        api_key: str | None = None,
        host: str | None = None,
        namespace: str = "",
        text_metadata_key: str = "page_content",
        regenerate_model: str | None = "llama-text-embed-v2",
        copy_metadata: bool = True,
        cloud: str = "aws",
        region: str = "us-east-1",
        metric: str = "cosine",
        client: object | None = None,
    ) -> None:
        if client is not None:
            self.pc = client
        else:
            try:
                from pinecone import Pinecone
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "PineconeStore requires the pinecone SDK. Install it: pip install pinecone"
                ) from exc
            self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.namespace = namespace
        self.text_metadata_key = text_metadata_key
        self.regenerate_model = regenerate_model
        self.copy_metadata = copy_metadata
        self.cloud = cloud
        self.region = region
        self.metric = metric
        self._host = host
        self._index = self._index_handle(index_name)
        self._index_cache: dict[str, object] = {index_name: self._index}

    # ------------------------------------------------------------------ #
    # Handles
    # ------------------------------------------------------------------ #
    def _index_handle(self, name: str):
        if name == self.index_name and self._host:
            return self.pc.Index(host=self._host)
        return self.pc.Index(name)

    def _handle(self, name: str):
        if name not in self._index_cache:
            self._index_cache[name] = self._index_handle(name)
        return self._index_cache[name]

    # ------------------------------------------------------------------ #
    # VectorStore: reads
    # ------------------------------------------------------------------ #
    @property
    def dim(self) -> int:
        return int(self.pc.describe_index(self.index_name).dimension)

    def count(self) -> int:
        stats = self._index.describe_index_stats()
        namespaces = getattr(stats, "namespaces", None) or {}
        if self.namespace and self.namespace in namespaces:
            return int(getattr(namespaces[self.namespace], "vector_count", 0))
        return int(getattr(stats, "total_vector_count", 0))

    def _list_ids(self) -> Iterator[list[str]]:
        token = None
        while True:
            resp = self._index.list_paginated(
                namespace=self.namespace, limit=100, pagination_token=token
            )
            items = getattr(resp, "vectors", None) or []
            ids = [getattr(v, "id", v.get("id") if isinstance(v, dict) else v) for v in items]
            if not ids:
                break
            yield ids
            pagination = getattr(resp, "pagination", None)
            token = getattr(pagination, "next", None) if pagination else None
            if not token:
                break

    #: Pinecone fetch sends ids in the GET query string; keep batches small enough
    #: to stay under the server's URI-length limit (414 otherwise).
    _FETCH_MAX = 100

    def iter_vectors(self, batch_size: int = 1000) -> Iterator[VectorBatch]:
        fetch_size = min(max(1, batch_size), self._FETCH_MAX)
        buf: list[str] = []
        for page in self._list_ids():
            buf.extend(page)
            while len(buf) >= fetch_size:
                yield self._fetch_batch(buf[:fetch_size])
                buf = buf[fetch_size:]
        if buf:
            yield self._fetch_batch(buf)

    def _fetch_batch(self, ids: list[str]) -> VectorBatch:
        resp = self._index.fetch(ids=ids, namespace=self.namespace)
        records = getattr(resp, "vectors", None) or {}
        present = [(i, records[i]) for i in ids if i in records]

        def _values(rec):
            return getattr(rec, "values", None) if not isinstance(rec, dict) else rec.get("values")

        def _metadata(rec):
            return (getattr(rec, "metadata", None) if not isinstance(rec, dict) else rec.get("metadata")) or {}

        has_values = present and all(_values(r) is not None and len(_values(r)) for _, r in present)
        if has_values:
            out_ids = [i for i, _ in present]
            vecs = np.array([_values(r) for _, r in present], dtype=VECTOR_DTYPE)
            return VectorBatch(out_ids, vecs)

        # Fallback: regenerate old vectors from the chunk text via Pinecone inference.
        if not self.regenerate_model:
            raise RuntimeError(
                "fetched records have no dense values and regenerate_model is unset; "
                "cannot obtain old vectors"
            )
        out_ids = [i for i, _ in present]
        texts = [_metadata(r).get(self.text_metadata_key, "") for _, r in present]
        if any(not t for t in texts):
            raise RuntimeError(
                f"missing '{self.text_metadata_key}' metadata for some ids; cannot regenerate vectors"
            )
        vecs = self._inference_embed(texts, input_type="passage")
        return VectorBatch(out_ids, vecs)

    def _inference_embed(self, texts: list[str], input_type: str) -> np.ndarray:
        res = self.pc.inference.embed(
            model=self.regenerate_model,
            inputs=texts,
            parameters={"input_type": input_type, "truncate": "END"},
        )
        data = getattr(res, "data", None) or res
        out = []
        for d in data:
            v = getattr(d, "values", None) if not isinstance(d, dict) else d.get("values")
            out.append(v)
        return np.array(out, dtype=VECTOR_DTYPE)

    # ------------------------------------------------------------------ #
    # VectorStore: writes
    # ------------------------------------------------------------------ #
    def upsert(self, collection: str, ids: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=VECTOR_DTYPE)
        ids = [str(i) for i in ids]
        dim = int(vectors.shape[1])
        self._ensure_index(collection, dim)
        target = self._handle(collection)

        md_map: dict[str, dict] = {}
        if self.copy_metadata:
            md_map = self._fetch_metadata(ids)

        # 2048-dim float vectors ~8KB each; keep batches well under Pinecone's 2MB cap.
        batch = 200
        for s in range(0, len(ids), batch):
            chunk = ids[s:s + batch]
            items = [
                (vid, vectors[s + j].tolist(), md_map.get(vid, {}))
                for j, vid in enumerate(chunk)
            ]
            target.upsert(vectors=items, namespace=self.namespace)

    def _fetch_metadata(self, ids: list[str]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for s in range(0, len(ids), self._FETCH_MAX):
            resp = self._index.fetch(ids=ids[s:s + self._FETCH_MAX], namespace=self.namespace)
            records = getattr(resp, "vectors", None) or {}
            for vid, rec in records.items():
                md = (getattr(rec, "metadata", None) if not isinstance(rec, dict) else rec.get("metadata")) or {}
                out[str(vid)] = md
        return out

    def _ensure_index(self, name: str, dim: int) -> None:
        if self._exists(name):
            return
        from pinecone import ServerlessSpec

        logger.info("creating Pinecone index '%s' (dim=%d, metric=%s)", name, dim, self.metric)
        self.pc.create_index(
            name=name,
            dimension=dim,
            metric=self.metric,
            spec=ServerlessSpec(cloud=self.cloud, region=self.region),
        )
        # Wait until the new index is ready before writing.
        for _ in range(60):
            try:
                desc = self.pc.describe_index(name)
                if getattr(getattr(desc, "status", None), "ready", False):
                    return
            except Exception:  # noqa: BLE001
                pass
            time.sleep(1.0)

    def _exists(self, name: str) -> bool:
        try:
            indexes = self.pc.list_indexes()
            names = indexes.names() if hasattr(indexes, "names") else [
                getattr(x, "name", x) for x in indexes
            ]
            return name in names
        except Exception:  # noqa: BLE001
            try:
                self.pc.describe_index(name)
                return True
            except Exception:  # noqa: BLE001
                return False
