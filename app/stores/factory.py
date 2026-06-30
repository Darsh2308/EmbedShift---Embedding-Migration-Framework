"""Construct a VectorStore from a backend name + params (used by the API)."""

from __future__ import annotations

from app.stores.base import VectorStore
from app.stores.file_store import FileStore

SUPPORTED_BACKENDS = ("file", "qdrant", "pinecone")


def make_store(backend: str = "file", **params) -> VectorStore:
    """Build a VectorStore for ``backend``.

    file   : source_path (+ optional output_dir)
    qdrant : collection (+ location | url/api_key)
    """
    if backend == "file":
        if "source_path" not in params:
            raise ValueError("file backend requires 'source_path'")
        return FileStore(params["source_path"], output_dir=params.get("output_dir"))

    if backend == "qdrant":
        from app.stores.qdrant_store import QdrantStore

        if "collection" not in params:
            raise ValueError("qdrant backend requires 'collection'")
        return QdrantStore(
            collection=params["collection"],
            location=params.get("location"),
            url=params.get("url"),
            api_key=params.get("api_key"),
        )

    if backend == "pinecone":
        from app.stores.pinecone_store import PineconeStore

        if "collection" not in params:
            raise ValueError("pinecone backend requires 'collection' (the source index name)")
        return PineconeStore(
            index_name=params["collection"],
            api_key=params.get("api_key"),
            host=params.get("host"),
            namespace=params.get("namespace") or "",
            cloud=params.get("cloud") or "aws",
            region=params.get("region") or "us-east-1",
        )

    raise ValueError(f"unknown backend '{backend}'. Supported: {SUPPORTED_BACKENDS}")
