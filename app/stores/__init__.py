"""Data layer: read old vectors out, write mapped vectors in."""

from app.stores.base import (
    SamplePairs,
    VectorBatch,
    VectorStore,
    make_sample_pairs,
)
from app.stores.factory import make_store
from app.stores.file_store import FileStore
from app.stores.formats import (
    load_texts,
    load_vectors,
    save_vectors,
)
from app.stores.memory_store import InMemoryBackend, InMemoryVectorStore

# PineconeStore is intentionally NOT imported here (lazy `pinecone` dependency);
# import it from app.stores.pinecone_store or via make_store("pinecone", ...).

__all__ = [
    "VectorStore",
    "VectorBatch",
    "SamplePairs",
    "make_sample_pairs",
    "FileStore",
    "InMemoryVectorStore",
    "InMemoryBackend",
    "make_store",
    "save_vectors",
    "load_vectors",
    "load_texts",
]
