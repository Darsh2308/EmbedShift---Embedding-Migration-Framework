"""Read/write helpers for the file formats the FileStore understands.

Supported (no extra dependencies):
  - ``.jsonl``  one object per line: ``{"id": ..., "vector": [...], "text": "..."}``
                human-readable; ideal for samples and small sets.
  - ``.npz``    numpy archive with ``ids`` and ``vectors`` arrays; compact and fast;
                the recommended format for the bulk corpus.
  - ``.npy``    a single 2D vectors array; ids come from a sibling ``<stem>.ids.json``
                if present, otherwise they default to "0", "1", ...

(``.parquet`` is intentionally deferred to a later phase to avoid pulling in
pyarrow/pandas while we validate the core math.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from app.stores.base import VECTOR_DTYPE

SUPPORTED_FORMATS = (".jsonl", ".npz", ".npy")


def detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )
    return suffix


def _ids_sidecar(path: Path) -> Path:
    return path.with_suffix(".ids.json")


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def save_vectors(
    path: str | Path,
    ids: list[str],
    vectors: np.ndarray,
    texts: list[str] | None = None,
) -> Path:
    """Persist ``(ids, vectors[, texts])`` to ``path``; format chosen by extension."""
    path = Path(path)
    fmt = detect_format(path)
    vectors = np.asarray(vectors, dtype=VECTOR_DTYPE)
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D (n, d); got shape {vectors.shape}")
    if len(ids) != vectors.shape[0]:
        raise ValueError(f"ids ({len(ids)}) and vectors ({vectors.shape[0]}) length mismatch")
    if texts is not None and len(texts) != len(ids):
        raise ValueError("texts length must match ids length")

    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for i, vid in enumerate(ids):
                record = {"id": str(vid), "vector": vectors[i].tolist()}
                if texts is not None:
                    record["text"] = texts[i]
                f.write(json.dumps(record))
                f.write("\n")
    elif fmt == ".npz":
        arrays = {"ids": np.array([str(i) for i in ids]), "vectors": vectors}
        if texts is not None:
            arrays["texts"] = np.array(texts)
        np.savez(path, **arrays)
    elif fmt == ".npy":
        np.save(path, vectors)
        _ids_sidecar(path).write_text(
            json.dumps([str(i) for i in ids]), encoding="utf-8"
        )
    return path


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def load_vectors(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Load all ``(ids, vectors)`` from ``path`` into memory."""
    path = Path(path)
    fmt = detect_format(path)

    if fmt == ".jsonl":
        ids: list[str] = []
        rows: list[list[float]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                ids.append(str(obj["id"]))
                rows.append(obj["vector"])
        vectors = np.array(rows, dtype=VECTOR_DTYPE) if rows else np.empty((0, 0), VECTOR_DTYPE)
        return ids, vectors

    if fmt == ".npz":
        with np.load(path, allow_pickle=False) as data:
            ids = [str(i) for i in data["ids"].tolist()]
            vectors = np.asarray(data["vectors"], dtype=VECTOR_DTYPE)
        return ids, vectors

    # .npy
    vectors = np.asarray(np.load(path), dtype=VECTOR_DTYPE)
    sidecar = _ids_sidecar(path)
    if sidecar.exists():
        ids = [str(i) for i in json.loads(sidecar.read_text(encoding="utf-8"))]
    else:
        ids = [str(i) for i in range(vectors.shape[0])]
    return ids, vectors


def stream_jsonl(path: str | Path) -> Iterator[tuple[str, np.ndarray]]:
    """Yield ``(id, vector)`` one record at a time without loading the whole file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            yield str(obj["id"]), np.asarray(obj["vector"], dtype=VECTOR_DTYPE)


def load_texts(path: str | Path) -> dict[str, str]:
    """Load an id -> text mapping from a ``.jsonl`` file of ``{"id", "text"}`` rows."""
    path = Path(path)
    texts: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts[str(obj["id"])] = obj["text"]
    return texts
