"""Step 4 — transform the whole corpus through the mapper (the instant cutover).

Streams every old vector from the store, maps it (light matrix math, not the
heavy new model), and writes the result to a NEW output file — never the source.

Built for large corpora:
  - Batched streaming: memory stays at O(batch), not O(corpus).
  - Resumable: a checkpoint records how many records and bytes were committed, so
    an interrupted run continues from where it stopped (O(1) recovery via
    byte-offset truncation, no full-file rescan).

Output format is ``.jsonl`` ({"id", "vector"}) — appendable (hence resumable) and
loadable by FileStore / any vector DB ingestion. The mapped vectors are the
permanent answer, not a placeholder.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.core.mapper import BaseMapper
from app.models.migration import TransformSummary
from app.stores.base import VectorStore
from app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TransformProgress:
    written: int
    total: int
    batches: int

    @property
    def fraction(self) -> float:
        return self.written / self.total if self.total else 1.0


def _ckpt_path(output_path: Path) -> Path:
    return output_path.with_suffix(".ckpt.json")


def _write_ckpt(path: Path, written: int, total: int, batch_size: int, nbytes: int, done: bool) -> None:
    path.write_text(
        json.dumps(
            {"written": written, "total": total, "batch_size": batch_size, "bytes": nbytes, "done": done}
        ),
        encoding="utf-8",
    )


def transform_corpus(
    store: VectorStore,
    mapper: BaseMapper,
    output_path: str | Path,
    batch_size: int = 1000,
    resume: bool = False,
    progress_cb: Callable[[TransformProgress], None] | None = None,
) -> TransformSummary:
    """Map every vector in ``store`` and stream the results to ``output_path``."""
    output_path = Path(output_path)
    if output_path.suffix != ".jsonl":
        output_path = output_path.with_suffix(".jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = _ckpt_path(output_path)

    total = store.count()

    # ---- establish starting point (fresh vs resume) ----
    written = 0
    resumed = False
    if resume and ckpt.exists() and output_path.exists():
        meta = json.loads(ckpt.read_text(encoding="utf-8"))
        written = int(meta.get("written", 0))
        committed_bytes = int(meta.get("bytes", 0))
        # Drop any partial trailing batch written after the last checkpoint.
        os.truncate(output_path, committed_bytes)
        resumed = written > 0
        logger.info("Resuming transform at %d/%d records", written, total)
    else:
        if output_path.exists():
            output_path.unlink()
        if ckpt.exists():
            ckpt.unlink()

    # ---- stream, map, append ----
    skip = written
    seen = 0
    batches = 0
    with output_path.open("a", encoding="utf-8") as f:
        for batch in store.iter_vectors(batch_size=batch_size):
            bsize = len(batch)
            if seen + bsize <= skip:  # whole batch already written
                seen += bsize
                continue
            start = max(0, skip - seen)  # partial-batch resume offset
            seen += bsize

            ids = batch.ids[start:]
            if not ids:
                continue
            mapped = mapper.transform(batch.vectors[start:])
            buf = "\n".join(
                json.dumps({"id": vid, "vector": mapped[i].tolist()}) for i, vid in enumerate(ids)
            )
            f.write(buf + "\n")
            f.flush()
            os.fsync(f.fileno())

            written += len(ids)
            batches += 1
            _write_ckpt(ckpt, written, total, batch_size, f.tell(), done=False)
            if progress_cb:
                progress_cb(TransformProgress(written, total, batches))

    _write_ckpt(ckpt, written, total, batch_size, output_path.stat().st_size, done=True)
    logger.info("Transform complete: %d vectors -> %s", written, output_path)

    return TransformSummary(
        output_path=str(output_path),
        n_written=written,
        n_total=total,
        batches=batches,
        resumed=resumed,
        done=True,
    )


def transform_to_store(
    source: VectorStore,
    mapper: BaseMapper,
    dest: VectorStore,
    collection: str,
    batch_size: int = 1000,
    progress_cb: Callable[[TransformProgress], None] | None = None,
) -> TransformSummary:
    """Map every vector in ``source`` and upsert the result into ``dest[collection]``.

    For databases, ``upsert`` is idempotent by id, so re-running is safe (a simple,
    robust form of resume) — never overwrites the source collection.
    """
    total = source.count()
    written = 0
    batches = 0
    for batch in source.iter_vectors(batch_size=batch_size):
        mapped = mapper.transform(batch.vectors)
        dest.upsert(collection, batch.ids, mapped)
        written += len(batch)
        batches += 1
        if progress_cb:
            progress_cb(TransformProgress(written, total, batches))

    logger.info("Transform complete: %d vectors -> collection '%s'", written, collection)
    return TransformSummary(
        output_path=collection,
        n_written=written,
        n_total=total,
        batches=batches,
        resumed=False,
        done=True,
    )
