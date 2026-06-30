"""Dump an existing Pinecone index to local files for the migration + validation.

Writes:
  - old.npz    : (id, old_vector) for every chunk  (1024-dim llama vectors)
  - texts.jsonl: {"id","text"} for every chunk      (from the index metadata)

These two files give the migration its training-pair text (--texts) with ids
guaranteed to align with the stored vectors, and let the validation harness run
fast/offline. Old vectors are fetched from the index; if the index hides raw
values (integrated inference), they are regenerated from the chunk text via
Pinecone inference.

Usage:
  export PINECONE_API_KEY=...
  python scripts/pinecone_dump.py --index indian-law --host <INDEX_HOST>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from app.stores import save_vectors  # noqa: E402
from app.stores.pinecone_store import PineconeStore  # noqa: E402

_TEXT_KEYS = ["text", "page_content", "chunk_text", "content"]


def _autodetect_text_key(meta: dict) -> str:
    for k in _TEXT_KEYS:
        if k in meta and isinstance(meta[k], str) and meta[k]:
            return k
    # fall back to the first string field
    for k, v in meta.items():
        if isinstance(v, str) and v:
            return k
    return "text"


def main() -> int:
    ap = argparse.ArgumentParser(description="Dump a Pinecone index to old.npz + texts.jsonl")
    ap.add_argument("--index", required=True, help="source index name (e.g. indian-law)")
    ap.add_argument("--host", help="index host (recommended for v5 SDK)")
    ap.add_argument("--api-key", default=os.environ.get("PINECONE_API_KEY"))
    ap.add_argument("--namespace", default="")
    ap.add_argument("--text-key", default=None, help="metadata key holding chunk text (auto if unset)")
    ap.add_argument("--regenerate-model", default="llama-text-embed-v2")
    ap.add_argument("--out-vectors", default="data/old.npz")
    ap.add_argument("--out-texts", default="data/texts.jsonl")
    ap.add_argument("--batch", type=int, default=200)
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("set PINECONE_API_KEY or pass --api-key")

    store = PineconeStore(
        args.index, api_key=args.api_key, host=args.host, namespace=args.namespace,
        text_metadata_key=args.text_key or "text", regenerate_model=args.regenerate_model,
    )
    print(f"connected to '{args.index}' — {store.count()} vectors, dim {store.dim}")

    ids: list[str] = []
    chunks: list[np.ndarray] = []
    for batch in store.iter_vectors(batch_size=args.batch):
        ids.extend(batch.ids)
        chunks.append(batch.vectors)
        print(f"  read {len(ids)} vectors...", end="\r")
    vectors = np.vstack(chunks) if chunks else np.empty((0, store.dim), dtype=np.float32)
    os.makedirs(os.path.dirname(args.out_vectors) or ".", exist_ok=True)
    save_vectors(args.out_vectors, ids, vectors)
    print(f"\nwrote {len(ids)} vectors -> {args.out_vectors}")

    meta = store._fetch_metadata(ids)
    key = args.text_key or _autodetect_text_key(next(iter(meta.values()), {}))
    empty = 0
    with open(args.out_texts, "w", encoding="utf-8") as f:
        for vid in ids:
            text = (meta.get(vid, {}) or {}).get(key, "")
            if not text:
                empty += 1
            f.write(json.dumps({"id": vid, "text": text}) + "\n")
    print(f"wrote {len(ids)} texts -> {args.out_texts} (text key '{key}', {empty} empty)")
    if empty:
        print(f"WARNING: {empty} chunks had no text under '{key}'. Pass --text-key explicitly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
