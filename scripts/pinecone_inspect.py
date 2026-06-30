"""Read-only sanity check of a Pinecone index before migrating.

Confirms connectivity and answers the two questions the plan flagged:
  1. Does fetch() return raw dense values, or must we regenerate from text?
  2. Which metadata key holds the chunk text?

Usage:
  export PINECONE_API_KEY=...
  python scripts/pinecone_inspect.py --index indian-law           # host auto-resolved
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect a Pinecone index (read-only)")
    ap.add_argument("--index", required=True)
    ap.add_argument("--api-key", default=os.environ.get("PINECONE_API_KEY"))
    ap.add_argument("--namespace", default="")
    args = ap.parse_args()
    if not args.api_key:
        raise SystemExit("set PINECONE_API_KEY or pass --api-key")

    from pinecone import Pinecone

    pc = Pinecone(api_key=args.api_key)
    desc = pc.describe_index(args.index)
    host = getattr(desc, "host", None)
    dim = getattr(desc, "dimension", None)
    metric = getattr(desc, "metric", None)
    print(f"index   : {args.index}")
    print(f"host    : {host}")
    print(f"dim     : {dim}")
    print(f"metric  : {metric}")

    index = pc.Index(host=host) if host else pc.Index(args.index)
    stats = index.describe_index_stats()
    print(f"vectors : {getattr(stats, 'total_vector_count', '?')}")
    print(f"namespaces: {list((getattr(stats, 'namespaces', None) or {}).keys())}")

    # grab one id and fetch it
    page = index.list_paginated(namespace=args.namespace, limit=1)
    items = getattr(page, "vectors", None) or []
    if not items:
        print("\n(no vectors found in this namespace)")
        return 0
    sample_id = getattr(items[0], "id", None) or items[0]
    rec = index.fetch(ids=[sample_id], namespace=args.namespace).vectors[sample_id]

    values = getattr(rec, "values", None)
    metadata = getattr(rec, "metadata", None) or {}
    print(f"\nsample id          : {sample_id}")
    print(f"fetch returns values: {'YES (' + str(len(values)) + '-dim)' if values else 'NO -> will regenerate from text'}")
    print(f"metadata keys      : {list(metadata.keys())}")
    for k, v in metadata.items():
        if isinstance(v, str):
            print(f"  text candidate '{k}': {v[:80]!r}")
    print("\nNext: pick the metadata key that holds the chunk text for --text-key (dump step).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
