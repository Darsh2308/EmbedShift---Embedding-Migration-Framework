"""Validate a finished migration by re-embedding the full corpus with the new model.

Because the corpus is small, we compute the TRUE Voyage vectors for every chunk
(the ceiling) and measure, using the same recall math the gate uses:

  - recall@k of OLD vectors    vs true-Voyage retrieval  (do-nothing baseline)
  - recall@k of MAPPED vectors vs true-Voyage retrieval  (our method)
  - quality_retained = recall_mapped / recall_max

Optionally, with --questions FILE, it runs real queries (embedded with Voyage,
input_type="query" — never mapped) and reports, per question, the top-k overlap
between the MAPPED index and the full-Voyage index (the ceiling).

Usage:
  export VOYAGE_API_KEY=...
  python scripts/eval_indexes.py --old data/old.npz --texts data/texts.jsonl \
      --mapper artifacts/indian-law-voyage_mapper.npz --k 10 --questions questions.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from app.core.evaluation import evaluate_mapper, recall_at_k, retrieve  # noqa: E402
from app.core.mapper import load_mapper  # noqa: E402
from app.embedders_voyage import make_voyage_embedder  # noqa: E402
from app.stores import load_texts, load_vectors  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare old / mapped / full-Voyage retrieval")
    ap.add_argument("--old", default="data/old.npz", help="old vectors (id, vector)")
    ap.add_argument("--texts", default="data/texts.jsonl", help="id->text jsonl")
    ap.add_argument("--mapper", required=True, help="trained mapper artifact (.npz)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--dim", type=int, default=2048, help="Voyage output dimension")
    ap.add_argument("--questions", help="optional text file, one question per line")
    args = ap.parse_args()

    ids, old = load_vectors(args.old)
    text_map = load_texts(args.texts)
    texts = [text_map[i] for i in ids]
    mapper = load_mapper(args.mapper)

    doc_embed = make_voyage_embedder(input_type="document", output_dimension=args.dim)
    print(f"re-embedding {len(ids)} chunks with Voyage (ground truth)...")
    true = doc_embed(texts)
    mapped = mapper.transform(old)

    print("\n=== Corpus recall vs full-Voyage (the ceiling) ===")
    res = evaluate_mapper(old, true, mapper, k=args.k, max_queries=None)
    print(f"  recall@{args.k}  ceiling (true-Voyage) : {res.recall_at_k_max:.3f}")
    print(f"  recall@{args.k}  mapped (ours)         : {res.recall_at_k_mapped:.3f}")
    print(f"  recall@{args.k}  do-nothing (old)      : {res.recall_at_k_old:.3f}")
    print(f"  QUALITY RETAINED                       : {res.quality_retained:.1%}")
    print(f"  (cosine mapped vs true, FYI)           : {res.mean_cosine_mapped_vs_true:.3f}")

    # recall at several k for a fuller picture
    n = len(ids)
    qidx = np.arange(n)
    print(f"\n  recall@k breakdown (n={n}):")
    for k in (1, 5, 10):
        if k >= n:
            continue
        gold = retrieve(true, true, k, exclude_idx=qidx)
        r_map = recall_at_k(retrieve(true, mapped, k, exclude_idx=qidx), gold)
        r_old = recall_at_k(retrieve(old, old, k, exclude_idx=qidx), gold)
        print(f"    k={k:<3}  mapped={r_map:.3f}   do-nothing={r_old:.3f}")

    if args.questions and os.path.exists(args.questions):
        with open(args.questions, encoding="utf-8") as f:
            questions = [ln.strip() for ln in f if ln.strip()]
        print(f"\n=== Question retrieval: MAPPED vs full-Voyage top-{args.k} ({len(questions)} questions) ===")
        q_embed = make_voyage_embedder(input_type="query", output_dimension=args.dim)
        qv = q_embed(questions)
        top_mapped = retrieve(qv, mapped, args.k)
        top_true = retrieve(qv, true, args.k)
        overlaps = []
        for i, q in enumerate(questions):
            ov = len(set(top_mapped[i].tolist()) & set(top_true[i].tolist())) / args.k
            overlaps.append(ov)
            print(f"  [{ov:4.0%}] {q[:70]}")
        print(f"\n  mean top-{args.k} overlap (mapped vs full-Voyage): {np.mean(overlaps):.1%}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
