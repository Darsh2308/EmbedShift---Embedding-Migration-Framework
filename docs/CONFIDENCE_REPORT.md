# Reading the Confidence Report

The confidence report is the project's credibility: it tells you, **before you
commit**, how much search quality a migration will keep. Example:

```
recall@10  ceiling (true-new) : 1.000
recall@10  mapped (ours)      : 0.927
recall@10  do-nothing (old)   : 0.345
QUALITY RETAINED              : 92.8%   ->  PASS ✅
```

## What each number means

- **ceiling (true-new)** — what a *full re-embedding* would retrieve. This is the
  gold standard, so it's `1.000` by construction. It's the best you could do.
- **mapped (ours)** — what the migrated (mapped) vectors retrieve, compared to the
  ceiling. This is the number that matters.
- **do-nothing (old)** — what you'd get if you kept the old model's vectors. The
  migration must beat this to be worth doing.
- **QUALITY RETAINED** = `mapped / ceiling`. "We keep ~93% of full re-embedding
  quality for ~3% of the cost."

## How retrieval is measured

- For each held-out query, the **gold set** is the top-k neighbors a full
  re-embedding would return.
- **Queries are embedded with the NEW model** (never mapped) — exactly how
  production works. Corpus vectors are the mapped ones.
- recall@k = overlap between the mapped index's top-k and the gold top-k.

## The gate

A migration **passes** only if both hold:

1. `quality_retained >= confidence_threshold` (default 0.90), and
2. `mapped recall > do-nothing recall` (it genuinely beats keeping the old model).

If it fails, the transform is **skipped** (no half-migrated index) — override with
`force` only if you understand the quality cost.

## Why recall, not cosine

The report also shows cosine similarity (`mapped vs true`) as a secondary signal,
but the gate **never** uses it. A mapper can score high on cosine yet still reorder
neighbors and hurt search — only recall@k reflects real retrieval quality.

## When it fails

- **MLP not tried / `mapper_kind=linear`** → try `auto` so a small MLP can be
  trialed when linear underperforms.
- **Still failing with `auto`** → the models may be too dissimilar: the old vectors
  simply don't contain information the new model needs. No mapper can invent it
  back; a full re-embedding is the only path to full quality.
- **Borderline** → increase the sample size (more training pairs) or revisit the
  threshold for your use case.
