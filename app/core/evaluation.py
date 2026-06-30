r"""Evaluation — measure how much retrieval quality the mapper retains.

The success metric (from the design doc): **retrieval recall@k of mapped vectors
vs. true-new vectors, on a held-out query set.** Three systems are compared
against the same gold standard:

    - ceiling    : true-new corpus vectors           -> recall@k_max  (= 1.0 here)
    - our method : mapped corpus vectors             -> recall@k_mapped
    - do-nothing : old corpus vectors (old model)    -> recall@k_old

    quality_retained = recall@k_mapped / recall@k_max

Gold standard (self-contained, no external labels needed): the set of top-k
neighbors that a **full re-embedding** (true-new corpus) would return for each
query. This is exactly the recall-of-approximate-search-vs-exact-search method
used in ANN benchmarking. recall@k_max is then 1.0 by construction.

Critical query-time rule (easy to get wrong, see the doc): queries for the
true-new and mapped indexes are embedded with the **new model** — we never map
the query. The do-nothing index uses the old model for both query and corpus.

Cosine similarity is reported too, but only as a secondary sanity signal — a
mapper can look great on cosine and still hurt search, so the gate uses recall.
"""

from __future__ import annotations

import numpy as np

from app.core.mapper import BaseMapper
from app.core.metrics import l2_normalize, mean_cosine_similarity
from app.models.evaluation import EvaluationResult, MapperInfo
from app.utils.numerics import safe_matmul


def _top_k_indices(sims: np.ndarray, k: int) -> np.ndarray:
    """Indices of the top-k columns per row, sorted by descending similarity."""
    nc = sims.shape[1]
    k = min(k, nc)
    # argpartition gets the top-k cheaply; then sort just those k by actual score.
    part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(sims.shape[0])[:, None]
    order = np.argsort(-sims[rows, part], axis=1)
    return part[rows, order]


def retrieve(
    queries: np.ndarray,
    corpus: np.ndarray,
    k: int,
    exclude_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Top-k corpus indices for each query, by cosine similarity.

    ``exclude_idx[i]`` (optional) is a corpus position to drop for query i — used
    for leave-one-out when a query also appears in the corpus.
    """
    qn = l2_normalize(queries)
    cn = l2_normalize(corpus)
    with safe_matmul():
        sims = qn @ cn.T  # (n_queries, n_corpus)
    if exclude_idx is not None:
        sims[np.arange(sims.shape[0]), exclude_idx] = -np.inf
    return _top_k_indices(sims, k)


def recall_at_k(retrieved: np.ndarray, gold: np.ndarray) -> float:
    """Mean over queries of |retrieved ∩ gold| / |gold|."""
    if retrieved.shape[0] != gold.shape[0]:
        raise ValueError("retrieved and gold must have the same number of queries")
    recalls = []
    for r, g in zip(retrieved, gold):
        gold_set = set(g.tolist())
        if not gold_set:
            continue
        hits = sum(1 for x in r.tolist() if x in gold_set)
        recalls.append(hits / len(gold_set))
    return float(np.mean(recalls)) if recalls else 0.0


def evaluate_mapper(
    old_corpus: np.ndarray,
    new_corpus: np.ndarray,
    mapper: BaseMapper,
    k: int = 10,
    max_queries: int | None = 1000,
    seed: int = 0,
) -> EvaluationResult:
    """Run the three-way recall comparison on a held-out corpus.

    ``old_corpus`` / ``new_corpus`` are the held-out evaluation slice for which we
    have *both* the stored old vectors and the freshly computed true-new vectors.
    Queries are drawn from this same set (leave-one-out), capped at ``max_queries``.
    """
    old_corpus = np.asarray(old_corpus, dtype=np.float32)
    new_corpus = np.asarray(new_corpus, dtype=np.float32)
    n = old_corpus.shape[0]
    if old_corpus.shape[0] != new_corpus.shape[0]:
        raise ValueError("old_corpus and new_corpus must have the same number of rows")
    if k < 1:
        raise ValueError("k must be >= 1")
    if n <= k:
        raise ValueError(f"need more than k+1 corpus vectors for leave-one-out; got n={n}, k={k}")

    mapped_corpus = mapper.transform(old_corpus)
    if mapped_corpus.shape[1] != new_corpus.shape[1]:
        raise ValueError(
            f"mapper output dim {mapped_corpus.shape[1]} != true-new dim {new_corpus.shape[1]}"
        )

    # Choose queries (a subset of the corpus, leave-one-out).
    if max_queries is not None and max_queries < n:
        rng = np.random.default_rng(seed)
        q_idx = np.sort(rng.choice(n, size=max_queries, replace=False))
    else:
        q_idx = np.arange(n)

    q_new = new_corpus[q_idx]  # queries embedded with the NEW model
    q_old = old_corpus[q_idx]  # queries embedded with the OLD model (do-nothing)

    # Gold standard = what full re-embedding (true-new) would retrieve.
    gold = retrieve(q_new, new_corpus, k, exclude_idx=q_idx)
    retrieved_mapped = retrieve(q_new, mapped_corpus, k, exclude_idx=q_idx)
    retrieved_old = retrieve(q_old, old_corpus, k, exclude_idx=q_idx)

    recall_max = recall_at_k(gold, gold)  # 1.0 by construction; computed for honesty
    recall_mapped = recall_at_k(retrieved_mapped, gold)
    recall_old = recall_at_k(retrieved_old, gold)
    quality = recall_mapped / recall_max if recall_max > 0 else 0.0

    cosine = mean_cosine_similarity(mapped_corpus, new_corpus)

    return EvaluationResult(
        k=k,
        n_queries=int(len(q_idx)),
        n_corpus=int(n),
        recall_at_k_max=round(recall_max, 6),
        recall_at_k_mapped=round(recall_mapped, 6),
        recall_at_k_old=round(recall_old, 6),
        quality_retained=round(quality, 6),
        mean_cosine_mapped_vs_true=round(cosine, 6),
        mapper=_mapper_info(mapper),
    )


def _mapper_info(mapper: BaseMapper) -> MapperInfo:
    return MapperInfo(
        kind=getattr(mapper, "kind", "unknown"),
        d_old=int(getattr(mapper, "d_old", 0)),
        d_new=int(getattr(mapper, "d_new", 0)),
        lambda_=float(getattr(mapper, "lambda_", 0.0) or 0.0),
        normalize_output=bool(getattr(mapper, "normalize_output", False)),
    )
