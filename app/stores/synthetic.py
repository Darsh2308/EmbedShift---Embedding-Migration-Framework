"""Generate synthetic vectors for tests, demos, and benchmarking.

Useful before a real corpus exists: produce old vectors (and optionally a known
linear relationship to "new" vectors) to exercise the pipeline end to end.
"""

from __future__ import annotations

import numpy as np

from app.stores.base import VECTOR_DTYPE
from app.utils.numerics import safe_matmul


def make_old_vectors(
    n: int,
    dim: int,
    seed: int = 0,
    with_text: bool = False,
) -> tuple[list[str], np.ndarray, list[str] | None]:
    """Return ``(ids, vectors, texts|None)`` of random old vectors."""
    rng = np.random.default_rng(seed)
    ids = [f"doc-{i}" for i in range(n)]
    vectors = rng.standard_normal((n, dim)).astype(VECTOR_DTYPE)
    texts = [f"synthetic document number {i}" for i in range(n)] if with_text else None
    return ids, vectors, texts


def make_linear_pair(
    n: int,
    d_old: int,
    d_new: int,
    seed: int = 0,
    noise: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Old vectors, a true mapping matrix W, and the resulting new vectors.

    ``new = old @ W + noise`` — a ground-truth linear relationship the mapper
    (Phase 2) should be able to recover.
    """
    rng = np.random.default_rng(seed)
    old = rng.standard_normal((n, d_old)).astype(VECTOR_DTYPE)
    W = rng.standard_normal((d_old, d_new)).astype(VECTOR_DTYPE)
    with safe_matmul():
        new = old @ W
    if noise > 0:
        new = new + noise * rng.standard_normal((n, d_new)).astype(VECTOR_DTYPE)
    return old, W, new.astype(VECTOR_DTYPE)


def make_related_spaces(
    n: int,
    d_old: int,
    d_new: int,
    d_latent: int | None = None,
    noise: float = 0.05,
    nonlinearity: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Two embedding spaces that are different *views* of the same latent content.

    Each "model" projects a shared latent representation through its own random
    matrix (plus optional nonlinearity + noise). This mimics two real embedding
    models: their neighborhoods are related but not identical, so a do-nothing
    baseline genuinely underperforms and a mapper has real work to do. With
    ``nonlinearity > 0`` the relationship is no longer perfectly linear, so a
    linear mapper recovers *most* but not all quality — exactly the real tradeoff.

    Returns ``(old, new)`` vectors of shape ``(n, d_old)`` and ``(n, d_new)``.
    """
    rng = np.random.default_rng(seed)
    d_latent = d_latent or max(2, min(d_old, d_new) // 2)
    latent = rng.standard_normal((n, d_latent)).astype(VECTOR_DTYPE)
    A_old = rng.standard_normal((d_latent, d_old)).astype(VECTOR_DTYPE)
    A_new = rng.standard_normal((d_latent, d_new)).astype(VECTOR_DTYPE)
    with safe_matmul():
        old = latent @ A_old
        new = latent @ A_new
        if nonlinearity > 0:
            new = new + nonlinearity * ((latent ** 2) @ A_new)
    if noise > 0:
        old = old + noise * rng.standard_normal(old.shape).astype(VECTOR_DTYPE)
        new = new + noise * rng.standard_normal(new.shape).astype(VECTOR_DTYPE)
    return old.astype(VECTOR_DTYPE), new.astype(VECTOR_DTYPE)
