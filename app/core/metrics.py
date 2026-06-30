"""Numeric helpers used by the mapper and (later) the evaluation gate.

Phase 2 only needs MSE / cosine for choosing the ridge strength (lambda).
The *real* quality gate — retrieval recall@k — is added in Phase 3; cosine and
MSE are fine for hyperparameter selection but must never stand in for recall as
the final success metric.
"""

from __future__ import annotations

import numpy as np


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale each row to unit L2 norm. Zero rows are left at (near) zero."""
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / np.maximum(norms, eps)


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean squared error between predicted and true vectors."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {true.shape}")
    return float(np.mean((pred - true) ** 2))


def mean_cosine_similarity(pred: np.ndarray, true: np.ndarray, eps: float = 1e-12) -> float:
    """Average row-wise cosine similarity between predicted and true vectors."""
    pred = np.atleast_2d(np.asarray(pred, dtype=np.float64))
    true = np.atleast_2d(np.asarray(true, dtype=np.float64))
    if pred.shape != true.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {true.shape}")
    p = l2_normalize(pred, eps)
    t = l2_normalize(true, eps)
    return float(np.mean(np.sum(p * t, axis=-1)))
