"""Numeric utilities and platform workarounds."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np


@contextmanager
def safe_matmul():
    """Suppress *spurious* floating-point warnings emitted by ``matmul``.

    NumPy 2.x built against Apple's Accelerate BLAS (the default on macOS) raises
    false ``RuntimeWarning: divide by zero / overflow / invalid value encountered
    in matmul`` even when the result is perfectly finite and correct — its SIMD
    kernels touch padding lanes that set the FP error flags without affecting the
    output. Suppressing here keeps the math core's logs clean.

    This does NOT hide genuine numerical failures: callers still guard real
    non-finite results explicitly (see ``ensure_finite``).
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        yield


def ensure_finite(arr: np.ndarray, what: str = "result") -> np.ndarray:
    """Raise a clear error if ``arr`` contains NaN or Inf."""
    if not np.all(np.isfinite(arr)):
        raise FloatingPointError(
            f"{what} contains non-finite values (NaN/Inf). "
            "If lambda is 0 with a rank-deficient sample, try a larger lambda."
        )
    return arr
