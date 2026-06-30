r"""The mapper — learns f(old_vector) ≈ new_vector from a small sample.

Math (linear map via ridge regression, with mean-centering):

    Given matched pairs  X (old, n×d_old)  and  Y (new, n×d_new):

      μ_x = mean(X),  μ_y = mean(Y)            # store both
      Xc  = X − μ_x,  Yc  = Y − μ_y            # center (handles the intercept)
      W   = (Xcᵀ Xc + λI)⁻¹ Xcᵀ Yc            # ridge solution, shape d_old×d_new

      f(x) = (x − μ_x) · W + μ_y               # then optionally L2-normalize

Why mean-center?  Centering both sides lets the linear map fit the *shape* of the
relationship while μ_y carries the constant offset between the two spaces — so we
never have to penalize or fit a separate bias term.

Why solve via SVD instead of the closed-form inverse?  Forming Xcᵀ Xc squares the
condition number and can lose precision. Using the economy SVD  Xc = U S Vᵀ  gives

      W = V · diag( s / (s² + λ) ) · (Uᵀ Yc)

which is mathematically identical, more stable, and lets us reuse one SVD to try
many λ values cheaply during cross-validation.

Three quality details (from the design doc):
  1. Mean-center old and new before fitting (done here, means stored).
  2. L2-normalize the output if the new model uses cosine similarity
     (set normalize_output=True).
  3. If a linear map isn't accurate enough, upgrade to a small MLP — deferred to
     Phase 7; the BaseMapper interface below is what an MLP mapper will implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from app import __version__
from app.core.metrics import l2_normalize, mean_cosine_similarity, mse
from app.utils.numerics import ensure_finite, safe_matmul

#: Default ridge strengths tried by ``fit_cv`` (log-spaced).
DEFAULT_LAMBDAS: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


class BaseMapper(ABC):
    """Common interface for all mappers (linear now; MLP in Phase 7)."""

    @abstractmethod
    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> "BaseMapper": ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def save(self, path: str | Path) -> Path: ...

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)


class LinearMapper(BaseMapper):
    """Ridge-regression linear map between two embedding spaces."""

    kind = "linear"

    def __init__(self, normalize_output: bool = False) -> None:
        self.normalize_output = bool(normalize_output)
        self.W: np.ndarray | None = None
        self.mu_x: np.ndarray | None = None
        self.mu_y: np.ndarray | None = None
        self.lambda_: float | None = None
        #: Populated by fit_cv: {lambda: mean validation score}.
        self.cv_results_: dict[float, float] | None = None
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def d_old(self) -> int:
        self._require_fitted()
        return int(self.W.shape[0])

    @property
    def d_new(self) -> int:
        self._require_fitted()
        return int(self.W.shape[1])

    # ------------------------------------------------------------------ #
    # Core solver
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ridge_svd(Xc: np.ndarray, Yc: np.ndarray, lam: float) -> np.ndarray:
        """W = V diag(s/(s²+λ)) Uᵀ Yc, from the economy SVD of centered Xc."""
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)  # U(n,r) s(r,) Vt(r,d)
        filt = s / (s * s + lam)  # (r,)
        with safe_matmul():
            return Vt.T @ (filt[:, None] * (U.T @ Yc))  # (d_old, d_new)

    @staticmethod
    def _center(X: np.ndarray, Y: np.ndarray):
        mu_x = X.mean(axis=0)
        mu_y = Y.mean(axis=0)
        return mu_x, mu_y, X - mu_x, Y - mu_y

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, Y: np.ndarray, lambda_: float = 1.0) -> "LinearMapper":
        """Fit the map on matched pairs with a fixed ridge strength ``lambda_``."""
        X, Y = self._validate_pair(X, Y)
        if lambda_ < 0:
            raise ValueError("lambda_ must be >= 0")

        mu_x, mu_y, Xc, Yc = self._center(X, Y)
        W = self._ridge_svd(Xc, Yc, float(lambda_))
        ensure_finite(W, "mapper weights W")

        self.mu_x = mu_x.astype(np.float32)
        self.mu_y = mu_y.astype(np.float32)
        self.W = W.astype(np.float32)
        self.lambda_ = float(lambda_)
        self._fitted = True
        return self

    def fit_cv(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        lambdas: tuple[float, ...] | list[float] | None = None,
        folds: int = 5,
        seed: int = 0,
        metric: str = "mse",
    ) -> "LinearMapper":
        """Select ``lambda`` by k-fold cross-validation, then refit on all data.

        ``metric``: ``"mse"`` (lower is better) or ``"cosine"`` (higher is better).
        Centering is recomputed per training fold to avoid leakage. One SVD per
        fold is reused across all candidate lambdas.
        """
        X, Y = self._validate_pair(X, Y)
        lambdas = list(lambdas) if lambdas is not None else list(DEFAULT_LAMBDAS)
        if any(l < 0 for l in lambdas):
            raise ValueError("all lambdas must be >= 0")
        if metric not in ("mse", "cosine"):
            raise ValueError("metric must be 'mse' or 'cosine'")

        n = X.shape[0]
        folds = int(folds)
        if folds < 2:
            raise ValueError("folds must be >= 2")
        if folds > n:
            raise ValueError(f"folds ({folds}) cannot exceed number of samples ({n})")

        rng = np.random.default_rng(seed)
        fold_id = rng.permutation(n) % folds

        scores: dict[float, list[float]] = {l: [] for l in lambdas}
        for f in range(folds):
            val = fold_id == f
            train = ~val
            Xtr, Ytr = X[train], Y[train]
            Xval, Yval = X[val], Y[val]

            mu_x, mu_y, Xc, Yc = self._center(Xtr, Ytr)
            U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
            Xval_c = Xval - mu_x
            with safe_matmul():
                UtY = U.T @ Yc
                for lam in lambdas:
                    filt = s / (s * s + lam)
                    W = Vt.T @ (filt[:, None] * UtY)
                    pred = Xval_c @ W + mu_y
                    score = (
                        mse(pred, Yval)
                        if metric == "mse"
                        else mean_cosine_similarity(pred, Yval)
                    )
                    scores[lam].append(score)

        mean_scores = {lam: float(np.mean(v)) for lam, v in scores.items()}
        best = min(mean_scores, key=mean_scores.get) if metric == "mse" else max(
            mean_scores, key=mean_scores.get
        )

        self.fit(X, Y, lambda_=best)
        self.cv_results_ = mean_scores
        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map old vectors into the new space. Accepts a single vector or a batch."""
        self._require_fitted()
        X = np.asarray(X, dtype=np.float32)
        single = X.ndim == 1
        if single:
            X = X[None, :]
        if X.ndim != 2:
            raise ValueError(f"X must be 1D or 2D; got shape {X.shape}")
        if X.shape[1] != self.d_old:
            raise ValueError(
                f"input dim {X.shape[1]} does not match mapper's d_old {self.d_old}"
            )

        with safe_matmul():
            out = (X - self.mu_x) @ self.W + self.mu_y
        if self.normalize_output:
            out = l2_normalize(out)
        out = out.astype(np.float32)
        return out[0] if single else out

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> Path:
        """Save W + means + config to a ``.npz`` artifact."""
        self._require_fitted()
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            kind=np.array(self.kind),
            W=self.W,
            mu_x=self.mu_x,
            mu_y=self.mu_y,
            lambda_=np.array(float(self.lambda_), dtype=np.float64),
            normalize_output=np.array(int(self.normalize_output)),
            version=np.array(__version__),
        )
        return path

    @classmethod
    def _from_npz(cls, data) -> "LinearMapper":
        m = cls(normalize_output=bool(int(data["normalize_output"])))
        m.W = np.asarray(data["W"], dtype=np.float32)
        m.mu_x = np.asarray(data["mu_x"], dtype=np.float32)
        m.mu_y = np.asarray(data["mu_y"], dtype=np.float32)
        m.lambda_ = float(data["lambda_"])
        m._fitted = True
        return m

    @classmethod
    def load(cls, path: str | Path) -> "LinearMapper":
        mapper = load_mapper(path)
        if not isinstance(mapper, cls):
            raise TypeError(f"artifact is not a {cls.__name__}")
        return mapper

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("mapper is not fitted; call fit() or fit_cv() first")

    @staticmethod
    def _validate_pair(X: np.ndarray, Y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must both be 2D (n, d)")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y must have the same rows; got {X.shape[0]} vs {Y.shape[0]}")
        if X.shape[0] < 1:
            raise ValueError("need at least one training pair")
        return X, Y


def load_mapper(path: str | Path) -> BaseMapper:
    """Load any mapper artifact, dispatching on its stored ``kind``."""
    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(".npz")
    if not path.exists():
        raise FileNotFoundError(f"mapper artifact not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        kind = str(data["kind"])
        if kind == LinearMapper.kind:
            return LinearMapper._from_npz(data)
        if kind == "mlp":
            from app.core.mlp import MLPMapper  # lazy import to avoid a cycle

            return MLPMapper._from_npz(data)
        raise ValueError(f"unknown mapper kind '{kind}'")


#: Default mapper used by the pipeline.
Mapper = LinearMapper
