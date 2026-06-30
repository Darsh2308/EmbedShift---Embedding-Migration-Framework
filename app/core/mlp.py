r"""MLP mapper — a small neural net for when a linear map isn't enough.

When old and new models are *similar*, the linear ridge map (Phase 2) is best and
shouldn't be beaten. But when the relationship between the two embedding spaces is
nonlinear, a small multi-layer perceptron can capture distinctions the linear map
can't. Per the design doc, we don't reach for this first — it overfits easily on a
tiny sample — so the pipeline only upgrades to it when linear fails the gate.

Implemented in pure NumPy (no torch/tensorflow):
  - 1-2 hidden layers, ReLU (or tanh), linear output.
  - Inputs/outputs are mean-centered and globally scaled for stable training.
  - Adam optimizer + L2 weight decay + early stopping on an internal split — the
    three guards against overfitting on small samples.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app import __version__
from app.core.mapper import BaseMapper
from app.core.metrics import l2_normalize
from app.utils.numerics import ensure_finite, safe_matmul


class MLPMapper(BaseMapper):
    kind = "mlp"

    def __init__(
        self,
        hidden: int = 256,
        n_layers: int = 1,
        activation: str = "relu",
        lr: float = 1e-3,
        epochs: int = 300,
        batch_size: int = 128,
        weight_decay: float = 1e-4,
        patience: int = 20,
        val_fraction: float = 0.15,
        normalize_output: bool = False,
        seed: int = 0,
    ) -> None:
        if activation not in ("relu", "tanh"):
            raise ValueError("activation must be 'relu' or 'tanh'")
        self.hidden = int(hidden)
        self.n_layers = int(n_layers)
        self.activation = activation
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.val_fraction = float(val_fraction)
        self.normalize_output = bool(normalize_output)
        self.seed = int(seed)

        self.params: list[list[np.ndarray]] | None = None
        self.mu_x: np.ndarray | None = None
        self.mu_y: np.ndarray | None = None
        self.sigma_x: float = 1.0
        self.sigma_y: float = 1.0
        self._fitted = False

    # ------------------------------------------------------------------ #
    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def d_old(self) -> int:
        self._require_fitted()
        return int(self.params[0][0].shape[0])

    @property
    def d_new(self) -> int:
        self._require_fitted()
        return int(self.params[-1][0].shape[1])

    # ------------------------------------------------------------------ #
    # Activations
    # ------------------------------------------------------------------ #
    def _act(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(z, 0.0) if self.activation == "relu" else np.tanh(z)

    def _act_deriv(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return (z > 0).astype(z.dtype)
        return 1.0 - np.tanh(z) ** 2

    def _init_params(self, d_old: int, d_new: int, rng) -> list[list[np.ndarray]]:
        dims = [d_old] + [self.hidden] * self.n_layers + [d_new]
        params = []
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            scale = np.sqrt(2.0 / fan_in)  # He init (works for relu and is fine for tanh)
            W = (rng.standard_normal((dims[i], dims[i + 1])) * scale).astype(np.float32)
            b = np.zeros(dims[i + 1], dtype=np.float32)
            params.append([W, b])
        return params

    # ------------------------------------------------------------------ #
    # Forward / backward
    # ------------------------------------------------------------------ #
    def _forward(self, X: np.ndarray, params):
        acts = [X]
        pre = []
        a = X
        last = len(params) - 1
        with safe_matmul():
            for i, (W, b) in enumerate(params):
                z = a @ W + b
                pre.append(z)
                a = z if i == last else self._act(z)
                acts.append(a)
        return a, acts, pre

    def _backward(self, pred, Y, acts, pre, params):
        n = pred.shape[0]
        grads = [None] * len(params)
        delta = (pred - Y) * (2.0 / n)
        with safe_matmul():
            for i in reversed(range(len(params))):
                W, _ = params[i]
                gW = acts[i].T @ delta + self.weight_decay * W
                gb = delta.sum(axis=0)
                grads[i] = (gW.astype(np.float32), gb.astype(np.float32))
                if i > 0:
                    delta = (delta @ W.T) * self._act_deriv(pre[i - 1])
        return grads

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, Y: np.ndarray) -> "MLPMapper":
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be 2D (n, d)")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows")
        if X.shape[0] < 2:
            raise ValueError("MLP needs at least 2 training pairs")

        rng = np.random.default_rng(self.seed)
        mu_x = X.mean(axis=0)
        mu_y = Y.mean(axis=0)
        sigma_x = float(X.std()) or 1.0
        sigma_y = float(Y.std()) or 1.0
        Xc = (X - mu_x) / sigma_x
        Yc = (Y - mu_y) / sigma_y

        # Internal split for early stopping.
        n = X.shape[0]
        idx = rng.permutation(n)
        n_val = min(max(1, int(self.val_fraction * n)), n - 1)
        v_idx, t_idx = idx[:n_val], idx[n_val:]
        Xt, Yt, Xv, Yv = Xc[t_idx], Yc[t_idx], Xc[v_idx], Yc[v_idx]

        params = self._init_params(X.shape[1], Y.shape[1], rng)
        m_state = [[np.zeros_like(W), np.zeros_like(b)] for W, b in params]
        v_state = [[np.zeros_like(W), np.zeros_like(b)] for W, b in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0

        best_loss = np.inf
        best_params = None
        stale = 0
        bs = min(self.batch_size, len(Xt)) or len(Xt)

        for _epoch in range(self.epochs):
            perm = rng.permutation(len(Xt))
            for s in range(0, len(Xt), bs):
                bi = perm[s:s + bs]
                pred, acts, pre = self._forward(Xt[bi], params)
                grads = self._backward(pred, Yt[bi], acts, pre, params)
                step += 1
                for i, (gW, gb) in enumerate(grads):
                    for j, g in enumerate((gW, gb)):
                        m_state[i][j] = beta1 * m_state[i][j] + (1 - beta1) * g
                        v_state[i][j] = beta2 * v_state[i][j] + (1 - beta2) * (g * g)
                        m_hat = m_state[i][j] / (1 - beta1 ** step)
                        v_hat = v_state[i][j] / (1 - beta2 ** step)
                        params[i][j] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

            val_pred, _, _ = self._forward(Xv, params)
            val_loss = float(np.mean((val_pred - Yv) ** 2))
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_params = [[W.copy(), b.copy()] for W, b in params]
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        self.params = best_params if best_params is not None else params
        for W, b in self.params:
            ensure_finite(W, "MLP weights")
        self.mu_x = mu_x.astype(np.float32)
        self.mu_y = mu_y.astype(np.float32)
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def transform(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X = np.asarray(X, dtype=np.float32)
        single = X.ndim == 1
        if single:
            X = X[None, :]
        if X.shape[1] != self.d_old:
            raise ValueError(f"input dim {X.shape[1]} != mapper d_old {self.d_old}")

        a = (X - self.mu_x) / self.sigma_x
        out, _, _ = self._forward(a, self.params)
        out = out * self.sigma_y + self.mu_y
        if self.normalize_output:
            out = l2_normalize(out)
        out = out.astype(np.float32)
        return out[0] if single else out

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> Path:
        self._require_fitted()
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "kind": np.array(self.kind),
            "n_weight": np.array(len(self.params)),
            "hidden": np.array(self.hidden),
            "n_layers": np.array(self.n_layers),
            "activation": np.array(self.activation),
            "normalize_output": np.array(int(self.normalize_output)),
            "mu_x": self.mu_x,
            "mu_y": self.mu_y,
            "sigma_x": np.array(self.sigma_x, dtype=np.float64),
            "sigma_y": np.array(self.sigma_y, dtype=np.float64),
            "version": np.array(__version__),
        }
        for i, (W, b) in enumerate(self.params):
            data[f"W{i}"] = W
            data[f"b{i}"] = b
        np.savez(path, **data)
        return path

    @classmethod
    def _from_npz(cls, data) -> "MLPMapper":
        m = cls(
            hidden=int(data["hidden"]),
            n_layers=int(data["n_layers"]),
            activation=str(data["activation"]),
            normalize_output=bool(int(data["normalize_output"])),
        )
        L = int(data["n_weight"])
        m.params = [
            [np.asarray(data[f"W{i}"], dtype=np.float32), np.asarray(data[f"b{i}"], dtype=np.float32)]
            for i in range(L)
        ]
        m.mu_x = np.asarray(data["mu_x"], dtype=np.float32)
        m.mu_y = np.asarray(data["mu_y"], dtype=np.float32)
        m.sigma_x = float(data["sigma_x"])
        m.sigma_y = float(data["sigma_y"])
        m._fitted = True
        return m

    @classmethod
    def load(cls, path: str | Path) -> "MLPMapper":
        from app.core.mapper import load_mapper

        mapper = load_mapper(path)
        if not isinstance(mapper, cls):
            raise TypeError(f"artifact is not a {cls.__name__}")
        return mapper

    # ------------------------------------------------------------------ #
    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("mapper is not fitted; call fit() first")
