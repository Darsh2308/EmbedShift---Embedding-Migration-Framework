"""Embedder registry — how the NEW model is plugged into the service.

An embedder is a callable ``texts -> (n, d) array``. Over HTTP the client can't
pass a Python function, so the server keeps a registry of named embedders and the
client selects one by name. Register your real model at startup:

    from app.embedders import register_embedder
    register_embedder("my-model", lambda texts: model.encode(list(texts)))

A dependency-free deterministic ``hashing`` embedder is registered by default so
the API is usable out of the box (note: it only correlates with stored vectors
if those were themselves text-derived — real migrations register a real model).
"""

from __future__ import annotations

import hashlib
from typing import Callable, Sequence

import numpy as np

Embedder = Callable[[Sequence[str]], np.ndarray]

_REGISTRY: dict[str, Embedder] = {}


def register_embedder(name: str, fn: Embedder) -> None:
    _REGISTRY[name] = fn


def get_embedder(name: str) -> Embedder:
    if name not in _REGISTRY:
        raise KeyError(f"unknown embedder '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_embedders() -> list[str]:
    return sorted(_REGISTRY)


def make_hashing_embedder(dim: int = 256, seed: int = 0) -> Embedder:
    """A deterministic bag-of-words hashing embedder (no dependencies)."""

    def _bucket(token: str) -> int:
        digest = hashlib.md5(f"{seed}:{token}".encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little") % dim

    def embed(texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in str(text).split():
                out[i, _bucket(token)] += 1.0
        return out

    return embed


# Default built-in.
register_embedder("hashing", make_hashing_embedder(dim=256))
