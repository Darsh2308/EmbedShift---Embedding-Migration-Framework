"""Tests for the Voyage embedder using a stubbed `voyageai` module (no API calls)."""

import sys
import types

import numpy as np

from app.embedders_voyage import make_voyage_embedder


def _install_fake_voyageai(monkeypatch, dim=2048):
    """Inject a fake `voyageai` module that records calls and returns dim-sized vecs."""
    calls = {"batches": [], "input_types": set(), "models": set(), "dims": set()}

    class _Resp:
        def __init__(self, n):
            self.embeddings = [[0.01 * (i + 1)] * dim for i in range(n)]

    class _Client:
        def __init__(self, api_key=None):
            calls["api_key"] = api_key

        def embed(self, texts, model=None, input_type=None, output_dimension=None):
            calls["batches"].append(len(texts))
            calls["models"].add(model)
            calls["input_types"].add(input_type)
            calls["dims"].add(output_dimension)
            return _Resp(len(texts))

    fake = types.ModuleType("voyageai")
    fake.Client = _Client
    fake_error = types.ModuleType("voyageai.error")
    fake_error.RateLimitError = type("RateLimitError", (Exception,), {})
    fake.error = fake_error
    monkeypatch.setitem(sys.modules, "voyageai", fake)
    monkeypatch.setitem(sys.modules, "voyageai.error", fake_error)
    return calls


def test_voyage_embed_shape_and_dtype(monkeypatch):
    calls = _install_fake_voyageai(monkeypatch, dim=2048)
    embed = make_voyage_embedder(output_dimension=2048)
    out = embed(["a", "b", "c"])
    assert out.shape == (3, 2048)
    assert out.dtype == np.float32
    assert calls["models"] == {"voyage-3-large"}
    assert calls["input_types"] == {"document"}
    assert calls["dims"] == {2048}


def test_voyage_batches_over_128(monkeypatch):
    calls = _install_fake_voyageai(monkeypatch)
    embed = make_voyage_embedder(batch_size=128, output_dimension=2048)
    out = embed([f"t{i}" for i in range(300)])
    assert out.shape == (300, 2048)
    assert calls["batches"] == [128, 128, 44]  # respects the 128 cap


def test_voyage_query_input_type(monkeypatch):
    calls = _install_fake_voyageai(monkeypatch)
    embed = make_voyage_embedder(input_type="query", output_dimension=2048)
    embed(["what is bail?"])
    assert calls["input_types"] == {"query"}


def test_voyage_empty_input(monkeypatch):
    _install_fake_voyageai(monkeypatch)
    embed = make_voyage_embedder(output_dimension=2048)
    out = embed([])
    assert out.shape == (0, 2048)
