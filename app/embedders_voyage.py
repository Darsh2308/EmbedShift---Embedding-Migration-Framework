"""Voyage AI embedder — the NEW model for a live migration.

An embedder is `texts -> (n, d) float32`. The migration pipeline only ever embeds
*documents* (it re-embeds sampled corpus chunks to build training pairs), so the
embedder used by `run_migration` must use ``input_type="document"``. Queries are
embedded separately (validation harness) with ``input_type="query"`` — Voyage uses
different internal prompts for the two, and getting this right matters for recall.

``voyageai`` is imported lazily so the framework runs without it.

CLI usage:
    emf migrate ... --embedder-module app.embedders_voyage:voyage_document
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from app.embedders import Embedder
from app.utils.logging import get_logger

logger = get_logger(__name__)


def make_voyage_embedder(
    model: str = "voyage-3-large",
    input_type: str = "document",  # "document" for corpus, "query" for queries
    output_dimension: int = 2048,  # voyage-3-large supports 256/512/1024/2048
    batch_size: int = 128,         # Voyage hard cap per request
    api_key: str | None = None,    # falls back to VOYAGE_API_KEY env var
    min_request_interval: float = 0.0,  # seconds between requests (free tier: ~21 for 3 RPM)
    max_retries: int = 6,          # retry on rate-limit errors with backoff
) -> Embedder:
    """Return a `texts -> (n, output_dimension) float32` callable backed by Voyage.

    ``min_request_interval`` paces requests (set it for the free tier's 3 RPM cap);
    rate-limit errors are retried with exponential backoff so a constrained account
    completes (slowly) instead of crashing.
    """

    def embed(texts: Sequence[str]) -> np.ndarray:
        import voyageai
        from voyageai.error import RateLimitError

        client = voyageai.Client(api_key=api_key)
        items = [str(t) for t in texts]
        out: list[list[float]] = []
        last_call = 0.0
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            wait = 5.0
            for attempt in range(max_retries + 1):
                if min_request_interval:
                    delta = min_request_interval - (time.monotonic() - last_call)
                    if delta > 0:
                        time.sleep(delta)
                try:
                    resp = client.embed(
                        chunk, model=model, input_type=input_type,
                        output_dimension=output_dimension,
                    )
                    last_call = time.monotonic()
                    out.extend(resp.embeddings)
                    break
                except RateLimitError:
                    if attempt == max_retries:
                        raise
                    logger.warning("Voyage rate-limited; backing off %.0fs", wait)
                    time.sleep(wait)
                    wait = min(wait * 2, 60.0)
                    last_call = time.monotonic()
        if not out:
            return np.empty((0, output_dimension), dtype=np.float32)
        return np.asarray(out, dtype=np.float32)

    return embed


# Ready-made callables for the --embedder-module CLI path (2048-dim for this test).
voyage_document = make_voyage_embedder(input_type="document", output_dimension=2048)
voyage_query = make_voyage_embedder(input_type="query", output_dimension=2048)

# Throttled variants for Voyage's FREE tier (3 RPM / 10K TPM): small batches + spacing.
voyage_document_free = make_voyage_embedder(
    input_type="document", output_dimension=2048, batch_size=8, min_request_interval=22.0
)
voyage_query_free = make_voyage_embedder(
    input_type="query", output_dimension=2048, batch_size=8, min_request_interval=22.0
)
