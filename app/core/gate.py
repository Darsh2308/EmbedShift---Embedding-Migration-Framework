"""The confidence gate — turns measured recall into a go/no-go verdict.

This is what makes the framework trustworthy and sellable: we quote expected
quality *before* anyone commits. The gate decides on **recall**, never cosine.

A migration passes only if BOTH hold:
  1. quality_retained >= threshold   (we keep enough of the new model's quality)
  2. recall_mapped > recall_old      (we genuinely beat doing nothing)
"""

from __future__ import annotations

import numpy as np

from app.core.evaluation import evaluate_mapper
from app.core.mapper import BaseMapper
from app.models.evaluation import ConfidenceReport, EvaluationResult, GateVerdict


def run_gate(result: EvaluationResult, threshold: float) -> GateVerdict:
    meets_threshold = result.quality_retained >= threshold
    beats_do_nothing = result.recall_at_k_mapped > result.recall_at_k_old
    passed = bool(meets_threshold and beats_do_nothing)

    reasons = [
        f"quality retained {result.quality_retained:.1%} "
        f"{'>=' if meets_threshold else '<'} threshold {threshold:.1%}",
        f"mapped recall {result.recall_at_k_mapped:.3f} "
        f"{'>' if beats_do_nothing else '<='} do-nothing {result.recall_at_k_old:.3f}",
    ]

    if passed:
        recommendation = (
            f"PROCEED: the mapper retains ~{result.quality_retained:.0%} of full "
            "re-embedding quality. Safe to transform the full corpus and cut over."
        )
    elif not beats_do_nothing:
        recommendation = (
            "STOP: the mapped index does not beat keeping the old model. The two "
            "models are likely too dissimilar for mapping — consider full re-embedding."
        )
    else:
        recommendation = (
            f"STOP: quality retained ({result.quality_retained:.0%}) is below the "
            f"{threshold:.0%} threshold. Try a higher-capacity mapper (MLP) or a "
            "larger sample; if it still fails, the models may be too dissimilar."
        )

    return GateVerdict(
        passed=passed,
        threshold=round(float(threshold), 6),
        quality_retained=result.quality_retained,
        beats_do_nothing=beats_do_nothing,
        reasons=reasons,
        recommendation=recommendation,
    )


def build_report(result: EvaluationResult, threshold: float) -> ConfidenceReport:
    return ConfidenceReport(evaluation=result, verdict=run_gate(result, threshold))


def evaluate_and_gate(
    old_corpus: np.ndarray,
    new_corpus: np.ndarray,
    mapper: BaseMapper,
    threshold: float,
    k: int = 10,
    max_queries: int | None = 1000,
    seed: int = 0,
) -> ConfidenceReport:
    """Convenience: evaluate the mapper and apply the gate in one call."""
    result = evaluate_mapper(
        old_corpus, new_corpus, mapper, k=k, max_queries=max_queries, seed=seed
    )
    return build_report(result, threshold)
