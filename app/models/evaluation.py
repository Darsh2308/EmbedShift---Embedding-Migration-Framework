"""Schemas for the evaluation report and confidence gate.

These are pydantic models so they serialize to JSON for free and slot directly
into the FastAPI ``/evaluate`` endpoint in Phase 5.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class MapperInfo(BaseModel):
    kind: str
    d_old: int
    d_new: int
    lambda_: float
    normalize_output: bool


class EvaluationResult(BaseModel):
    """The measured numbers — recall@k is the credibility metric, cosine is secondary."""

    k: int
    n_queries: int
    n_corpus: int

    recall_at_k_max: float = Field(..., description="Ceiling: true-new vectors (gold standard).")
    recall_at_k_mapped: float = Field(..., description="Our method: mapped vectors.")
    recall_at_k_old: float = Field(..., description="Do-nothing: keep the old model.")
    quality_retained: float = Field(..., description="recall_mapped / recall_max.")

    mean_cosine_mapped_vs_true: float = Field(
        ..., description="Secondary sanity signal only — never the gate."
    )
    mapper: MapperInfo


class GateVerdict(BaseModel):
    """The go/no-go decision, based on recall (never cosine)."""

    passed: bool
    threshold: float
    quality_retained: float
    beats_do_nothing: bool
    reasons: list[str]
    recommendation: str


class ConfidenceReport(BaseModel):
    """Evaluation + verdict — what we hand the company before they commit."""

    evaluation: EvaluationResult
    verdict: GateVerdict

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        if path.suffix != ".json":
            path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ConfidenceReport":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_text(self) -> str:
        e, v = self.evaluation, self.verdict
        status = "PASS ✅" if v.passed else "FAIL ❌"
        lines = [
            "=" * 60,
            "  EMBEDDING MIGRATION — CONFIDENCE REPORT",
            "=" * 60,
            f"  Mapper        : {e.mapper.kind}  "
            f"(d_old={e.mapper.d_old} -> d_new={e.mapper.d_new}, "
            f"lambda={e.mapper.lambda_:g}, normalize={e.mapper.normalize_output})",
            f"  Eval set      : {e.n_queries} queries over {e.n_corpus} corpus vectors, k={e.k}",
            "-" * 60,
            f"  recall@{e.k}  ceiling (true-new) : {e.recall_at_k_max:.3f}",
            f"  recall@{e.k}  mapped (ours)      : {e.recall_at_k_mapped:.3f}",
            f"  recall@{e.k}  do-nothing (old)   : {e.recall_at_k_old:.3f}",
            "-" * 60,
            f"  QUALITY RETAINED               : {e.quality_retained:.1%}",
            f"  (cosine mapped vs true, FYI)   : {e.mean_cosine_mapped_vs_true:.3f}",
            "-" * 60,
            f"  GATE THRESHOLD                 : {v.threshold:.1%}",
            f"  VERDICT                        : {status}",
        ]
        for r in v.reasons:
            lines.append(f"    - {r}")
        lines.append("-" * 60)
        lines.append(f"  {v.recommendation}")
        lines.append("=" * 60)
        return "\n".join(lines)
