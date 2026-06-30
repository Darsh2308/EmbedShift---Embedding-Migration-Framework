"""Schemas for the migration pipeline: config, transform summary, and result."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.models.evaluation import ConfidenceReport


class MigrationConfig(BaseModel):
    """Everything that controls a migration run (serializable; FastAPI-ready)."""

    # Sampling (Step 1)
    sample_size: Optional[int] = Field(None, description="Absolute sample size; overrides fraction.")
    sample_fraction: float = Field(0.03, gt=0, le=1.0, description="Fraction of corpus to re-embed.")
    validation_fraction: float = Field(0.2, gt=0, lt=1.0, description="Sample slice held out for the gate.")
    seed: int = 0

    # Mapper (Step 2)
    mapper_kind: str = Field("auto", description="auto | linear | mlp")
    normalize_output: bool = True
    use_cv: bool = True
    lambda_: float = Field(1.0, ge=0)
    cv_folds: int = Field(5, ge=2)
    cv_metric: str = "cosine"

    # MLP fallback (used by 'auto' upgrade and 'mlp')
    mlp_hidden: int = Field(256, ge=1)
    mlp_layers: int = Field(1, ge=1, le=3)
    mlp_lr: float = Field(1e-3, gt=0)
    mlp_epochs: int = Field(300, ge=1)
    mlp_batch_size: int = Field(128, ge=1)
    mlp_weight_decay: float = Field(1e-4, ge=0)
    mlp_patience: int = Field(20, ge=1)

    # Evaluation + gate (Step 3)
    k: int = Field(10, ge=1)
    max_queries: Optional[int] = 1000
    confidence_threshold: float = Field(0.90, ge=0, le=1.0)

    # Transform (Step 4)
    output_collection: str = "corpus_v2"
    batch_size: int = Field(1000, ge=1)
    resume: bool = False
    force: bool = Field(False, description="Transform even if the confidence gate fails.")

    # Paths
    output_dir: str = "data"
    artifacts_dir: str = "artifacts"

    @field_validator("cv_metric")
    @classmethod
    def _check_metric(cls, v: str) -> str:
        if v not in ("mse", "cosine"):
            raise ValueError("cv_metric must be 'mse' or 'cosine'")
        return v

    @field_validator("mapper_kind")
    @classmethod
    def _check_kind(cls, v: str) -> str:
        if v not in ("auto", "linear", "mlp"):
            raise ValueError("mapper_kind must be 'auto', 'linear', or 'mlp'")
        return v


class TransformSummary(BaseModel):
    output_path: str
    n_written: int
    n_total: int
    batches: int
    resumed: bool = False
    done: bool = True


class MigrationResult(BaseModel):
    """The full outcome of a migration run."""

    report: ConfidenceReport
    transformed: bool
    skipped_reason: Optional[str] = None

    output_path: Optional[str] = None
    n_transformed: int = 0
    transform: Optional[TransformSummary] = None

    mapper_path: str
    report_path: str

    sample_size: int
    train_size: int
    val_size: int

    mapper_kind: str = "linear"
    mapper_attempts: Optional[list[dict]] = None

    def to_text(self) -> str:
        lines = [self.report.to_text(), ""]
        if self.transformed:
            lines.append(
                f"  TRANSFORM: wrote {self.n_transformed} mapped vectors -> {self.output_path}"
            )
        else:
            lines.append(f"  TRANSFORM: SKIPPED ({self.skipped_reason})")
        lines.append(f"  Mapper artifact : {self.mapper_path}")
        lines.append(f"  Report artifact : {self.report_path}")
        return "\n".join(lines)
