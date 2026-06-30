"""Request/response schemas for the migration API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from app.models.evaluation import ConfidenceReport, MapperInfo


# --------------------------- connect --------------------------- #
class ConnectRequest(BaseModel):
    backend: str = Field("file", description="Source backend: 'file' or 'qdrant'.")
    # file backend
    source_path: Optional[str] = Field(None, description="Path to old vectors (.jsonl/.npz/.npy).")
    # qdrant backend
    collection: Optional[str] = Field(None, description="Source collection (qdrant).")
    location: Optional[str] = Field(None, description="Qdrant location (':memory:' or a path).")
    url: Optional[str] = Field(None, description="Qdrant server URL.")
    api_key: Optional[str] = Field(None, description="Qdrant API key.")
    # common
    texts_path: Optional[str] = Field(None, description="Path to id->text jsonl for the sample.")
    output_dir: Optional[str] = None
    artifacts_dir: Optional[str] = None


class ConnectResponse(BaseModel):
    session_id: str
    count: int
    dim: int


# --------------------------- sample --------------------------- #
class SampleRequest(BaseModel):
    session_id: str
    embedder: str = Field("hashing", description="Name of a registered new-model embedder.")
    sample_size: Optional[int] = None
    sample_fraction: Optional[float] = None
    validation_fraction: Optional[float] = None
    seed: Optional[int] = None


class SampleResponse(BaseModel):
    session_id: str
    sample_size: int
    train_size: int
    val_size: int
    d_old: int
    d_new: int


# --------------------------- train --------------------------- #
class TrainRequest(BaseModel):
    session_id: str
    mapper_kind: Optional[str] = Field(None, description="auto | linear | mlp")
    use_cv: Optional[bool] = None
    lambda_: Optional[float] = None
    cv_folds: Optional[int] = None
    cv_metric: Optional[str] = None
    normalize_output: Optional[bool] = None


class TrainResponse(BaseModel):
    session_id: str
    mapper: MapperInfo
    mapper_kind: str
    mapper_attempts: Optional[list[dict]] = None
    cv_results: Optional[dict[str, float]] = None


# --------------------------- evaluate --------------------------- #
class EvaluateRequest(BaseModel):
    session_id: str
    k: Optional[int] = None
    max_queries: Optional[int] = None
    confidence_threshold: Optional[float] = None


class EvaluateResponse(BaseModel):
    session_id: str
    passed: bool
    report: ConfidenceReport


# --------------------------- transform --------------------------- #
class TransformRequest(BaseModel):
    session_id: str
    output_collection: Optional[str] = None
    batch_size: Optional[int] = None
    force: Optional[bool] = None
    resume: Optional[bool] = None


class TransformResponse(BaseModel):
    job_id: str
    status: str


# --------------------------- jobs --------------------------- #
class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: dict = {}
    result: Optional[dict] = None
    error: Optional[str] = None
