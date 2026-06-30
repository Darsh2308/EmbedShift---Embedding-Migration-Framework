"""HTTP endpoints for the migration pipeline.

    POST /connect    register the old-vectors source           -> session_id
    POST /sample     sample + re-embed with the new model
    POST /train      fit the mapper
    POST /evaluate   run the confidence gate                    -> report
    POST /transform  full cutover (background job)              -> job_id
    GET  /jobs/{id}  poll a background job
    GET  /embedders  list registered new-model embedders
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app import services
from app.embedders import list_embedders
from app.models.api import (
    ConnectRequest,
    ConnectResponse,
    EvaluateRequest,
    EvaluateResponse,
    JobResponse,
    SampleRequest,
    SampleResponse,
    TrainRequest,
    TrainResponse,
    TransformRequest,
    TransformResponse,
)
from app.models.evaluation import MapperInfo

router = APIRouter(tags=["migration"])


def _get_session(session_id: str):
    session = services.SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"unknown session '{session_id}'")
    return session


@router.get("/embedders")
def embedders() -> dict:
    return {"embedders": list_embedders()}


@router.post("/connect", response_model=ConnectResponse)
def connect(req: ConnectRequest) -> ConnectResponse:
    try:
        session = services.connect(
            source_path=req.source_path,
            texts_path=req.texts_path,
            output_dir=req.output_dir,
            artifacts_dir=req.artifacts_dir,
            backend=req.backend,
            collection=req.collection,
            location=req.location,
            url=req.url,
            api_key=req.api_key,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except (ValueError, ImportError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return ConnectResponse(
        session_id=session.id, count=session.store.count(), dim=session.store.dim
    )


@router.post("/sample", response_model=SampleResponse)
def sample(req: SampleRequest) -> SampleResponse:
    session = _get_session(req.session_id)
    try:
        split = services.sample(
            session,
            req.embedder,
            sample_size=req.sample_size,
            sample_fraction=req.sample_fraction,
            validation_fraction=req.validation_fraction,
            seed=req.seed,
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return SampleResponse(
        session_id=session.id,
        sample_size=split.sample_size,
        train_size=split.train_size,
        val_size=split.val_size,
        d_old=split.d_old,
        d_new=split.d_new,
    )


@router.post("/train", response_model=TrainResponse)
def train(req: TrainRequest) -> TrainResponse:
    session = _get_session(req.session_id)
    try:
        mapper = services.train(
            session,
            mapper_kind=req.mapper_kind,
            use_cv=req.use_cv,
            lambda_=req.lambda_,
            cv_folds=req.cv_folds,
            cv_metric=req.cv_metric,
            normalize_output=req.normalize_output,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    cv_results = getattr(mapper, "cv_results_", None)
    cv = {str(k): v for k, v in cv_results.items()} if cv_results else None
    return TrainResponse(
        session_id=session.id,
        mapper=MapperInfo(
            kind=mapper.kind,
            d_old=mapper.d_old,
            d_new=mapper.d_new,
            lambda_=float(getattr(mapper, "lambda_", 0.0) or 0.0),
            normalize_output=mapper.normalize_output,
        ),
        mapper_kind=session.mapper_kind or mapper.kind,
        mapper_attempts=session.mapper_attempts or None,
        cv_results=cv,
    )


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest) -> EvaluateResponse:
    session = _get_session(req.session_id)
    try:
        report = services.evaluate(
            session,
            k=req.k,
            max_queries=req.max_queries,
            confidence_threshold=req.confidence_threshold,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return EvaluateResponse(session_id=session.id, passed=report.verdict.passed, report=report)


@router.post("/transform", response_model=TransformResponse)
def transform(req: TransformRequest) -> TransformResponse:
    session = _get_session(req.session_id)
    try:
        job = services.transform(
            session,
            output_collection=req.output_collection,
            batch_size=req.batch_size,
            force=req.force,
            resume=req.resume,
        )
    except services.GateNotPassedError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return TransformResponse(job_id=job.id, status=job.status)


@router.get("/jobs/{job_id}", response_model=JobResponse)
def job_status(job_id: str) -> JobResponse:
    job = services.JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job '{job_id}'")
    return JobResponse(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        result=job.result,
        error=job.error,
    )
