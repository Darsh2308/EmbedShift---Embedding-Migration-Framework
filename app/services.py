"""Service layer: migration sessions, background jobs, and the per-step logic.

The HTTP steps (connect/sample/train/evaluate/transform) share state, so each
migration is a server-side ``MigrationSession`` keyed by id. The long transform
runs in a background thread tracked as a ``Job`` the client can poll.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.core.gate import evaluate_and_gate
from app.core.mapper import BaseMapper
from app.core.pipeline import SampleSplit, prepare_sample, select_mapper
from app.core.transform import TransformProgress, transform_corpus, transform_to_store
from app.embedders import get_embedder
from app.models.evaluation import ConfidenceReport
from app.models.migration import MigrationConfig
from app.stores import load_texts, make_store
from app.stores.base import VectorStore
from app.utils.logging import get_logger

logger = get_logger(__name__)


class GateNotPassedError(Exception):
    """Raised when a transform is attempted but the confidence gate failed."""


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
@dataclass
class MigrationSession:
    id: str
    store: VectorStore
    config: MigrationConfig
    texts_path: Optional[str] = None
    backend: str = "file"
    embedder_name: Optional[str] = None
    split: Optional[SampleSplit] = None
    mapper: Optional[BaseMapper] = None
    mapper_kind: Optional[str] = None
    mapper_attempts: Optional[list[dict]] = None
    report: Optional[ConfidenceReport] = None


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, MigrationSession] = {}
        self._lock = threading.Lock()

    def create(
        self, store: VectorStore, config: MigrationConfig, texts_path: str | None, backend: str = "file"
    ) -> MigrationSession:
        sid = uuid.uuid4().hex[:12]
        session = MigrationSession(
            id=sid, store=store, config=config, texts_path=texts_path, backend=backend
        )
        with self._lock:
            self._sessions[sid] = session
        return session

    def get(self, sid: str) -> Optional[MigrationSession]:
        with self._lock:
            return self._sessions.get(sid)


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    id: str
    status: str = "pending"  # pending | running | completed | failed
    progress: dict = field(default_factory=dict)
    result: Optional[dict] = None
    error: Optional[str] = None


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        jid = uuid.uuid4().hex[:12]
        job = Job(id=jid)
        with self._lock:
            self._jobs[jid] = job
        return job

    def get(self, jid: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(jid)

    def update(self, job: Job, **changes) -> None:
        with self._lock:
            for k, v in changes.items():
                setattr(job, k, v)

    def submit(self, job: Job, target) -> None:
        def _run() -> None:
            self.update(job, status="running")
            try:
                result = target(job)
                self.update(job, result=result, status="completed")
            except Exception as exc:  # noqa: BLE001 - surface any failure to the client
                logger.exception("job %s failed", job.id)
                self.update(job, error=str(exc), status="failed")

        threading.Thread(target=_run, daemon=True).start()


SESSIONS = SessionManager()
JOBS = JobManager()


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def connect(
    source_path: str | None = None,
    texts_path: str | None = None,
    output_dir: str | None = None,
    artifacts_dir: str | None = None,
    backend: str = "file",
    collection: str | None = None,
    location: str | None = None,
    url: str | None = None,
    api_key: str | None = None,
    store: VectorStore | None = None,
) -> MigrationSession:
    if store is None:
        if backend == "file":
            store = make_store("file", source_path=source_path, output_dir=output_dir)
        elif backend == "qdrant":
            store = make_store(
                "qdrant", collection=collection, location=location, url=url, api_key=api_key
            )
        else:
            raise ValueError(f"unknown backend '{backend}'")
    config = MigrationConfig()
    if output_dir:
        config.output_dir = output_dir
    if artifacts_dir:
        config.artifacts_dir = artifacts_dir
    return SESSIONS.create(store, config, texts_path, backend=backend)


def sample(
    session: MigrationSession,
    embedder_name: str,
    *,
    sample_size: int | None = None,
    sample_fraction: float | None = None,
    validation_fraction: float | None = None,
    seed: int | None = None,
) -> SampleSplit:
    cfg = session.config
    if sample_size is not None:
        cfg.sample_size = sample_size
    if sample_fraction is not None:
        cfg.sample_fraction = sample_fraction
    if validation_fraction is not None:
        cfg.validation_fraction = validation_fraction
    if seed is not None:
        cfg.seed = seed

    embed = get_embedder(embedder_name)  # raises KeyError
    texts = load_texts(session.texts_path) if session.texts_path else None
    split = prepare_sample(session.store, embed, texts, cfg)
    session.split = split
    session.embedder_name = embedder_name
    # invalidate downstream state
    session.mapper = None
    session.report = None
    return split


def train(
    session: MigrationSession,
    *,
    mapper_kind: str | None = None,
    use_cv: bool | None = None,
    lambda_: float | None = None,
    cv_folds: int | None = None,
    cv_metric: str | None = None,
    normalize_output: bool | None = None,
) -> BaseMapper:
    if session.split is None:
        raise RuntimeError("no sample yet; call /sample first")
    cfg = session.config
    if mapper_kind is not None:
        cfg.mapper_kind = mapper_kind
    if use_cv is not None:
        cfg.use_cv = use_cv
    if lambda_ is not None:
        cfg.lambda_ = lambda_
    if cv_folds is not None:
        cfg.cv_folds = cv_folds
    if cv_metric is not None:
        cfg.cv_metric = cv_metric
    if normalize_output is not None:
        cfg.normalize_output = normalize_output

    mapper, kind, attempts = select_mapper(session.split, cfg)
    session.mapper = mapper
    session.mapper_kind = kind
    session.mapper_attempts = attempts
    session.report = None  # invalidate stale evaluation
    return mapper


def evaluate(
    session: MigrationSession,
    *,
    k: int | None = None,
    max_queries: int | None = None,
    confidence_threshold: float | None = None,
) -> ConfidenceReport:
    if session.mapper is None or session.split is None:
        raise RuntimeError("no mapper yet; call /train first")
    cfg = session.config
    if k is not None:
        cfg.k = k
    if max_queries is not None:
        cfg.max_queries = max_queries
    if confidence_threshold is not None:
        cfg.confidence_threshold = confidence_threshold

    report = evaluate_and_gate(
        session.split.val_old, session.split.val_new, session.mapper,
        threshold=cfg.confidence_threshold, k=cfg.k,
        max_queries=cfg.max_queries, seed=cfg.seed,
    )

    adir = Path(cfg.artifacts_dir)
    session.mapper.save(adir / f"{cfg.output_collection}_mapper.npz")
    report.save(adir / f"{cfg.output_collection}_report.json")
    session.report = report
    return report


def transform(
    session: MigrationSession,
    *,
    output_collection: str | None = None,
    batch_size: int | None = None,
    force: bool | None = None,
    resume: bool | None = None,
) -> Job:
    if session.mapper is None:
        raise RuntimeError("no mapper yet; call /train first")
    if session.report is None:
        raise RuntimeError("not evaluated yet; call /evaluate first")
    cfg = session.config
    if output_collection is not None:
        cfg.output_collection = output_collection
    if batch_size is not None:
        cfg.batch_size = batch_size
    if force is not None:
        cfg.force = force
    if resume is not None:
        cfg.resume = resume

    if not session.report.verdict.passed and not cfg.force:
        raise GateNotPassedError(
            "confidence gate failed; set force=true to transform anyway"
        )

    job = JOBS.create()
    store = session.store
    mapper = session.mapper
    backend = session.backend
    out = Path(cfg.output_dir) / f"{cfg.output_collection}.jsonl"
    collection = cfg.output_collection

    def target(job: Job) -> dict:
        def cb(p: TransformProgress) -> None:
            JOBS.update(
                job,
                progress={
                    "written": p.written,
                    "total": p.total,
                    "fraction": round(p.fraction, 4),
                    "batches": p.batches,
                },
            )

        if backend == "file":
            summary = transform_corpus(
                store, mapper, out, batch_size=cfg.batch_size, resume=cfg.resume, progress_cb=cb
            )
        else:
            # DB backend: upsert mapped vectors into a new collection (never the source).
            summary = transform_to_store(
                store, mapper, store, collection, batch_size=cfg.batch_size, progress_cb=cb
            )
        return {
            "transform": summary.model_dump(),
            "output_path": summary.output_path,
            "n_transformed": summary.n_written,
        }

    JOBS.submit(job, target)
    return job
