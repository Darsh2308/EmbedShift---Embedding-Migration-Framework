"""Core math: the mapper and evaluation metrics."""

from app.core.mapper import (
    DEFAULT_LAMBDAS,
    BaseMapper,
    LinearMapper,
    Mapper,
    load_mapper,
)
from app.core.evaluation import evaluate_mapper, recall_at_k, retrieve
from app.core.gate import build_report, evaluate_and_gate, run_gate
from app.core.metrics import l2_normalize, mean_cosine_similarity, mse
from app.core.mlp import MLPMapper
from app.core.pipeline import Embedder, run_migration, select_mapper
from app.core.transform import TransformProgress, transform_corpus, transform_to_store

__all__ = [
    "BaseMapper",
    "LinearMapper",
    "MLPMapper",
    "Mapper",
    "load_mapper",
    "select_mapper",
    "transform_to_store",
    "DEFAULT_LAMBDAS",
    "mse",
    "mean_cosine_similarity",
    "l2_normalize",
    "evaluate_mapper",
    "retrieve",
    "recall_at_k",
    "run_gate",
    "build_report",
    "evaluate_and_gate",
    "transform_corpus",
    "TransformProgress",
    "run_migration",
    "Embedder",
]
