"""The migration pipeline — orchestrates the four steps end to end.

    Step 1  sample      : pull a small sample, embed its text with the NEW model
    Step 2  train       : fit the mapper on the sample's matched pairs
    Step 3  gate        : evaluate on a held-out slice, decide go/no-go (recall@k)
    Step 4  transform   : if the gate passes (or force), map the whole corpus

The new model is supplied as an ``Embedder`` callable (``texts -> vectors``) so
this module stays independent of any specific provider. ``prepare_sample`` is the
shared building block used by both ``run_migration`` (programmatic) and the
HTTP endpoints (step-by-step).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from app.core.evaluation import evaluate_mapper
from app.core.gate import evaluate_and_gate
from app.core.mapper import BaseMapper, LinearMapper
from app.core.mlp import MLPMapper
from app.core.transform import transform_corpus, transform_to_store
from app.models.migration import MigrationConfig, MigrationResult
from app.stores.base import VectorStore, make_sample_pairs
from app.stores.formats import load_texts
from app.utils.logging import get_logger

logger = get_logger(__name__)

#: A "new model": maps a list of texts to a (n, d_new) array of vectors.
Embedder = Callable[[Sequence[str]], np.ndarray]


@dataclass
class SampleSplit:
    """Matched sample pairs, shuffled and split into train / validation."""

    train_old: np.ndarray
    train_new: np.ndarray
    val_old: np.ndarray
    val_new: np.ndarray
    sample_size: int
    train_size: int
    val_size: int

    @property
    def d_old(self) -> int:
        return int(self.train_old.shape[1])

    @property
    def d_new(self) -> int:
        return int(self.train_new.shape[1])


def prepare_sample(
    store: VectorStore,
    embed: Embedder,
    texts: str | Path | Mapping[str, str] | None,
    config: MigrationConfig,
) -> SampleSplit:
    """Step 1: sample old vectors, re-embed their text, shuffle, split train/val.

    ``texts`` may be a path/mapping of id -> source text, or ``None`` to use each
    doc's id as its text (handy for embedders that look up by id).
    """
    total = store.count()
    if total == 0:
        raise ValueError("store is empty")

    sample_size = config.sample_size or max(1, round(config.sample_fraction * total))
    sample_size = min(sample_size, total)

    batch = store.fetch_sample(sample_size, seed=config.seed)
    if texts is None:
        text_map = {i: i for i in batch.ids}
    elif isinstance(texts, (str, Path)):
        text_map = load_texts(texts)
    else:
        text_map = texts
    pairs = make_sample_pairs(batch, text_map)

    new_vecs = np.asarray(embed(pairs.texts), dtype=np.float32)
    old_vecs = np.asarray(pairs.old_vectors, dtype=np.float32)
    if new_vecs.ndim != 2 or new_vecs.shape[0] != len(pairs):
        raise ValueError(f"embedder must return ({len(pairs)}, d_new); got shape {new_vecs.shape}")

    rng = np.random.default_rng(config.seed + 1)
    perm = rng.permutation(sample_size)
    old_vecs, new_vecs = old_vecs[perm], new_vecs[perm]

    val_size = int(round(config.validation_fraction * sample_size))
    if val_size <= config.k:
        raise ValueError(
            f"validation slice ({val_size}) must be larger than k ({config.k}); "
            "increase sample size / validation_fraction or lower k."
        )
    return SampleSplit(
        train_old=old_vecs[:-val_size],
        train_new=new_vecs[:-val_size],
        val_old=old_vecs[-val_size:],
        val_new=new_vecs[-val_size:],
        sample_size=sample_size,
        train_size=sample_size - val_size,
        val_size=val_size,
    )


def _fit_linear(old: np.ndarray, new: np.ndarray, config: MigrationConfig) -> LinearMapper:
    mapper = LinearMapper(normalize_output=config.normalize_output)
    if config.use_cv and old.shape[0] >= max(2, config.cv_folds):
        folds = min(config.cv_folds, old.shape[0])
        mapper.fit_cv(old, new, folds=folds, metric=config.cv_metric, seed=config.seed)
    else:
        mapper.fit(old, new, lambda_=config.lambda_)
    return mapper


def _fit_mlp(old: np.ndarray, new: np.ndarray, config: MigrationConfig) -> MLPMapper:
    mapper = MLPMapper(
        hidden=config.mlp_hidden,
        n_layers=config.mlp_layers,
        lr=config.mlp_lr,
        epochs=config.mlp_epochs,
        batch_size=config.mlp_batch_size,
        weight_decay=config.mlp_weight_decay,
        patience=config.mlp_patience,
        normalize_output=config.normalize_output,
        seed=config.seed,
    )
    return mapper.fit(old, new)


def train_mapper(split: SampleSplit, config: MigrationConfig) -> LinearMapper:
    """Step 2 (linear): fit the linear mapper on the full training pairs."""
    return _fit_linear(split.train_old, split.train_new, config)


def select_mapper(split: SampleSplit, config: MigrationConfig) -> tuple[BaseMapper, str, list[dict]]:
    """Choose the mapper per ``config.mapper_kind``.

    'linear' / 'mlp' fit that kind directly. 'auto' fits linear first and only
    upgrades to the MLP if linear underperforms on a held-out *selection* slice
    carved from the training data (so the gate's validation slice stays untouched).
    The chosen kind is refit on the full training set before returning.
    """
    kind = config.mapper_kind
    attempts: list[dict] = []

    if kind == "linear":
        return train_mapper(split, config), "linear", attempts
    if kind == "mlp":
        return _fit_mlp(split.train_old, split.train_new, config), "mlp", attempts

    # ---- auto ----
    n = split.train_old.shape[0]
    sel_size = int(round(0.2 * n))
    if sel_size <= config.k or n - sel_size < 2:
        # Too small to compare fairly — stick with the safe linear default.
        logger.info("auto: sample too small to trial an MLP; using linear")
        return train_mapper(split, config), "linear", attempts

    rng = np.random.default_rng(config.seed + 2)
    perm = rng.permutation(n)
    sel, fit = perm[:sel_size], perm[sel_size:]
    fit_old, fit_new = split.train_old[fit], split.train_new[fit]
    sel_old, sel_new = split.train_old[sel], split.train_new[sel]

    lin = _fit_linear(fit_old, fit_new, config)
    q_lin = evaluate_mapper(sel_old, sel_new, lin, k=config.k,
                            max_queries=config.max_queries, seed=config.seed).quality_retained
    attempts.append({"kind": "linear", "quality_retained": q_lin})
    logger.info("auto: linear quality_retained=%.3f on selection slice", q_lin)

    if q_lin >= config.confidence_threshold:
        return train_mapper(split, config), "linear", attempts  # refit linear on full train

    # Linear fell short — trial the MLP.
    mlp = _fit_mlp(fit_old, fit_new, config)
    q_mlp = evaluate_mapper(sel_old, sel_new, mlp, k=config.k,
                            max_queries=config.max_queries, seed=config.seed).quality_retained
    attempts.append({"kind": "mlp", "quality_retained": q_mlp})
    logger.info("auto: mlp quality_retained=%.3f on selection slice", q_mlp)

    if q_mlp > q_lin:
        logger.info("auto: upgrading to MLP mapper")
        return _fit_mlp(split.train_old, split.train_new, config), "mlp", attempts
    return train_mapper(split, config), "linear", attempts


def run_migration(
    store: VectorStore,
    embed: Embedder,
    texts: str | Path | Mapping[str, str] | None,
    config: MigrationConfig | None = None,
    dest_store: VectorStore | None = None,
) -> MigrationResult:
    """Run sample -> train -> gate -> transform and return the full result.

    If ``dest_store`` is given, mapped vectors are written there (a DB collection);
    otherwise they're streamed to a ``.jsonl`` file under ``config.output_dir``.
    """
    config = config or MigrationConfig()

    logger.info("Step 1/4: sampling and re-embedding")
    split = prepare_sample(store, embed, texts, config)

    logger.info("Step 2/4: selecting & training mapper on %d pairs (kind=%s)", split.train_size, config.mapper_kind)
    mapper, mapper_kind, mapper_attempts = select_mapper(split, config)

    logger.info("Step 3/4: evaluating on %d held-out pairs (k=%d)", split.val_size, config.k)
    report = evaluate_and_gate(
        split.val_old, split.val_new, mapper,
        threshold=config.confidence_threshold, k=config.k,
        max_queries=config.max_queries, seed=config.seed,
    )

    artifacts_dir = Path(config.artifacts_dir)
    mapper_path = mapper.save(artifacts_dir / f"{config.output_collection}_mapper.npz")
    report_path = report.save(artifacts_dir / f"{config.output_collection}_report.json")

    transformed = False
    skipped_reason = None
    summary = None
    output_path = None

    if report.verdict.passed or config.force:
        if config.force and not report.verdict.passed:
            logger.warning("Step 4/4: gate FAILED but force=True — transforming anyway")
        else:
            logger.info("Step 4/4: gate passed — transforming full corpus")
        if dest_store is not None:
            summary = transform_to_store(
                store, mapper, dest_store, config.output_collection, batch_size=config.batch_size
            )
        else:
            out = Path(config.output_dir) / f"{config.output_collection}.jsonl"
            summary = transform_corpus(store, mapper, out, batch_size=config.batch_size, resume=config.resume)
        transformed = True
        output_path = summary.output_path
    else:
        skipped_reason = "confidence gate failed; pass force=True to override"
        logger.warning("Step 4/4: SKIPPED — %s", skipped_reason)

    return MigrationResult(
        report=report,
        transformed=transformed,
        skipped_reason=skipped_reason,
        output_path=output_path,
        n_transformed=summary.n_written if summary else 0,
        transform=summary,
        mapper_path=str(mapper_path),
        report_path=str(report_path),
        sample_size=split.sample_size,
        train_size=split.train_size,
        val_size=split.val_size,
        mapper_kind=mapper_kind,
        mapper_attempts=mapper_attempts or None,
    )
