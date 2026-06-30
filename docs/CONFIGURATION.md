# Configuration Reference

All migration behavior is controlled by `MigrationConfig`
([app/models/migration.py](../app/models/migration.py)). Defaults are sensible;
override only what you need.

## Sampling (Step 1)

| Option | Default | Meaning |
|---|---|---|
| `sample_fraction` | `0.03` | Fraction of the corpus to re-embed (1–5% typical). |
| `sample_size` | `None` | Absolute sample size; overrides `sample_fraction` if set. |
| `validation_fraction` | `0.2` | Portion of the sample held out for the confidence gate. |
| `seed` | `0` | Random seed (reproducible sampling/splits). |

## Mapper (Step 2)

| Option | Default | Meaning |
|---|---|---|
| `mapper_kind` | `auto` | `auto` (linear, upgrade to MLP if it fails), `linear`, or `mlp`. |
| `normalize_output` | `true` | L2-normalize mapped vectors (use with cosine search). |
| `use_cv` | `true` | Cross-validate the ridge strength λ. |
| `lambda_` | `1.0` | Ridge λ when `use_cv=false`. |
| `cv_folds` | `5` | CV folds. |
| `cv_metric` | `cosine` | CV objective: `cosine` or `mse`. |

### MLP fallback (used by `auto` upgrade and `mlp`)

| Option | Default | Meaning |
|---|---|---|
| `mlp_hidden` | `256` | Hidden layer width. |
| `mlp_layers` | `1` | Hidden layers (1–3). |
| `mlp_lr` | `1e-3` | Adam learning rate. |
| `mlp_epochs` | `300` | Max epochs (early stopping usually stops sooner). |
| `mlp_batch_size` | `128` | Mini-batch size. |
| `mlp_weight_decay` | `1e-4` | L2 regularization. |
| `mlp_patience` | `20` | Early-stopping patience. |

## Evaluation + gate (Step 3)

| Option | Default | Meaning |
|---|---|---|
| `k` | `10` | recall@k cutoff. |
| `max_queries` | `1000` | Cap on eval queries (for speed). |
| `confidence_threshold` | `0.90` | Min quality-retained to pass the gate. |

## Transform (Step 4)

| Option | Default | Meaning |
|---|---|---|
| `output_collection` | `corpus_v2` | Destination collection / output file name. |
| `batch_size` | `1000` | Streaming batch size. |
| `resume` | `false` | Resume an interrupted file transform. |
| `force` | `false` | Transform even if the gate fails. |

## Paths

| Option | Default | Meaning |
|---|---|---|
| `output_dir` | `data` | Where the `.jsonl` output is written (file backend). |
| `artifacts_dir` | `artifacts` | Where the mapper + report are saved. |

## Environment variables

Set in `.env` (see [.env.example](../.env.example)): `APP_NAME`, `ENVIRONMENT`,
`LOG_LEVEL`, `DATA_DIR`, `ARTIFACTS_DIR`, `SAMPLE_FRACTION`, `CONFIDENCE_THRESHOLD`.
