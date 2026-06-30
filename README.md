# Embedding Migration Framework

Migrate to a new embedding model **without re-embedding your entire corpus**.
Re-embed a small sample, learn a cheap "mapper" that translates old vectors into the
new model's space, verify quality before committing, then transform everything instantly.

- **What & why:** see [Info.md](Info.md)
- **Roadmap:** see [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)
- **Config reference:** see [docs/CONFIGURATION.md](docs/CONFIGURATION.md)
- **Reading the report:** see [docs/CONFIDENCE_REPORT.md](docs/CONFIDENCE_REPORT.md)

**Stack:** Python · FastAPI · NumPy

---

## Quickstart

```bash
# 1. Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (optional) configure
cp .env.example .env

# 4. Run the API
uvicorn app.main:app --reload

# 5. Check it's alive
curl http://127.0.0.1:8000/health
#  -> {"status":"ok", ...}

# Interactive docs: http://127.0.0.1:8000/docs
```

## API

Run a full migration over HTTP (interactive docs at `/docs`):

| Method | Path | Purpose |
|---|---|---|
| POST | `/connect` | Register the old-vectors source → `session_id` |
| POST | `/sample` | Sample + re-embed with the new model |
| POST | `/train` | Fit the mapper |
| POST | `/evaluate` | Run the confidence gate → report |
| POST | `/transform` | Full cutover (background job) → `job_id` |
| GET | `/jobs/{id}` | Poll a transform job |
| GET | `/embedders` | List registered new-model embedders |

The new model is a registered **embedder** (`texts → vectors`). Register yours at startup:

```python
from app.embedders import register_embedder
register_embedder("my-model", lambda texts: model.encode(list(texts)))
```

## Vector databases

File export is the default backend; live DB connectors sit behind the same
`VectorStore` interface. Qdrant is supported today (optional dependency):

```bash
pip install -r requirements-db.txt    # qdrant-client
```

```jsonc
// POST /connect against Qdrant instead of a file
{ "backend": "qdrant", "collection": "old", "url": "http://localhost:6333",
  "texts_path": "texts.jsonl" }
```

Mapped vectors are written to a **new** collection (`corpus_v2`) via idempotent
upserts — the source collection is never touched.

## CLI

```bash
# run a migration without writing any code (uses a registered embedder)
python -m app.cli migrate --source data/old.npz --texts data/texts.jsonl \
    --sample-fraction 0.03 --threshold 0.90

# bring your own new model: a callable `texts -> vectors` in any importable module
python -m app.cli migrate --source data/old.npz --embedder-module mymodels:embed

# serve the API
python -m app.cli serve --port 8000
```

Installed as the `emf` command after `pip install -e .`.

## Try it (no setup)

```bash
python examples/quickstart.py      # self-contained end-to-end demo
```

## Docker

```bash
docker compose up --build          # API on :8000 + Qdrant on :6333
```

## Run tests

```bash
pytest
```

## Project structure

```
app/
  main.py        # FastAPI entrypoint
  config.py      # env-based settings
  api/           # endpoints (routers)
  core/          # mapper + evaluation (the math)   [Phase 2-3]
  stores/        # VectorStore interface + adapters [Phase 1, 6]
  models/        # pydantic request/response schemas
  utils/         # logging, helpers
tests/           # pytest suite
data/            # input/exported vectors (gitignored)
artifacts/       # trained mappers + reports (gitignored)
```

## Status

- **Phase 0 — Project setup: complete.** Runnable FastAPI skeleton with `/health` and tests.
- **Phase 1 — Data layer: complete.** `VectorStore` interface + `FileStore` (`.jsonl`/`.npz`/`.npy`), streaming reads, reproducible sampling, sample↔text pairing, write-back.
- **Phase 2 — The mapper: complete.** `LinearMapper` (SVD-based ridge regression with mean-centering, optional L2-normalize, dimension-mismatch support), cross-validated λ selection, save/load artifacts, metrics (MSE/cosine).
- **Phase 3 — Evaluation + confidence gate: complete.** recall@k harness (true-new gold standard, queries embedded with the new model), three-way comparison (ceiling / mapped / do-nothing), quality-retained, and a pass/fail confidence report (JSON + human-readable).
- **Phase 4 — Transform pipeline: complete.** Resumable streaming full-corpus transform (checkpoint + byte-offset recovery) and the end-to-end orchestration `run_migration` (sample → train → gate → transform), gated by the confidence verdict. Verified on a 100k-vector corpus (~93% quality retained, transformed in seconds).
- **Phase 5 — FastAPI endpoints: complete.** Stateful session API (`/connect`, `/sample`, `/train`, `/evaluate`, `/transform`), background transform jobs (`/jobs/{id}`), and a pluggable new-model embedder registry (`/embedders`). Full migration runs purely over HTTP.
- **Phase 6 — Live vector DB connectors: complete.** `QdrantStore` (real connector, lazy `qdrant-client`) and `InMemoryVectorStore`, both behind the same `VectorStore` interface; store-to-store transform with idempotent write-back to a new collection; backend selection in `/connect`. Verified end-to-end against a local Qdrant (read `old` → map → write `corpus_v2`, 100% live recall).
- **Phase 7 — Robustness + MLP mapper: complete.** Pure-NumPy `MLPMapper` (1–2 layers, Adam, weight decay, early stopping) and `auto` selection that fits linear first and upgrades to the MLP only when linear fails on a held-out selection slice. On a nonlinear model pair the upgrade lifted quality 63%→86% and flipped the gate fail→pass.
- **Phase 8 — Packaging + deploy: complete.** Dockerfile + docker-compose (API + Qdrant), `emf` CLI, configuration & confidence-report docs, a self-contained example, and GitHub Actions CI (lint + tests).

All phases complete. 🎉
