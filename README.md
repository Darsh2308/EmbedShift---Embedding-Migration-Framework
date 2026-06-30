# Embedding Migration Framework

Migrate to a new embedding model **without re-embedding your entire corpus** —
*when the old and new models belong to the same family.*

Re-embed a small sample, learn a cheap "mapper" that translates old vectors into the
new model's space, verify quality with a confidence gate before committing, then
transform everything instantly.

- **What & why:** see [Info.md](Info.md)
- **Roadmap:** see [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md)
- **Config reference:** see [docs/CONFIGURATION.md](docs/CONFIGURATION.md)
- **Reading the report:** see [docs/CONFIDENCE_REPORT.md](docs/CONFIDENCE_REPORT.md)

**Stack:** Python · FastAPI · NumPy

---

## Problem statement

An embedding model is a translator: it turns each document into a vector, and search,
recommendations, and RAG all work by comparing those vectors. **Every model places
vectors in its own space** — different dimensions, different geometry — so a vector
made by Model A is meaningless to Model B.

The consequence: the moment you upgrade to a better model, **all your stored vectors
become useless.** The only accepted fix today is to **re-embed the entire corpus** with
the new model, which at scale (millions–billions of documents) is:

- **Expensive** — large API/GPU bills.
- **Slow** — long migration windows.
- **Risky** — it requires the original source text, which many teams never kept.

Because re-embedding is so painful, teams stay stuck on outdated models. **Can we switch
models without paying to re-embed everything?**

---

## How we solve it

Instead of re-translating every document, we **learn a small translator (the "mapper")**
that converts old-model vectors into the new model's space:

```
f(old_vector) ≈ new_vector
```

The pipeline (`sample → train → gate → transform`):

1. **Sample (1–5%).** Old vectors are already stored. Run *only the sample* through the
   new model to build matched `(old, new)` pairs. The old model is never needed.
2. **Train the mapper.** Fit a `LinearMapper` (SVD-based ridge regression with
   mean-centering, cross-validated λ, dimension-mismatch support). Upgrades to a small
   pure-NumPy `MLPMapper` only when linear underperforms on a held-out slice (`auto`).
   Seconds on a laptop — no GPU.
3. **Confidence gate.** On held-out queries, measure **recall@k** of the mapped index
   against a full-re-embedding gold standard, and compare to the do-nothing baseline.
   Report a single **quality-retained** number and a **pass/fail** verdict *before*
   committing. (See [docs/CONFIDENCE_REPORT.md](docs/CONFIDENCE_REPORT.md).)
4. **Transform.** Only if the gate passes, stream all old vectors through the mapper
   (one matrix multiply each — instant, nearly free) and write them to a **new**
   collection. The source is never touched, so rollback stays possible.

Key correctness rule: **queries are embedded directly with the new model, never mapped.**
The mapper only moves the stored corpus into the new space; queries already live there.

---

## Findings / Conclusion

We validated the approach end-to-end (up to a 100k-vector corpus, file + live Qdrant
backends) across model pairs. The result is sharp and worth stating plainly:

> **The mapper works for migrations within the same model family. It does not work for
> migrating across different model families.**

- ✅ **Same family** (same architecture / training lineage, e.g. a size or version bump
  within one provider's line). The two spaces are geometrically related, so a learned
  linear (or small MLP) map bridges them well. The gate **passes** — typically ~90%+
  quality retained — and you get most of the new model's quality for ~3–5% of the cost.

- ❌ **Different families** (e.g. one provider's model → an unrelated provider's model).
  The spaces encode fundamentally different geometry. No linear or small nonlinear map
  reliably bridges them; **quality-retained falls below the gate threshold and the
  migration fails the gate.**

**Why this is a hard limit, not a tuning problem.** The mapper can only re-project
information *already present* in the old vectors. A different-family model is different
precisely because it captures distinctions the old model never encoded — and **no mapper
can invent back information that was never there.** Larger MLPs and bigger samples shift
the margin slightly but do not change the conclusion.

**The confidence gate is therefore the product.** It detects the cross-family case
automatically and **refuses the migration** rather than silently shipping a degraded
index. When the gate fails, the only path to full quality is a real re-embedding.

**Bottom line:** this is a *same-family migration accelerator* — honest, gated, and
scalable — not a universal re-embedding eliminator.

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
  core/          # mapper + evaluation (the math)
  stores/        # VectorStore interface + adapters
  models/        # pydantic request/response schemas
  utils/         # logging, helpers
tests/           # pytest suite
data/            # input/exported vectors (gitignored)
artifacts/       # trained mappers + reports (gitignored)
```

## Status

All phases complete. 🎉

- **Phase 0 — Project setup.** Runnable FastAPI skeleton with `/health` and tests.
- **Phase 1 — Data layer.** `VectorStore` interface + `FileStore` (`.jsonl`/`.npz`/`.npy`), streaming reads, reproducible sampling, sample↔text pairing, write-back.
- **Phase 2 — The mapper.** `LinearMapper` (SVD-based ridge regression with mean-centering, optional L2-normalize, dimension-mismatch support), cross-validated λ selection, save/load artifacts, metrics (MSE/cosine).
- **Phase 3 — Evaluation + confidence gate.** recall@k harness (true-new gold standard, queries embedded with the new model), three-way comparison (ceiling / mapped / do-nothing), quality-retained, and a pass/fail confidence report (JSON + human-readable).
- **Phase 4 — Transform pipeline.** Resumable streaming full-corpus transform (checkpoint + byte-offset recovery) and the end-to-end orchestration `run_migration` (sample → train → gate → transform), gated by the confidence verdict. Verified on a 100k-vector corpus (~93% quality retained, same-family pair, transformed in seconds).
- **Phase 5 — FastAPI endpoints.** Stateful session API (`/connect`, `/sample`, `/train`, `/evaluate`, `/transform`), background transform jobs (`/jobs/{id}`), and a pluggable new-model embedder registry (`/embedders`).
- **Phase 6 — Live vector DB connectors.** `QdrantStore` and `InMemoryVectorStore` behind the same `VectorStore` interface; store-to-store transform with idempotent write-back to a new collection. Verified end-to-end against a local Qdrant.
- **Phase 7 — Robustness + MLP mapper.** Pure-NumPy `MLPMapper` (1–2 layers, Adam, weight decay, early stopping) and `auto` selection that upgrades from linear only when linear fails on a held-out slice.
- **Phase 8 — Packaging + deploy.** Dockerfile + docker-compose (API + Qdrant), `emf` CLI, configuration & confidence-report docs, a self-contained example, and GitHub Actions CI (lint + tests).
</content>
</invoke>
