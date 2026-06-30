# Embedding Migration Framework — Phase-wise Development Plan

**Stack:** Python · FastAPI · NumPy
**Strategy:** Build the *math and quality* first on simple file-based data, wire it into an API, then add live DB connectors last.

Each phase lists: **Goal · Tasks · Deliverable · Done when**.

---

## Phase 0 — Project Setup (foundation)

**Goal:** A clean, runnable skeleton everyone can build on.

**Tasks**
- Init repo + virtual environment (`venv` / `uv`).
- `requirements.txt` / `pyproject.toml`: `fastapi`, `uvicorn`, `numpy`, `pydantic`, `pydantic-settings`, `pytest`, `python-multipart`.
- Project structure:
  ```
  app/
    main.py            # FastAPI app entrypoint
    config.py          # settings (env-based)
    api/               # routers/endpoints
    core/              # mapper, evaluation (the math)
    stores/            # VectorStore interface + adapters
    models/            # pydantic request/response schemas
    utils/
  tests/
  data/                # sample/exported vectors (gitignored)
  artifacts/           # saved mappers + reports (gitignored)
  ```
- Settings/config via env vars, `.env.example`.
- Basic logging setup.
- `GET /health` endpoint.
- `pytest` wired with one trivial passing test.
- `.gitignore`, `README.md` (point to `Info.md`).

**Deliverable:** `uvicorn app.main:app` runs, `/health` returns OK, `pytest` passes.
**Done when:** A new dev can clone, install, run the server, and hit `/health`.

---

## Phase 1 — Data Layer (VectorStore abstraction, file-based)

**Goal:** Read old vectors in and write mapped vectors out, decoupled from any specific DB.

**Tasks**
- Define `VectorStore` interface (Protocol/ABC):
  - `iter_vectors(batch_size)` → stream `(id, vector)`
  - `fetch_sample(n)` → random sample of `(id, vector)`
  - `upsert(collection, ids, vectors)`
  - `count()`
- Implement `FileStore`: load/save `.npy`, `.parquet`, `.jsonl` (`{id, vector}`).
- Batched/streaming reads (don't load millions into RAM at once).
- Helper to load matched **(old_vector, source_text)** pairs for the sample.
- Unit tests with a small synthetic dataset.

**Deliverable:** Load N old vectors from a file, sample M of them, write vectors back.
**Done when:** Round-trip read → sample → write works on synthetic data, tested.

---

## Phase 2 — The Mapper (core math)

**Goal:** Train and apply the linear mapper.

**Tasks**
- `Mapper` class:
  - `fit(X_old, Y_new)` → ridge regression `W = (XᵀX + λI)⁻¹ XᵀY`.
  - **Mean-center** old & new before fitting; store `μ_x`, `μ_y`.
  - `transform(X)` → `(X − μ_x) · W + μ_y`, optional **L2-normalize**.
  - Handles **dimension mismatch** (`d_old × d_new`).
  - `save()` / `load()` artifact (W + means + config).
- Tune `λ` (ridge) via small grid / cross-validation on the sample.
- Unit tests: known synthetic transform is recovered; dim-mismatch works; save/load is lossless.

**Deliverable:** `Mapper` that learns from pairs and transforms new old-vectors.
**Done when:** On synthetic data with a known linear relationship, the mapper recovers it to high accuracy.

---

## Phase 3 — Evaluation & Confidence Gate

**Goal:** Honestly measure how much quality the mapper retains — the project's credibility.

**Tasks**
- Train/validation split of the sample pairs (hold out a slice never seen in training).
- **recall@k** harness on a held-out query set:
  - `recall@k_max` (true-new vectors) — ceiling
  - `recall@k_mapped` (mapped vectors) — our method
  - `recall@k_old` (old vectors) — do-nothing baseline
- Compute **quality retained = `recall@k_mapped / recall@k_max`**.
- Also report cosine similarity `f(old)` vs true-new (secondary; never a substitute for recall).
- **Confidence gate:** pass/fail against a configurable threshold; produce a **confidence report** (JSON + human-readable).

**Deliverable:** A report stating expected quality *before* committing, with a go/no-go.
**Done when:** Report shows all three recall numbers + quality-retained + verdict on a test dataset.

---

## Phase 4 — Transform Pipeline (full cutover)

**Goal:** Apply the trained mapper to the entire corpus and produce the migrated index.

**Tasks**
- Stream all old vectors through `Mapper.transform` in batches.
- Write mapped vectors to a **new collection / output file** (never overwrite source).
- Progress tracking + resumability for large corpora.
- End-to-end orchestration: `sample → train → evaluate(gate) → transform`.
- Guard: refuse to transform if the confidence gate failed (unless forced).

**Deliverable:** Full corpus transformed to mapped vectors, ready to load into search.
**Done when:** A synthetic 100k-vector corpus is transformed start-to-finish via one pipeline call.

---

## Phase 5 — FastAPI Endpoints (wire it together)

**Goal:** Expose the pipeline over HTTP.

**Tasks**
- Endpoints:
  - `POST /connect` — register a data source (file path / DB config).
  - `POST /sample` — pull the sample + (re-embed via new model callable).
  - `POST /train` — fit the mapper.
  - `POST /evaluate` — run the confidence gate, return report.
  - `POST /transform` — full cutover.
  - `GET /jobs/{id}` — status of long-running jobs.
- Pydantic request/response schemas.
- Background jobs for long steps (transform/sample).
- Plug in the **new model** as a configurable callable (`text → vector`).
- API tests covering the happy path.

**Deliverable:** Complete migration runnable purely via API calls.
**Done when:** A scripted client runs connect → sample → train → evaluate → transform successfully.

---

## Phase 6 — Live Vector DB Connectors

**Goal:** Connect to real vector databases behind the same `VectorStore` interface.

**Tasks**
- Adapters: `QdrantStore`, `PineconeStore`, `WeaviateStore`, `PgVectorStore` (prioritize by need).
- Streaming reads (pagination) + batched upserts.
- Auth/config handling per backend.
- Integration tests (local Qdrant via Docker first — easiest).

**Deliverable:** Migrate directly against a running vector DB, no manual file export.
**Done when:** End-to-end migration runs against a local Qdrant instance.

---

## Phase 7 — Robustness & Quality Upgrades

**Goal:** Handle the hard cases and harden for real corpora.

**Tasks**
- **MLP mapper** (1–2 layers) as a fallback when the linear map underperforms on the validation slice.
- Auto-selection: try linear first, upgrade only if it fails the gate.
- Better λ / hyperparameter search.
- Error handling, input validation, large-scale memory/throughput tuning.
- Structured logging + metrics for each pipeline stage.

**Deliverable:** Framework that adapts mapper complexity to the data and survives messy inputs.
**Done when:** A dissimilar model pair triggers the MLP path and improves quality vs linear.

---

## Phase 8 — Packaging, Docs & Deployment

**Goal:** Ship it.

**Tasks**
- Dockerfile + `docker-compose` (API + sample Qdrant).
- CLI wrapper (optional) for non-API users.
- Full docs: setup, API reference, config options, the confidence report explained.
- Example end-to-end walkthrough with a public dataset.
- CI (lint + tests).

**Deliverable:** A deployable, documented product.
**Done when:** A new user can run a full migration from the README alone.

---

## Build Order Summary

```
Phase 0  Setup
Phase 1  Data layer (file-based)        ─┐
Phase 2  Mapper (math)                   │  Core value
Phase 3  Evaluation + confidence gate   ─┘  (prove it works)
Phase 4  Transform pipeline
Phase 5  FastAPI endpoints              ── Usable product
Phase 6  Live DB connectors             ─┐
Phase 7  Robustness + MLP                │  Production-ready
Phase 8  Packaging + deploy             ─┘
```

**Guiding principle:** Phases 1–3 are the heart — if the mapper + confidence gate prove the quality on real data, the rest is plumbing.
