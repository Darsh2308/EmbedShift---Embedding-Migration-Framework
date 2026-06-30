# Embedding Migration Framework — Project Info

A tool that lets a company switch from an old embedding model to a new one
**without re-embedding their entire corpus** — saving ~95% of the cost, compute, and time.

---

## 1. The Problem

- An embedding model is a **translator**: it turns each document into a list of numbers (a **vector**).
- Search, recommendations, and RAG all work by **comparing these vectors**.
- **Every model speaks its own number-language.** Each model places vectors in its own space — different dimensions, different geometry. A vector made by Model A is meaningless to Model B.
- So when you upgrade to a better model, **all your old stored vectors instantly become useless.**
- The only fix today is to **re-embed the whole corpus** with the new model. At scale (millions–billions of docs) that is:
  - **Expensive** — huge API/GPU cost.
  - **Slow** — long migration windows.
  - **Risky** — you need the original source text, which many teams never kept.
- Because it's so painful, companies stay **stuck on outdated models**.

---

## 2. What We Are Building

> Let a company migrate to a new embedding model **without re-embedding the whole corpus** — cutting migration cost and time by ~95% while keeping search quality high.

We are **not** claiming "zero re-embedding." We re-embed a **tiny sample**, learn from it, then cheaply transform the rest.

**Stack:** Python · FastAPI · NumPy

---

## 3. The Core Idea (in plain language)

Instead of re-translating every document, we **teach a small "translator" (the mapper)** that converts old-language numbers into new-language numbers:

```
f(old_vector) ≈ new_vector
```

- Re-embed a small sample → learn the pattern → apply it to everything else.
- The mapped vectors aren't *perfect* copies, but they're **close enough** that search still works well.

---

## 4. How It Works — Step by Step

(Example: a company with **1,000,000 documents**)

1. **Sample a small subset (1–5%, e.g. 30,000 docs).**
   - Their old vectors are already stored.
   - Run *only these* through the **new model** → **30,000 matched pairs** ("old looks like this, new looks like this").
   - The old model is never needed — old vectors are already saved.

2. **Train the mapper** on those 30,000 pairs.
   - Just solving one matrix equation (ridge regression). Takes **seconds on a laptop**.
   - The other 970,000 docs are **never touched** in this step.

3. **Confidence gate (quality check).**
   - Test the mapper on held-out pairs it never saw during training.
   - Produce a single number, e.g. *"search retains ~92% of full re-embedding quality."*
   - Good → proceed. Bad → **stop and warn the company before they commit.**
   - This gate is what makes the tool **trustworthy and sellable.**

4. **Transform everything (instant cutover).**
   - Run all 1,000,000 old vectors through the mapper — light math, instant, nearly free.
   - Load results into the search index. **The company is now fully switched.**
   - These mapped vectors are the **permanent answer — not a placeholder.**

**No Step 5.** We do *not* slowly re-embed everything in the background later — if you eventually re-embed everything, you've saved nothing. The framework ends cleanly at Step 4.

---

## 5. Steady State (after migration)

- New documents keep arriving → embed those **directly with the new model** (cheap; it's just the incoming trickle).
- So the final index is a **permanent mix**:
  - **Translated-old vectors** — the migrated back-catalog.
  - **True-new vectors** — documents added after the switch.
- Both live in the new model's space, so search works across the whole mix.

---

## 6. Doubt #1 — How are old embeddings "translated" without the new model?

**Clarification first:** the small sample (1–5%) *does* go through the new model. Only the other **95–99% never touch the new model** — those are the ones we translate cheaply.

- An embedding is just a **list of numbers (a vector)**.
- The mapper is a **matrix `W`** (shape `d_old × d_new`, e.g. `768 × 1536`).
- "Translating" one vector = **one matrix multiplication**:

  ```
  new_vector_approx = old_vector · W
  ```

- This is plain arithmetic — **no model, no GPU, no API call.** Running the real new model is billions of parameters across many layers; this is one multiply. That's why it's instant and nearly free.

**Why does a multiplication work?** Because we *learned* `W` from the matched pairs, where we knew both the old and the true-new vector. `W` captures the systematic relationship between the two spaces, and we reuse it on vectors we never ran through the new model.

**Analogy — Celsius → Fahrenheit:**
```
F = C × 9/5 + 32
```
You don't re-measure the temperature; you transform the number you already have. The mapper is the learned, high-dimensional version of this.

**Bonus:** because the mapper works on **vectors, not text**, we don't need the source text for the 95–99% — only for the small sample.

---

## 7. Doubt #2 — Will a bigger new model (e.g. 4B → 32B) give better retrieval?

- **Yes — the 32B model itself will almost certainly retrieve better** than the 4B.
- **But the mapper will NOT give you full 32B quality.** Why:
  - The mapper only **re-projects what's already inside your old 4B vectors.**
  - The 32B is better *precisely because* it captures distinctions the 4B threw away.
  - The mapper **cannot invent back** information the old model never encoded.
- **Rule of thumb:** the bigger the gap between old and new models, the more quality you lose in mapping. Mapping shines when models are *similar*; a 4B → 32B jump is a *large* gap.
- This is exactly why the **confidence gate (Step 3)** exists — it tells you, before committing, how much 32B quality you'd actually keep.

**Bottom line:** mapping 4B → 32B is cheap, instant, and **better than doing nothing** — it recovers *most* of the gain, not all. For *full* 32B quality, only real re-embedding gets you there.

---

## 8. How the User Connects a Vector Database

Our service only cares about **vectors going in and out** — not the DB's internals. There are two interactions:

1. **Read OUT** — pull stored old vectors (+ IDs). For the sample, also the matching source text.
2. **Write IN** — load the mapped vectors into a **new collection** (e.g. `corpus_v2`), never overwriting the old one (so rollback stays possible).

**Architecture — a connector/adapter layer.** One common interface, one adapter per backend:

```
VectorStore (interface)
├── iter_vectors(batch_size)        # stream (id, vector) for the full transform
├── fetch_sample(n)                 # random sample of (id, vector)
├── upsert(collection, ids, vecs)   # write mapped vectors back
└── count()

Adapters: FileStore · PineconeStore · QdrantStore · WeaviateStore · PgVectorStore
```

**v1 recommendation: start file-based.** The user exports old vectors to `.npy` / `.parquet` / `.jsonl` (`{id, vector}`) and points us at it.
- Every vector DB can export → universal.
- No auth/network/rate-limit complexity while we prove the math and quality.
- The back-catalog is frozen during migration anyway, so a static dump is correct.

Live DB connectors (Pinecone, Qdrant, etc.) come later as adapters behind the same interface.

---

## 9. The Mapper — A Bit More Detail

Start with the simplest, best option: a **linear map (ridge regression)**.

```
W = (XᵀX + λI)⁻¹ XᵀY          X = old vectors, Y = new vectors
f(x) = normalize( (x − μ_x) · W + μ_y )
```

- **No GPU, no training loop** — milliseconds on a laptop.
- Tiny artifact: one matrix `W` + two mean vectors. Easy to save/reload.
- Naturally handles **dimension mismatch** (e.g. 768 → 1536), since `W` is `d_old × d_new`.

Three details that make or break quality:
1. **Mean-center** both old and new vectors before fitting (store the means).
2. **L2-normalize** the output if the new model uses cosine similarity.
3. Only upgrade to a **small MLP (1–2 layers)** if the linear map isn't accurate enough — don't reach for it first (it overfits on a tiny sample).

**Query-time rule (easy to get wrong):**
- Corpus vectors are *approximate*-new; incoming **queries are embedded directly** with the new model (real-new).
- They match because the mapper pushed old vectors *into* the new model's space.
- **Do NOT map the query.** Embed it with the new model. Mapping the query would be a bug.

---

## 10. What the Company Provides

| They provide | Why |
|---|---|
| The old stored vectors | The bulk of the corpus — we reuse these, never recompute. |
| The new model (callable: `text → vector`) | To embed the sample + future incoming docs. |
| Source text **for the sample only** (1–5%) | To create the matched training pairs. |

**They don't need:** the old model, or the source text for the other 95–99%.
**We produce:** a trained mapper, a confidence report, and the transformed index.

---

## 11. Blockers & The Honest Core Limitation

| Blocker | Detail |
|---|---|
| Approximation loss | Mapped vectors aren't the real new vectors → some quality drop. |
| Model dissimilarity | Works well when models are *similar*; very different models lose more. |
| Need training pairs | Requires source text (or saved sample) to build old↔new pairs. |
| Quality measurement | Easy to build a mapper that *looks* fine but tanks real search. Benchmarking (Step 3) is mandatory. |

**Core limitation:** the mapper can only rework information **already in the old vectors.** If the new model is better because it captures distinctions the old one threw away, no mapper can invent that back.

---

## 12. Success Metric

> **Retrieval recall@k of mapped vectors vs. true-new vectors, on a held-out query set.**

Compare three things:
- **Ceiling** — true new vectors → `recall@k_max`
- **Our method** — mapped vectors → `recall@k_mapped`
- **Do-nothing** — keep the old model → `recall@k_old`

We must **beat do-nothing**, and **quality retained = `recall@k_mapped / recall@k_max`**.

⚠️ You can also report cosine similarity of `f(old)` vs true-new — but **never let cosine stand in for recall.** A mapper can look great on cosine and still hurt search.

---

## 13. Positioning

A **migration accelerator**, not a re-embedding eliminator.

> Pay to re-embed a small sample, get most of the quality, switch over instantly, and keep it permanently.

Honest, sellable, and scalable.

- No mainstream plug-and-play product does mapping-based migration today — **this is the gap we fill.**
- The underlying technique exists in research: cross-lingual embedding alignment, **relative representations** (Moschella et al.), and **model stitching**.
- Vector databases (Pinecone, Weaviate, Qdrant) handle re-indexing but offer **no mapping-based migration feature**.
