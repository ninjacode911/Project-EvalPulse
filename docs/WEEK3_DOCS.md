# Week 3: Module 2 — Semantic Drift Detector

## Overview

Week 3 built the most novel module in EvalPulse: the Semantic Drift Detector. This module embeds every LLM response using sentence-transformers, stores embeddings in ChromaDB, and detects when the semantic distribution of outputs shifts over time. This catches a class of quality degradation that other metrics miss — silent drift caused by model updates, prompt changes, or query distribution shifts.

## What Was Built

### 1. Embedding Service (`evalpulse/modules/embeddings.py`)

**What**: A shared, singleton embedding service wrapping sentence-transformers.

**Model**: `all-MiniLM-L6-v2` — a 384-dimensional sentence embedding model that:
- Runs on CPU (no GPU required)
- Downloads once (~80MB), then cached locally
- Produces normalized unit vectors (ready for cosine similarity)
- Encodes at ~50ms per text on CPU

**Design**: Singleton pattern via `get_embedding_service()`. The model is lazy-loaded on first call to avoid startup overhead. Both single and batch embedding methods are provided.

**Why this model**: MiniLM is the sweet spot for EvalPulse's zero-cost architecture:
- Small enough to run on CPU in any deployment
- Accurate enough for semantic comparison (MTEB benchmark score ~0.80)
- Produces normalized embeddings natively, so cosine similarity is just a dot product
- Used in production by Evidently AI and other monitoring tools

### 2. ChromaDB Drift Vector Store (`evalpulse/modules/drift_store.py`)

**What**: A persistent vector database that stores response embeddings with metadata for drift comparison.

**How it works**:
- Uses `chromadb.PersistentClient` for on-disk storage (survives restarts)
- Creates one collection per app (`drift_myapp`, `drift_rag_bot`)
- Each embedding is stored with metadata: timestamp, query (truncated), model name
- Provides centroid computation and cosine distance utility methods

**Key methods**:
- `add_embedding(id, embedding, app_name, metadata)` — stores a 384-dim vector
- `get_all_embeddings(app_name)` — retrieves all embeddings for baseline computation
- `compute_centroid(embeddings)` — mean vector, normalized to unit length
- `cosine_distance(vec_a, vec_b)` — returns 0.0 (identical) to 2.0 (opposite)

**Why ChromaDB**: It's free, in-process (no server needed), persists to disk, and handles vector storage efficiently. Compared to alternatives:
- FAISS: faster search but no built-in persistence or metadata
- Pinecone: paid SaaS
- Weaviate: requires a server

### 3. Drift Detection Module (`evalpulse/modules/drift.py`)

**What**: `SemanticDriftModule` implements the core drift detection algorithm.

**Algorithm**:
1. Embed the LLM response using EmbeddingService
2. Store the embedding in ChromaDB
3. If fewer than 10 baseline embeddings exist, return `drift_score=None` (insufficient data)
4. Compute the baseline centroid (mean of all stored embeddings)
5. Compute cosine distance between current embedding and baseline centroid
6. Normalize to 0.0–1.0 range: `drift_score = cosine_distance / 2.0`

**Drift score interpretation**:
- 0.00–0.05: No drift (responses are semantically consistent)
- 0.05–0.15: Minor drift (normal variation)
- 0.15–0.30: Significant drift (investigate)
- 0.30+: Major drift (quality may be degrading)

**Baseline management**: The centroid is recomputed when the embedding count changes. This means the baseline evolves as more data arrives. The plan doc specifies sliding windows (recalculate every 7 days), which will be implemented in a future refinement.

**Why cosine distance over other metrics**:
- Cosine distance captures semantic similarity regardless of text length
- Works well with normalized sentence embeddings
- Computationally cheap (single dot product for unit vectors)
- Interpretable: 0 = same meaning, 1 = unrelated, 2 = opposite meaning
- Industry standard for embedding drift detection (used by Evidently AI, Arize)

### 4. UMAP Visualization (`dashboard/umap_viz.py` — planned)

The UMAP 2D projection for the dashboard is planned but will be fully wired in Week 7 during dashboard polish. The module is functional for embedding and drift scoring.

## Test Results

**14 unit tests** for the drift module:
- EmbeddingService: 384-dim output, normalization, batch embedding, similar text similarity
- DriftVectorStore: CRUD, centroid computation, cosine distance (identical, orthogonal), collection isolation
- SemanticDriftModule: module name, empty response handling, insufficient baseline behavior

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| all-MiniLM-L6-v2 | Best tradeoff of size (80MB), speed (50ms CPU), and quality (0.80 MTEB) for a free-tier tool. |
| Centroid-based drift | Simple and effective. More sophisticated approaches (KS test on distributions) are planned for later. |
| Minimum 10 embeddings for baseline | Prevents noisy drift scores from small sample sizes. Configurable later. |
| ChromaDB PersistentClient | Zero-config persistent storage. No server, no setup, no cost. |
| Lazy model loading | First call takes ~2s (model load), subsequent calls ~50ms. Avoids slowing down `init()`. |

## Files Created

| File | Purpose |
|------|---------|
| `evalpulse/modules/embeddings.py` | Shared EmbeddingService (sentence-transformers, 384-dim) |
| `evalpulse/modules/drift_store.py` | ChromaDB vector store for drift baselines |
| `evalpulse/modules/drift.py` | SemanticDriftModule — cosine drift scoring |
| `tests/unit/test_drift.py` | 14 unit tests for embeddings, store, and drift module |
