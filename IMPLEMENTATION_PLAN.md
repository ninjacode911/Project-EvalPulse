
# EvalPulse — Complete Implementation Plan

> **Project**: EvalPulse — Open-Source LLM Evaluation & Drift Monitoring Platform
> **Author**: Ninjacode911
> **Version**: 1.0 — March 2026
> **Timeline**: 8 Weeks (75 tasks)
> **Cost**: $0/month (fully free-tier architecture)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Summary](#2-architecture-summary)
3. [Tech Stack](#3-tech-stack)
4. [Directory Structure](#4-directory-structure)
5. [Data Models](#5-data-models)
6. [Health Score Formula](#6-health-score-formula)
7. [Critical Path](#7-critical-path)
8. [Week 1 — Project Setup + SDK + Storage + Dashboard Shell](#week-1)
9. [Week 2 — Module 4: Response Quality Scorer](#week-2)
10. [Week 3 — Module 2: Semantic Drift Detector](#week-3)
11. [Week 4 — Module 1: Hallucination Scorer + Health Score](#week-4)
12. [Week 5 — Module 3: RAG Quality Evaluator](#week-5)
13. [Week 6 — Module 5: Prompt Regression Tester + CI](#week-6)
14. [Week 7 — Alerts + Dashboard Polish + Deployment Prep](#week-7)
15. [Week 8 — Launch: PyPI, Benchmarks, Docs, Demo](#week-8)
16. [Module Dependency Matrix](#module-dependency-matrix)
17. [Risk Register & Mitigations](#risk-register)
18. [Target Metrics](#target-metrics)
19. [Verification Checklist](#verification-checklist)

---

## 1. Project Overview

EvalPulse is the first free, open-source, unified LLM evaluation and drift monitoring platform. It lets any developer drop into their LLM application in under 30 minutes — no credit card, no cloud account, no paid API — and immediately see a live dashboard of hallucination rates, semantic drift, response quality trends, and alert thresholds, all for $0/month.


### What It Solves

| Problem | EvalPulse Solution |
|---------|-------------------|
| No free tool combines eval + drift + dashboard | Unified platform with 5 modules + live Gradio dashboard |
| LLM outputs degrade silently over time | Semantic drift detection via embedding cosine distance |
| Hallucination detection requires paid APIs | SelfCheckGPT (free) + Groq free-tier LLM-as-judge |
| RAG pipeline quality is hard to measure | Faithfulness + context relevance + answer relevancy scoring |
| Prompt changes cause regressions | Golden dataset regression testing + GitHub Actions CI |

### Five Evaluation Modules

| # | Module | What It Measures | Key Tech |
|---|--------|-----------------|----------|
| M1 | Hallucination Scorer | Factual grounding, claim verification | SelfCheckGPT + Groq LLM-as-judge |
| M2 | Semantic Drift Detector | Output distribution shift over time | sentence-transformers + ChromaDB |
| M3 | RAG Quality Evaluator | Faithfulness, context relevance, answer relevancy | Evidently AI RAGEvals + embeddings |
| M4 | Response Quality Scorer | Sentiment, toxicity, length, language, denial rate | Evidently TextEvals + detoxify |
| M5 | Prompt Regression Tester | Quality regressions on prompt/model changes | Evidently TestSuite + GitHub Actions |

---

## 2. Architecture Summary

EvalPulse has three layers:

```
+------------------------------------------------------------------+
|  YOUR LLM APPLICATION                                             |
|  from evalpulse import track                                      |
|  @track(app='my-chatbot')                                        |
|  def ask_llm(query): return llm.generate(query)                  |
+------------------------------------------------------------------+
         | ~2ms overhead (queue push only)
         v
+------------------------------------------------------------------+
|  SDK LAYER  (evalpulse/sdk.py)                                   |
|  @track decorator | EvalContext | @atrack (async)                |
|  Captures: query, context, response, latency, model, tags        |
|  Pushes EvalEvent to asyncio.Queue (non-blocking)                |
+------------------------------------------------------------------+
         |
         v
+------------------------------------------------------------------+
|  EVALUATION ENGINE  (evalpulse/worker.py)                        |
|  Background thread with asyncio event loop                       |
|  ThreadPoolExecutor dispatches to modules in parallel:           |
|                                                                  |
|  +------------+  +------------+  +------------+                  |
|  | Module 1   |  | Module 2   |  | Module 3   |                  |
|  | Halluc.    |  | Drift      |  | RAG Eval   |                  |
|  +------------+  +------------+  +------------+                  |
|  +------------+  +------------+                                  |
|  | Module 4   |  | Module 5   |                                  |
|  | Quality    |  | Regression |                                  |
|  +------------+  +------------+                                  |
|                                                                  |
|  Results merged -> Health Score computed -> Alerts checked        |
+------------------------------------------------------------------+
         |
         v
+------------------------------------------------------------------+
|  OBSERVABILITY LAYER                                             |
|                                                                  |
|  +------------------+     +-----------------------------+        |
|  | SQLite / Postgres |     | Gradio Dashboard            |        |
|  | eval_records      |<--->| Tab 1: Overview             |        |
|  | alerts            |     | Tab 2: Hallucination        |        |
|  +------------------+     | Tab 3: Semantic Drift       |        |
|                           | Tab 4: RAG & Quality        |        |
|  +------------------+     +-----------------------------+        |
|  | ChromaDB          |     Hosted on HuggingFace Spaces          |
|  | Drift embeddings  |     Auto-refresh every 5-30 seconds       |
|  +------------------+                                            |
+------------------------------------------------------------------+
```

### SDK Integration (3 Lines)

```python
# Pattern A — Decorator
from evalpulse import track

@track(app='my-chatbot', tags=['production'])
def ask_llm(query, context):
    return groq_client.chat(query, context)

# Pattern B — Context Manager (RAG)
from evalpulse import EvalContext

with EvalContext(app='my-rag', query=user_query, context=retrieved_docs) as ctx:
    response = llm.generate(user_query)
    ctx.log(response)

# Pattern C — Async (FastAPI)
from evalpulse import atrack

@app.post('/chat')
@atrack(app='api-chatbot')
async def chat_endpoint(query: str):
    return await async_llm.generate(query)
```

---

## 3. Tech Stack

All free. No credit card required at any point.

### Core Evaluation

| Tool | License / Free Tier | Role in EvalPulse |
|------|-------------------|-------------------|
| Evidently AI | Apache 2.0, 25M+ downloads | Core eval: 100+ metrics, TextEvals, DataDrift, TestSuites, RAG evals |
| sentence-transformers | Open source, runs on CPU | Embeddings (all-MiniLM-L6-v2, 80MB, fast CPU) |
| ChromaDB | Open source, in-process | Embedding storage for drift detection. On-disk persistence. |
| SelfCheckGPT | MIT | Reference-free hallucination detection via consistency sampling |
| Groq API | Free: 14,400 req/day, Llama-3.1-70B | Optional LLM-as-judge. Falls back to embeddings if unavailable. |
| detoxify | Open source, local | Toxicity scoring. No API cost. |
| langdetect | Open source | Language detection |

### Storage & Backend

| Tool | Free Tier | Role |
|------|-----------|------|
| SQLite | Built into Python | Default local storage for eval records + alerts |
| Neon.tech | Free serverless Postgres 512MB | Optional cloud storage for shared dashboards |
| ChromaDB (persisted) | Free, on-disk | Embedding store for drift baselines |

### Dashboard & Deployment

| Tool | Free Tier | Role |
|------|-----------|------|
| Gradio | Apache 2.0 | Interactive dashboard UI with live charts |
| HuggingFace Spaces | Free CPU/GPU instances | Hosts the Gradio dashboard |
| Plotly | Open source | Line charts, scatter plots, histograms |

### CI/CD & Developer Tools

| Tool | Free Tier | Role |
|------|-----------|------|
| GitHub Actions | 2000 min/month free | Prompt regression CI on every commit |
| pytest | Open source | Test runner |
| ruff | Open source | Linter + formatter |

---

## 4. Directory Structure

```
evalpulse/
├── evalpulse/                       # Main Python package
│   ├── __init__.py                  # Public API: track, EvalContext, atrack, init, shutdown
│   ├── sdk.py                       # Decorator + context manager + event queue
│   ├── worker.py                    # Background EvaluationWorker (asyncio + ThreadPool)
│   ├── models.py                    # EvalRecord, EvalEvent Pydantic models
│   ├── config.py                    # evalpulse.yml loader + EvalPulseConfig validation
│   ├── health_score.py              # Composite health score formula (0-100)
│   ├── alerts.py                    # Threshold checking + Alert model + persistence
│   ├── notifications.py             # Email (smtplib) + Slack webhook dispatch
│   ├── cli.py                       # CLI: evalpulse init | regression run | dashboard
│   ├── modules/
│   │   ├── __init__.py              # Module registry + get_default_modules()
│   │   ├── base.py                  # EvalModule ABC (evaluate, is_available)
│   │   ├── quality.py               # M4: sentiment, toxicity, length, language, denial
│   │   ├── drift.py                 # M2: embedding drift scoring
│   │   ├── hallucination.py         # M1: SelfCheckGPT + LLM-as-judge
│   │   ├── rag_eval.py              # M3: faithfulness, context relevance, answer relevancy
│   │   ├── regression.py            # M5: golden dataset testing + Evidently TestSuite
│   │   ├── embeddings.py            # Shared EmbeddingService (sentence-transformers)
│   │   ├── drift_store.py           # ChromaDB vector store for drift baselines
│   │   ├── groq_client.py           # Groq API wrapper with rate limiting
│   │   └── golden_dataset.py        # Golden dataset schema + loader
│   └── storage/
│       ├── __init__.py              # get_storage() factory
│       ├── base.py                  # StorageBackend ABC
│       ├── sqlite_store.py          # Default SQLite backend (WAL mode)
│       └── postgres_store.py        # Optional Neon Postgres backend
├── dashboard/
│   ├── app.py                       # Gradio Blocks app (4 tabs + auto-refresh)
│   ├── charts.py                    # Plotly chart builder functions
│   ├── umap_viz.py                  # UMAP 2D embedding projection
│   └── demo_data.py                 # Synthetic data generator for HF Spaces demo
├── tests/
│   ├── conftest.py                  # Shared fixtures (tmp_db, sample_records, etc.)
│   ├── unit/                        # Unit tests per module (~60+ tests)
│   │   ├── test_config.py
│   │   ├── test_models.py
│   │   ├── test_sqlite_store.py
│   │   ├── test_sdk.py
│   │   ├── test_worker.py
│   │   ├── test_quality.py
│   │   ├── test_drift.py
│   │   ├── test_drift_store.py
│   │   ├── test_hallucination.py
│   │   ├── test_groq_client.py
│   │   ├── test_rag_eval.py
│   │   ├── test_regression.py
│   │   ├── test_alerts.py
│   │   └── test_dashboard.py
│   ├── integration/                 # End-to-end pipeline tests
│   │   ├── test_smoke.py
│   │   ├── test_quality_flow.py
│   │   ├── test_evidently_quality.py
│   │   ├── test_drift_flow.py
│   │   ├── test_hallucination_flow.py
│   │   ├── test_rag_flow.py
│   │   ├── test_all_modules.py
│   │   ├── test_regression_flow.py
│   │   ├── test_cli.py
│   │   ├── test_full_system.py
│   │   └── test_final.py
│   └── benchmarks/                  # Performance + accuracy benchmarks
│       ├── test_sdk_overhead.py
│       ├── test_truthfulqa.py
│       ├── test_halubench.py
│       ├── test_drift_detection.py
│       ├── test_rag_quality.py
│       └── test_package_size.py
├── examples/
│   ├── quickstart.py               # Minimal 10-line example
│   ├── demo_script.py              # Live demo script (50 queries + narration)
│   ├── groq_chatbot/               # Sample instrumented Groq chatbot
│   ├── rag_pipeline/               # Sample RAG pipeline with evaluation
│   ├── fastapi_app/                # Sample async FastAPI integration
│   └── golden_datasets/            # Sample golden test datasets
│       ├── sample_golden.json
│       └── truthfulqa_subset.json
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                   # Lint + unit tests on push/PR
│   │   └── regression_tests.yml     # Prompt regression CI (daily + on golden changes)
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── PULL_REQUEST_TEMPLATE.md
├── pyproject.toml                   # Package metadata, deps, ruff, pytest config
├── evalpulse.yml.example            # Full config template with all fields documented
├── LICENSE                          # Apache 2.0
├── README.md                        # Comprehensive docs with quick start
├── CONTRIBUTING.md                  # Development guide
├── CHANGELOG.md                     # Release notes
├── LAUNCH_CHECKLIST.md              # Pre-launch verification
└── IMPLEMENTATION_PLAN.md           # This file
```

---

## 5. Data Models

### EvalEvent (SDK output — lightweight input to evaluation engine)

```python
class EvalEvent(BaseModel):
    id: str                          # UUID, auto-generated
    app_name: str                    # From @track(app=...) or config
    timestamp: datetime              # Capture time
    query: str                       # User input
    context: Optional[str] = None    # Retrieved docs (RAG only)
    response: str                    # LLM output
    model_name: str = "unknown"      # e.g. 'llama-3.1-70b-groq'
    latency_ms: int = 0              # Inference latency
    tags: List[str] = []             # Developer-set labels
```

### EvalRecord (Full evaluation result — stored in DB)

```python
class EvalRecord(BaseModel):
    id: str                                    # UUID
    app_name: str
    timestamp: datetime
    query: str
    context: Optional[str] = None
    response: str
    model_name: str
    latency_ms: int
    tags: List[str] = []

    # Module 1: Hallucination
    hallucination_score: float = 0.0           # 0.0 = grounded, 1.0 = hallucinated
    hallucination_method: str = "none"         # 'selfcheck' | 'llm_judge' | 'both'
    flagged_claims: List[str] = []

    # Module 2: Drift
    embedding_vector: List[float] = []         # 384-dim MiniLM embedding
    drift_score: Optional[float] = None        # Computed per window

    # Module 3: RAG
    faithfulness_score: Optional[float] = None
    context_relevance: Optional[float] = None
    answer_relevancy: Optional[float] = None
    groundedness_score: Optional[float] = None

    # Module 4: Quality
    sentiment_score: float = 0.5
    toxicity_score: float = 0.0
    response_length: int = 0
    language_detected: str = "en"
    is_denial: bool = False

    # Composite
    health_score: int = 0                      # 0-100
```

### SQLite Schema

```sql
CREATE TABLE eval_records (
    id              TEXT PRIMARY KEY,
    app_name        TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    query           TEXT NOT NULL,
    context         TEXT,
    response        TEXT NOT NULL,
    model_name      TEXT,
    latency_ms      INTEGER,
    tags            TEXT,             -- JSON array
    hallucination_score  REAL,
    hallucination_method TEXT,
    flagged_claims       TEXT,        -- JSON array
    embedding_vector     TEXT,        -- JSON array (384 floats)
    drift_score          REAL,
    faithfulness_score   REAL,
    context_relevance    REAL,
    answer_relevancy     REAL,
    groundedness_score   REAL,
    sentiment_score      REAL,
    toxicity_score       REAL,
    response_length      INTEGER,
    language_detected    TEXT,
    is_denial            INTEGER,     -- 0 or 1
    health_score         INTEGER,
    created_at           TEXT DEFAULT (datetime('now'))
);

CREATE INDEX idx_app_timestamp ON eval_records(app_name, timestamp);
CREATE INDEX idx_health_score  ON eval_records(health_score);

CREATE TABLE alerts (
    id              TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    severity        TEXT NOT NULL,    -- 'warning' | 'critical'
    metric          TEXT NOT NULL,
    value           REAL,
    threshold       REAL,
    message         TEXT,
    record_id       TEXT REFERENCES eval_records(id),
    created_at      TEXT DEFAULT (datetime('now'))
);
```

---

## 6. Health Score Formula

```python
health_score = (
    (1 - hallucination_score) * 0.35  +   # Highest weight: most critical
    (1 - drift_score)         * 0.25  +   # Second: silent degradation
    rag_groundedness_score    * 0.20  +   # Third: RAG quality
    response_quality_score    * 0.15  +   # Fourth: surface quality
    regression_pass_rate      * 0.05      # Fifth: stability
) * 100
```

| Score Range | Status | Dashboard Indicator |
|-------------|--------|-------------------|
| 90 - 100 | Healthy | Green |
| 75 - 89 | Monitoring | Yellow |
| 60 - 74 | Degrading | Orange |
| 40 - 59 | Critical | Red |
| 0 - 39 | Failing | Red (flashing) |

When a module score is `None` (e.g., no RAG context), its weight is redistributed proportionally among available modules.

---

## 7. Critical Path

Items marked **[CP]** — any delay here delays the entire project.

```
1.1 Scaffold ─> 1.2 Config ─> 1.4 SQLite ─> 1.6 SDK @track ─> 1.7 Worker
    ─> 2.1 Module Interface ─> 2.2 Quality Module ─> 2.4 Worker Dispatch
    ─> 3.1 Embeddings ─> 3.2 ChromaDB ─> 3.3 Drift Module
    ─> 4.1 Hallucination Module ─> 4.6 Health Score
    ─> 5.1 RAG Module ─> 6.1 Golden Dataset ─> 6.2 Regression Runner
    ─> 6.5 GitHub Actions CI ─> 7.1 Alert Engine ─> 8.5 PyPI Package
```

**17 critical-path tasks** out of 75 total.

---

## WEEK 1: Project Setup + SDK + SQLite + Blank Dashboard <a id="week-1"></a>

**Goal**: `pip install -e .` works. `@track` decorator captures LLM calls and writes to SQLite. Blank Gradio dashboard with 4 tabs launches.

**Deliverables**: Installable package, working SDK, SQLite storage, blank dashboard, CI pipeline.

---

### Day 1 (Mon) — Repository Scaffolding + Config + Models

#### Task 1.1 [CP] — Initialize git repository and project structure
- **Files to create**: `.gitignore`, `LICENSE` (Apache 2.0), `README.md` (skeleton), `pyproject.toml`, all `__init__.py` files, `examples/` directory
- **pyproject.toml includes**: package metadata, all dependencies with version bounds, `[project.scripts]` entry, build system (setuptools)
- **Dependencies**: evidently, sentence-transformers, chromadb, selfcheckgpt, groq, detoxify, langdetect, gradio, plotly, pydantic>=2.0, pyyaml, umap-learn
- **Dev dependencies**: pytest, ruff, pytest-timeout, pytest-asyncio
- **Acceptance**: `pip install -e .` succeeds; `python -c "import evalpulse"` works; git repo initialized
- **Status**: [ ]

#### Task 1.2 [CP] — Configuration system
- **Files to create**: `evalpulse/config.py`, `evalpulse.yml.example`
- **Implementation**:
  - `EvalPulseConfig(BaseModel)`: app_name, storage_backend, groq_api_key, thresholds dict, alert channels, module enable/disable flags, embedding_model name, drift_window_size, baseline_window
  - `@classmethod load(path)` reads YAML + validates
  - `get_config()` singleton with `@lru_cache`
  - All fields have sensible defaults (works with zero config)
- **Acceptance**: Valid YAML -> typed config; missing file -> all defaults; invalid values -> ValidationError
- **Status**: [ ]

#### Task 1.3 — Data models
- **Files to create**: `evalpulse/models.py`
- **Implementation**: `EvalRecord` + `EvalEvent` Pydantic BaseModels with all fields from Section 5
- **Acceptance**: Round-trip `model_dump()` / `model_validate()` works; Optional fields accept None
- **Status**: [ ]

---

### Day 2 (Tue) — SQLite Storage Backend

#### Task 1.4 [CP] — SQLite storage backend
- **Files to create**: `evalpulse/storage/base.py`, `evalpulse/storage/sqlite_store.py`, update `evalpulse/storage/__init__.py`
- **StorageBackend ABC methods**: `save(record)`, `save_batch(records)`, `query(filters, limit, offset)`, `count(filters)`, `get_by_id(id)`, `get_latest(n)`, `get_time_series(metric, start, end, granularity)`, `close()`
- **SQLiteStore implementation**:
  - WAL mode for concurrent reads
  - `threading.local()` for connection pooling
  - JSON serialization for List[float] and List[str] fields
  - Indexes on (app_name, timestamp), (health_score)
  - Auto-create tables on first connection
- **Acceptance**: Insert 100 records -> query by time range -> verify count; JSON fields round-trip; 4-thread concurrent writes don't corrupt
- **Status**: [ ]

#### Task 1.5 — Unit tests for config, models, storage
- **Files to create**: `tests/conftest.py`, `tests/unit/test_config.py`, `tests/unit/test_models.py`, `tests/unit/test_sqlite_store.py`
- **Fixtures**: `tmp_config()`, `tmp_db()`, `sample_eval_record()`, `sample_eval_event()`
- **Acceptance**: >= 15 test cases passing
- **Status**: [ ]

---

### Day 3 (Wed) — SDK Layer

#### Task 1.6 [CP] — SDK core: @track, @atrack, EvalContext
- **Files to create**: `evalpulse/sdk.py`, update `evalpulse/__init__.py`
- **Implementation**:
  - `track(func=None, *, app_name, model_name, tags)`: sync decorator. Measures latency, captures return value as response, first arg as query. Pushes `EvalEvent` to `asyncio.Queue`. Returns original return value unmodified.
  - `atrack(...)`: async version for `async def` functions
  - `EvalContext`: context manager using `contextvars.ContextVar` for scope propagation
  - Module-level `_event_queue: asyncio.Queue(maxsize=10000)`
  - `init(config_path=None)`: loads config, initializes storage, starts worker
  - `shutdown()`: drains queue, stops worker
- **Overhead budget**: < 2ms per call (timestamp diff + queue.put_nowait only)
- **Acceptance**: 1000 decorated calls < 2s overhead; events appear in queue; EvalContext propagates app_name
- **Status**: [ ]

#### Task 1.7 [CP] — Background worker: event consumer
- **Files to create**: `evalpulse/worker.py`
- **Implementation**:
  - `EvaluationWorker` class running in daemon thread
  - Owns asyncio event loop in that thread
  - Reads `EvalEvent` items from queue continuously
  - Week 1 behavior: converts EvalEvent -> EvalRecord with zeroed scores, saves to storage
  - Batch writes: collect up to 10 events or 1 second, whichever first
  - Graceful shutdown: drain queue, flush pending, stop loop
- **Acceptance**: Track 100 calls -> wait 2s -> 100 records in SQLite; clean shutdown
- **Status**: [ ]

---

### Day 4 (Thu) — SDK Tests + Dashboard Shell

#### Task 1.8 — SDK and worker unit tests
- **Files to create**: `tests/unit/test_sdk.py`, `tests/unit/test_worker.py`
- **Test cases**: overhead < 5ms, @atrack works, EvalContext propagation, queue overflow behavior, decorator preserves signature/return, worker start/stop, batch writing, event-to-record conversion, graceful shutdown
- **Acceptance**: All tests pass; overhead benchmark < 5ms/call
- **Status**: [ ]

#### Task 1.9 [CP] — Blank Gradio dashboard
- **Files to create**: `dashboard/app.py`, `dashboard/charts.py`
- **Implementation**:
  - Gradio Blocks with 4 tabs: Overview, Hallucination Deep-Dive, Semantic Drift, RAG & Quality
  - Each tab: placeholder message
  - Custom CSS: EvalPulse branding (gradient header, tab styling)
  - Health Score gauge placeholder showing "N/A"
  - Total eval count from storage
  - `create_app()` function that returns the Blocks object
- **Acceptance**: `python dashboard/app.py` launches at localhost:7860; 4 tabs visible; count shows 0
- **Status**: [ ]

---

### Day 5 (Fri) — CI + Smoke Test + Polish

#### Task 1.10 — GitHub Actions CI workflow
- **Files to create**: `.github/workflows/ci.yml`, `.github/workflows/regression_tests.yml` (stub)
- **CI workflow**: on push/PR -> checkout -> Python 3.11+3.12 -> `pip install -e ".[dev]"` -> `ruff check .` -> `ruff format --check .` -> `pytest tests/unit/ -v -x --timeout=60`
- **Status**: [ ]

#### Task 1.11 — End-to-end smoke test
- **Files to create**: `tests/integration/test_smoke.py`, `examples/quickstart.py`
- **Smoke test flow**: `init()` -> define dummy LLM -> `@track` -> call 10x -> wait for flush -> assert 10 records in SQLite -> `shutdown()`
- **Status**: [ ]

#### Task 1.12 — pyproject.toml refinement
- **Add**: `[tool.ruff]` (line-length=100, Python 3.11, select rules), `[tool.pytest.ini_options]`, `[project.optional-dependencies]` dev extra
- **Status**: [ ]

---

## WEEK 2: Module 4 — Response Quality Scorer <a id="week-2"></a>

**Goal**: Sentiment, toxicity, language detection, denial rate, response length. Quality scores flowing through the worker pipeline. Dashboard Tab 4 showing quality breakdown.

**Deliverables**: Working quality module, module dispatch system in worker, Dashboard Tab 4 with charts.

---

### Day 1 (Mon) — Module Interface + Quality Scorer Core

#### Task 2.1 [CP] — Module interface definition
- **Files to create**: `evalpulse/modules/base.py`
- **EvalModule ABC**: `name: str` property, `async evaluate(event: EvalEvent) -> dict` (returns partial EvalRecord fields), `is_available() -> bool` classmethod
- **Contract**: All 5 modules implement this interface identically
- **Status**: [ ]

#### Task 2.2 [CP] — Quality module: sentiment, length, language, denial
- **Files to create**: `evalpulse/modules/quality.py`
- **Sub-metrics**:
  - Sentiment: Evidently TextEvals `Sentiment()` or VADER fallback, normalized to 0.0-1.0
  - Response length: `len(response.split())`
  - Language detection: `langdetect.detect()` wrapped in try/except (default "en")
  - Denial detection: regex for "I cannot", "I'm unable", "As an AI", etc. -> bool
- **Returns**: `{sentiment_score, toxicity_score, response_length, language_detected, is_denial}`
- **Status**: [ ]

### Day 2 (Tue) — Toxicity + Module Dispatch

#### Task 2.3 — Toxicity scoring with detoxify
- **Modify**: `evalpulse/modules/quality.py`
- **Implementation**: `detoxify.Detoxify('original')` lazy-loaded singleton. Extract `toxicity` key. Handle first-run model download (~100MB).
- **Acceptance**: Toxic input > 0.5; benign < 0.1; model loads once
- **Status**: [ ]

#### Task 2.4 [CP] — Worker module dispatch system
- **Modify**: `evalpulse/worker.py`, `evalpulse/modules/__init__.py`
- **Implementation**: `register_module()`, parallel dispatch via `ThreadPoolExecutor`, merge partial results into EvalRecord, `get_default_modules(config)` factory
- **Acceptance**: Worker dispatches to quality module; records have non-zero scores
- **Status**: [ ]

### Day 3 (Wed) — Quality Tests

#### Task 2.5 — Quality module unit tests
- **Files to create**: `tests/unit/test_quality.py`
- **Coverage**: >= 12 tests — sentiment (positive/negative/neutral), toxicity (toxic/benign), language (English/French), denial (8 patterns), length, edge cases (empty string, unicode, very long text), concurrent evaluation
- **Status**: [ ]

#### Task 2.6 — Evidently TextEvals API verification
- **Files to create**: `tests/integration/test_evidently_quality.py`
- **Purpose**: Validate our understanding of Evidently API before deeper integration. Create Report with TextEvals on 20-row DataFrame.
- **Status**: [ ]

### Day 4 (Thu) — Dashboard Tab 4

#### Task 2.7 — Dashboard Tab 4: Quality breakdown
- **Modify**: `dashboard/charts.py`, `dashboard/app.py`
- **Charts**: sentiment histogram, toxicity timeline (line), language pie chart, denial rate bar chart, quality summary table (avg/p50/p95)
- **Features**: auto-refresh every 30 seconds
- **Status**: [ ]

### Day 5 (Fri) — Integration + Docs

#### Task 2.8 — Week 2 integration test
- **Files to create**: `tests/integration/test_quality_flow.py`
- **Flow**: init -> track 20 calls (positive, negative, toxic, denial, foreign language) -> verify scores populated correctly
- **Status**: [ ]

#### Task 2.9 — Update README with quality module docs
- **Status**: [ ]

---

## WEEK 3: Module 2 — Semantic Drift Detector <a id="week-3"></a>

**Goal**: sentence-transformers embeds responses. ChromaDB stores baseline embeddings. Cosine distance drift scoring with sliding windows. Dashboard Tab 3 with drift timeline and UMAP projection.

**Deliverables**: Embedding service, ChromaDB store, drift module, UMAP visualization, Dashboard Tab 3.

---

### Day 1 (Mon) — Embedding Service + ChromaDB

#### Task 3.1 [CP] — Embedding service
- **Files to create**: `evalpulse/modules/embeddings.py`
- **EmbeddingService class**: lazy-load `SentenceTransformer('all-MiniLM-L6-v2', device='cpu')`, `embed(text) -> list[float]` (384-dim), `embed_batch(texts)`, singleton via `get_embedding_service()`, normalized embeddings
- **Acceptance**: 384-dim output; 100 texts < 5s on CPU; unit-normalized
- **Status**: [ ]

#### Task 3.2 [CP] — ChromaDB drift vector store
- **Files to create**: `evalpulse/modules/drift_store.py`
- **DriftVectorStore class**: `PersistentClient(path=...)`, collection per app_name, `add_embedding()`, `get_baseline_embeddings(window)`, `get_recent_embeddings(window)`, `compute_centroid()`, singleton
- **Acceptance**: Add 100 embeddings -> retrieve by window -> centroid correct
- **Status**: [ ]

### Day 2 (Tue) — Drift Detection Algorithm

#### Task 3.3 [CP] — Drift detection module
- **Files to create**: `evalpulse/modules/drift.py`
- **SemanticDriftModule**: embed response -> store in ChromaDB -> cosine distance vs baseline centroid -> normalize to 0.0-1.0
- **Drift logic**:
  - If < 10 baseline embeddings: return `drift_score=None`
  - Score = `1 - cosine_similarity(current, baseline_centroid)` / 2
  - Configurable baseline window (default: 7 days) and comparison window (default: 1 hour)
- **Acceptance**: Same-topic < 0.1; different-topic > 0.3; None when insufficient baseline
- **Status**: [ ]

#### Task 3.4 — Register drift module in worker
- **Status**: [ ]

### Day 3 (Wed) — Tests + UMAP

#### Task 3.5 — Drift module unit tests (>= 10 tests)
- **Files**: `tests/unit/test_drift.py`, `tests/unit/test_drift_store.py`
- **Coverage**: embedding dimensions, normalization, ChromaDB persistence, drift scoring, baseline filtering, centroid computation
- **Status**: [ ]

#### Task 3.6 — UMAP 2D projection visualization
- **Files to create**: `dashboard/umap_viz.py`
- **Implementation**: `create_umap_projection(embeddings, labels, timestamps)` -> Plotly scatter with color = time bucket ("baseline" vs "recent"), hover with query text
- **Status**: [ ]

### Day 4 (Thu) — Dashboard Tab 3

#### Task 3.7 — Dashboard Tab 3: Semantic Drift
- **Modify**: `dashboard/charts.py`, `dashboard/app.py`
- **Charts**: drift score timeline with threshold line (0.15), drift status badge (red/yellow/green), UMAP 2D projection, query drift analysis table
- **Status**: [ ]

### Day 5 (Fri) — Integration + Performance

#### Task 3.8 — Drift integration test
- **Files to create**: `tests/integration/test_drift_flow.py`
- **Flow**: 50 "weather" queries (baseline) -> 10 "cooking" queries (drift) -> verify drift > 0.2 for cooking -> verify UMAP differentiates clusters
- **Acceptance**: TPR >= 90% on obvious topic shift
- **Status**: [ ]

#### Task 3.9 — SDK overhead benchmark
- **Files to create**: `tests/benchmarks/test_sdk_overhead.py`
- **Benchmark**: 10,000 decorated vs undecorated calls. Assert < 5ms overhead at p99. Report p50/p95/p99.
- **Status**: [ ]

---

## WEEK 4: Module 1 — Hallucination Scorer + Health Score <a id="week-4"></a>

**Goal**: SelfCheckGPT consistency detection + optional Groq LLM-as-judge. Combined multi-method scoring. Health score computation. Dashboard Tab 2 + Tab 1 Overview updates.

**Deliverables**: Hallucination module, Groq client, health score calculator, Dashboard Tabs 1 & 2.

---

### Day 1 (Mon) — SelfCheckGPT + Groq Client

#### Task 4.1 [CP] — SelfCheckGPT scorer
- **Files to create**: `evalpulse/modules/hallucination.py`
- **HallucinationModule**:
  - Method 1 (SelfCheckGPT): generate N stochastic samples via Groq -> NLI consistency check -> per-sentence scores -> aggregate to 0.0-1.0
  - Fallback (no Groq): embedding similarity between response and query+context. Low similarity = potential hallucination.
  - Extract `flagged_claims`: sentences with consistency < 0.5
  - Lazy-load SelfCheckGPT model
- **Acceptance**: Factual < 0.3; fabricated > 0.5
- **Status**: [ ]

#### Task 4.2 — Groq API client wrapper
- **Files to create**: `evalpulse/modules/groq_client.py`
- **GroqClient**: token-bucket rate limiter (10 req/min), `generate_samples(prompt, n=3, temp=0.7)`, `judge_hallucination(query, context, response) -> float`, `is_available()`, retry with exponential backoff on 429
- **Acceptance**: 5 calls succeed; 11th throttled; unavailable -> graceful fallback
- **Status**: [ ]

### Day 2 (Tue) — LLM-as-Judge + Tests

#### Task 4.3 — LLM-as-judge hallucination scoring
- **Modify**: `evalpulse/modules/hallucination.py`
- **Implementation**: structured prompt -> parse 0-1 score -> combined scoring `0.6 * selfcheck + 0.4 * judge` -> set `hallucination_method`
- **Status**: [ ]

#### Task 4.4 — Hallucination + Groq client tests (>= 12 tests)
- **Files**: `tests/unit/test_hallucination.py`, `tests/unit/test_groq_client.py`
- **Coverage**: SelfCheckGPT scoring (mocked), Groq client rate limiter, judge prompt, combined weights, fallback, flagged claims, edge cases, retry logic
- **Status**: [ ]

### Day 3 (Wed) — Dashboard Tab 2

#### Task 4.5 — Dashboard Tab 2: Hallucination Deep-Dive
- **Charts**: hallucination rate over time (line + threshold), by model (grouped bar), flagged claims table, consistency scatter (selfcheck vs judge), distribution histogram
- **Status**: [ ]

### Day 4 (Thu) — Health Score

#### Task 4.6 [CP] — Health score computation
- **Files to create**: `evalpulse/health_score.py`
- **Implementation**: weighted composite formula (see Section 6), None-handling with proportional weight redistribution, 0-100 integer clamping, `compute_aggregate_health(records, window)`
- **Acceptance**: Perfect = 100; worst = 0; partial scores redistribute correctly
- **Status**: [ ]

#### Task 4.7 — Integrate health score into worker
- **Modify**: `evalpulse/worker.py` — compute after all modules, before save
- **Status**: [ ]

### Day 5 (Fri) — Dashboard Overview + Integration

#### Task 4.8 — Dashboard Tab 1: Overview
- **Charts**: health gauge (Plotly indicator, color zones), hallucination rate card, drift status card, total evals count, trend sparklines, alert log placeholder
- **Status**: [ ]

#### Task 4.9 — Week 4 integration test
- **Files to create**: `tests/integration/test_hallucination_flow.py`
- **Flow**: mix of factual + fabricated responses -> verify scores differentiated -> health scores computed
- **Status**: [ ]

---

## WEEK 5: Module 3 — RAG Quality Evaluator <a id="week-5"></a>

**Goal**: Faithfulness, context relevance, answer relevancy scoring. Dashboard Tab 4 extended with RAG metrics. Cross-module integration validated.

**Deliverables**: RAG module, Dashboard Tab 4 with RAG charts, data refresh architecture.

---

### Day 1 (Mon) — RAG Module

#### Task 5.1 [CP] — RAG quality evaluation module
- **Files to create**: `evalpulse/modules/rag_eval.py`
- **RAGQualityModule**:
  - Context relevance = cosine_sim(query_embedding, context_embedding)
  - Faithfulness = cosine_sim(response_sentences, context_chunks) via embedding
  - Answer relevancy = cosine_sim(query_embedding, response_embedding)
  - Groundedness = weighted average of faithfulness + context_relevance
  - Skip if `context is None` (return all None)
- **Acceptance**: Relevant context > 0.7; irrelevant < 0.4; no context -> None
- **Status**: [ ]

#### Task 5.2 — Evidently Report integration for batch RAG analysis
- **Status**: [ ]

### Day 2 (Tue) — Tests + Worker

#### Task 5.3 — RAG module unit tests (>= 10 tests)
- **Status**: [ ]

#### Task 5.4 — Register RAG module in worker
- **Status**: [ ]

### Day 3 (Wed) — Dashboard Tab 4 Extension

#### Task 5.5 — Dashboard Tab 4: RAG metrics
- **Charts**: groundedness timeline, RAG quality radar chart (4 axes), quality heatmap by model/time, CSV export button
- **Status**: [ ]

### Day 4 (Thu) — Dashboard Refresh

#### Task 5.6 — Dashboard data refresh architecture
- **Implementation**: `DataProvider` with 5-second cache TTL, `gr.Timer` auto-refresh (30s), loading states, empty-state graceful messages
- **Acceptance**: New data visible within 30s; empty DB shows helpful message; no flicker
- **Status**: [ ]

### Day 5 (Fri) — Integration + Benchmarks

#### Task 5.7 — RAG integration test + benchmark
- **Files**: `tests/integration/test_rag_flow.py`, `tests/benchmarks/test_rag_quality.py`
- **Flow**: 10 relevant + 10 irrelevant context calls -> verify score differentiation
- **Status**: [ ]

#### Task 5.8 — Cross-module integration test
- **Files to create**: `tests/integration/test_all_modules.py`
- **Flow**: 30 diverse calls (RAG/non-RAG, factual/hallucinated, on/off-topic) -> all scores populated -> health scores valid -> no module interference
- **Status**: [ ]

---

## WEEK 6: Module 5 — Prompt Regression Tester + CI <a id="week-6"></a>

**Goal**: Golden dataset testing, Evidently TestSuite regression, GitHub Actions CI, CLI commands.

**Deliverables**: Regression module, golden dataset infrastructure, CLI, GitHub Actions workflow, Dashboard regression display.

---

### Day 1 (Mon) — Golden Dataset + Runner

#### Task 6.1 [CP] — Golden dataset schema and loader
- **Files**: `evalpulse/modules/golden_dataset.py`, `examples/golden_datasets/sample_golden.json`
- **Models**: `GoldenExample` (query, expected_response, context, tags, thresholds), `GoldenDataset` (name, version, examples)
- **Status**: [ ]

#### Task 6.2 [CP] — Regression test runner
- **Files to create**: `evalpulse/modules/regression.py`
- **Implementation**: batch mode, run golden examples through LLM + all modules, compare against thresholds, `RegressionResult` + `RegressionFailure` models
- **Acceptance**: Perfect LLM passes; garbage LLM fails with clear violations
- **Status**: [ ]

### Day 2 (Tue) — Evidently TestSuite + Tests

#### Task 6.3 — Evidently TestSuite integration
- **Implementation**: structured assertions, HTML report generation, current vs previous comparison
- **Status**: [ ]

#### Task 6.4 — Regression module tests (>= 8 tests)
- **Status**: [ ]

### Day 3 (Wed) — GitHub Actions + CLI

#### Task 6.5 [CP] — GitHub Actions regression workflow
- **Complete**: `.github/workflows/regression_tests.yml` — triggers on manual/daily/golden-change, artifact upload, PR blocking on fail
- **Create**: `evalpulse/cli.py` — `evalpulse init`, `evalpulse regression run --dataset PATH`, `evalpulse dashboard`
- **Status**: [ ]

### Day 4 (Thu) — Dashboard Regression

#### Task 6.6 — Dashboard regression results display
- **Charts**: pass rate over time (bar), failure table (expandable), before/after comparison
- **Status**: [ ]

### Day 5 (Fri) — Integration + CLI Tests

#### Task 6.7 — Regression test with TruthfulQA subset
- **Files**: `tests/integration/test_regression_flow.py`, `examples/golden_datasets/truthfulqa_subset.json`
- **Status**: [ ]

#### Task 6.8 — CLI end-to-end test
- **Files**: `tests/integration/test_cli.py`
- **Status**: [ ]

---

## WEEK 7: Alerts + Dashboard Polish + Deployment Prep <a id="week-7"></a>

**Goal**: Alert engine with notifications, dashboard UX polish, HF Spaces deployment config, optional Postgres backend.

**Deliverables**: Alert engine, email + Slack notifications, polished dashboard, HF Spaces config, Postgres backend.

---

### Day 1 (Mon) — Alert Engine

#### Task 7.1 [CP] — Alert engine with configurable thresholds
- **Files to create**: `evalpulse/alerts.py`
- **AlertEngine**: check record against thresholds from evalpulse.yml, `Alert` model (id, timestamp, severity, metric, value, threshold, message, record_id), deduplication via cooldown (default 5 min), persistence to SQLite alerts table
- **Thresholds**: hallucination > 0.3, drift > 0.15, groundedness < 0.65, toxicity > 0.05, regression fail > 0.10
- **Status**: [ ]

#### Task 7.2 — Notification dispatch
- **Files to create**: `evalpulse/notifications.py`
- **Implementation**: `send_email()` via smtplib (Gmail), `send_slack()` via webhook POST, async fire-and-forget, graceful failure logging
- **Status**: [ ]

### Day 2 (Tue) — Alert Integration + Dashboard

#### Task 7.3 — Integrate alerts into worker pipeline
- **Modify**: `evalpulse/worker.py` (run AlertEngine.check after health score), `evalpulse/storage/sqlite_store.py` (add alerts table + methods)
- **Status**: [ ]

#### Task 7.4 — Dashboard alert log
- **Tab 1**: real alert table with severity badges, filtering, sort by time, alert frequency chart
- **Status**: [ ]

### Day 3 (Wed) — Dashboard Polish + HF Spaces

#### Task 7.5 — Dashboard visual polish
- **Polish**: consistent color scheme, responsive layout, animated health gauge, empty states ("No evaluations yet. Instrument your LLM with @track to get started."), loading spinners, mobile-friendly, footer
- **Status**: [ ]

#### Task 7.6 — HuggingFace Spaces deployment config
- **Files to create**: `dashboard/Dockerfile`, `dashboard/requirements.txt`, `dashboard/README.md` (HF metadata YAML header)
- **Acceptance**: `docker build` succeeds; Gradio starts on port 7860
- **Status**: [ ]

### Day 4 (Thu) — Tests

#### Task 7.7 — Alert engine tests (>= 10 tests)
- **Coverage**: all 5 threshold checks, cooldown deduplication, persistence, severity classification, notification dispatch (mocked)
- **Status**: [ ]

#### Task 7.8 — Dashboard rendering tests
- **Coverage**: all chart functions return valid Plotly Figures, `create_app()` works, DataProvider caching, empty states
- **Status**: [ ]

### Day 5 (Fri) — Postgres + Full System Test

#### Task 7.9 — Optional Neon Postgres backend
- **Files to create**: `evalpulse/storage/postgres_store.py`
- **Implementation**: same StorageBackend interface, auto-create tables, connection pooling, optional dependency in pyproject.toml
- **Status**: [ ]

#### Task 7.10 — Full system integration test
- **Files to create**: `tests/integration/test_full_system.py`
- **Flow**: init -> 50 diverse calls -> all modules score -> alerts fire -> dashboard renders all 4 tabs -> clean shutdown
- **Status**: [ ]

---

## WEEK 8: Launch — PyPI, Benchmarks, Docs, Demo <a id="week-8"></a>

**Goal**: Production-ready v0.1.0. PyPI published, HuggingFace Spaces live, benchmarks validated, examples complete, launch materials ready.

**Deliverables**: Published PyPI package, live HF Spaces demo, benchmark results, 3 example apps, comprehensive docs, launch checklist.

---

### Day 1 (Mon) — Documentation

#### Task 8.1 — Comprehensive README
- **Sections**: badges, "What is EvalPulse?", architecture diagram, Quick Start (5 lines), installation, config reference, all 5 module descriptions, dashboard screenshots, benchmarks table, contributing link, license
- **Status**: [ ]

#### Task 8.2 — Example applications
- **Create**: `examples/groq_chatbot/`, `examples/rag_pipeline/`, `examples/fastapi_app/` — each with `main.py` + `README.md`
- **Each demonstrates**: at least 2 EvalPulse modules in action
- **Status**: [ ]

### Day 2 (Tue) — Benchmarks

#### Task 8.3 — Accuracy benchmarks
- **Files to create**: `tests/benchmarks/test_truthfulqa.py`, `test_halubench.py`, `test_drift_detection.py`, `test_package_size.py`
- **Targets**: Hallucination F1 >= 0.70, drift TPR >= 90%, package < 50MB
- **Status**: [ ]

#### Task 8.4 — Performance benchmarks
- **SDK overhead**: at queue depths 0/100/1000/10000. Dashboard refresh latency.
- **Targets**: SDK < 5ms p99, dashboard < 5s
- **Status**: [ ]

### Day 3 (Wed) — PyPI + HuggingFace Deploy

#### Task 8.5 [CP] — PyPI packaging
- **Finalize**: version 0.1.0, classifiers, scripts entry, long description, dependency bounds
- **Acceptance**: `python -m build` -> valid wheel+sdist; `twine check dist/*` passes; fresh venv install works
- **Status**: [ ]

#### Task 8.6 — HuggingFace Spaces deployment
- **Create**: `dashboard/demo_data.py` (200 synthetic EvalRecords with realistic distributions)
- **Dashboard demo mode**: if `DEMO_MODE=true`, load synthetic data
- **Deploy**: to HuggingFace Spaces
- **Status**: [ ]

### Day 4 (Thu) — Contributing + Final Tests

#### Task 8.7 — Contributing guide + GitHub templates
- **Files**: `CONTRIBUTING.md`, `.github/ISSUE_TEMPLATE/bug_report.md`, `.github/ISSUE_TEMPLATE/feature_request.md`, `.github/PULL_REQUEST_TEMPLATE.md`
- **Status**: [ ]

#### Task 8.8 — Final integration test (golden path)
- **Files to create**: `tests/integration/test_final.py`
- **10-step verification**: init -> 100 diverse queries -> all modules score -> health scores -> alerts fire -> dashboard renders -> regression suite -> clean shutdown
- **Status**: [ ]

### Day 5 (Fri) — Launch

#### Task 8.9 — Changelog and release notes
- **Files to create**: `CHANGELOG.md` (v0.1.0 features, known limitations, v0.2.0 preview)
- **Status**: [ ]

#### Task 8.10 — Demo script + launch checklist
- **Files to create**: `examples/demo_script.py` (50 queries, live narration), `LAUNCH_CHECKLIST.md`
- **Launch checklist**: PyPI upload, HF Spaces deploy, GitHub release tag, README screenshots, social media (LinkedIn, Twitter/X, Reddit r/MachineLearning, HN Show HN, Dev.to)
- **Status**: [ ]

---

## Module Dependency Matrix <a id="module-dependency-matrix"></a>

| Module | Week | Depends On | Blocks |
|--------|------|-----------|--------|
| Quality (M4) | 2 | Config, Storage, Worker | Health Score, Dashboard Tab 4 |
| Drift (M2) | 3 | Config, Storage, Worker, EmbeddingService | Health Score, Dashboard Tab 3 |
| Hallucination (M1) | 4 | Config, Storage, Worker, GroqClient | Health Score, Dashboard Tab 2 |
| RAG (M3) | 5 | Config, Storage, Worker, EmbeddingService | Health Score, Dashboard Tab 4 |
| Regression (M5) | 6 | All modules, Golden Dataset | CI Workflow, Dashboard Tab 4 |

---

## Risk Register & Mitigations <a id="risk-register"></a>

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | SelfCheckGPT API changes since plan was written | Medium | High | Fall back to embedding-based consistency. Budget 1 extra day in Week 4. |
| 2 | Evidently AI API differs from documentation | Medium | Medium | Task 2.6 explicitly verifies API. Budget 0.5 extra day in Weeks 2 & 5. |
| 3 | Groq free tier rate limits hit during testing | Medium | Low | All Groq usage optional with fallback. Mocked in tests. Rate limiter in client. |
| 4 | sentence-transformers model download (~80MB) | Low | Low | Document first-run download in README. Cache in CI. |
| 5 | Package size exceeds 50MB target | Medium | Low | Make detoxify + sentence-transformers optional deps. Core stays small. |
| 6 | Hallucination false positives reducing trust | High | Medium | Confidence tiers (>0.5=High, 0.3-0.5=Suspected). Dashboard feedback button. |
| 7 | SQLite not suitable for high-volume multi-process | Low | Medium | WAL mode handles concurrent reads. Neon Postgres with 1 config line for high-volume. |
| 8 | ChromaDB memory growth over time | Low-Med | Low | Configurable retention (last 10K embeddings). Auto-prune. Persist to disk. |
| 9 | HuggingFace Spaces GPU queue wait | Medium | Low | CPU sufficient for MiniLM. Dashboard on CPU Spaces. GPU only for faster UMAP. |

---

## Target Metrics <a id="target-metrics"></a>

### Technical Metrics

| Metric | Target | Measured In |
|--------|--------|------------|
| Hallucination detection F1 on HaluBench | >= 0.70 | Week 8 benchmark |
| Drift detection true positive rate | >= 90% | Week 8 benchmark |
| RAG faithfulness Pearson correlation | >= 0.65 | Week 5 benchmark |
| SDK overhead per decorated call | < 5ms | Week 3 benchmark |
| Dashboard refresh latency | < 5 seconds | Week 7 test |
| Package size (excl. model downloads) | < 50MB | Week 8 benchmark |
| Time from pip install to first evaluation | < 30 minutes | Week 8 validation |

### Portfolio & Community Metrics

| Metric | Target | Timeline |
|--------|--------|---------|
| PyPI downloads | > 500 | First month |
| GitHub stars | > 150 | Within 4 weeks of launch |
| HuggingFace Spaces visits | > 300 unique | First 2 weeks |
| Dev.to / HackerNews launch post | Top 10 Show HN | Within 24 hours |
| LinkedIn demo impressions | > 8,000 | First week |

---

## Verification Checklist <a id="verification-checklist"></a>

### After Each Week

- [ ] `pytest tests/unit/ -v` — all unit tests pass
- [ ] `pytest tests/integration/ -v` — all integration tests pass
- [ ] `ruff check .` — no lint errors
- [ ] `ruff format --check .` — formatting correct
- [ ] `python dashboard/app.py` — dashboard renders without error
- [ ] `python examples/quickstart.py` — end-to-end flow works

### Final Verification (Week 8)

- [ ] `pip install evalpulse` in fresh venv works
- [ ] Track 100 LLM calls -> all 5 modules produce scores
- [ ] Health scores computed and in valid range (0-100)
- [ ] Alerts fire for intentionally bad responses
- [ ] Dashboard shows all 4 tabs with real data
- [ ] Regression suite runs on golden dataset
- [ ] Benchmarks: F1 >= 0.70, drift TPR >= 90%, SDK < 5ms, dashboard < 5s
- [ ] `python -m build` produces valid wheel + sdist
- [ ] `twine check dist/*` passes
- [ ] HuggingFace Spaces demo is live and functional

---

## Task Summary

| Week | Focus | Tasks | Critical Path | Tests |
|------|-------|-------|--------------|-------|
| 1 | Setup + SDK + SQLite + Dashboard | 12 | 6 (1.1, 1.2, 1.4, 1.6, 1.7, 1.9) | 3 |
| 2 | Module 4: Quality Scorer | 9 | 3 (2.1, 2.2, 2.4) | 3 |
| 3 | Module 2: Drift Detector | 9 | 3 (3.1, 3.2, 3.3) | 3 |
| 4 | Module 1: Hallucination + Health Score | 9 | 2 (4.1, 4.6) | 2 |
| 5 | Module 3: RAG Evaluator | 8 | 1 (5.1) | 3 |
| 6 | Module 5: Regression + CI | 8 | 3 (6.1, 6.2, 6.5) | 3 |
| 7 | Alerts + Polish + Deploy Prep | 10 | 1 (7.1) | 3 |
| 8 | Launch: PyPI + Benchmarks + Docs | 10 | 1 (8.5) | 3 |
| **Total** | | **75** | **20** | **23** |

---

> **Next step**: Begin Week 1, Day 1 — Task 1.1: Initialize git repository and project structure.
