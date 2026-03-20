# Week 1: Project Setup + SDK + SQLite Storage + Dashboard Shell

## Overview

Week 1 established the foundational architecture of EvalPulse — the scaffolding upon which all 5 evaluation modules, the dashboard, and the alert engine are built. By the end of this week, we have a fully installable Python package with a non-blocking SDK, persistent SQLite storage, and a live Gradio dashboard shell.

## What Was Built

### 1. Project Scaffold (`pyproject.toml`, package structure)

**What**: A modern Python package using `pyproject.toml` (PEP 621) with setuptools as the build backend. The package is installable via `pip install -e .` for development.

**Why**: Using `pyproject.toml` instead of `setup.py` follows 2026 Python packaging best practices. It centralizes all project metadata, dependencies, tool configuration (ruff, pytest), and build settings in a single file.

**Key decisions**:
- `requires-python = ">=3.11"` — uses modern Python features like `X | None` type syntax and `tomllib`
- All dependencies have version bounds (e.g., `pydantic>=2.0,<3.0`) to prevent breaking changes
- Dev dependencies are separated into `[project.optional-dependencies.dev]` so production installs stay lean
- Package discovery via `[tool.setuptools.packages.find]` with `include = ["evalpulse*"]` to only ship the core package

**Technical details**:
- 15 production dependencies, 4 dev dependencies
- CLI entry point registered: `evalpulse = "evalpulse.cli:main"`
- Ruff configured with line-length=100, targeting Python 3.11, with select rules E/F/I/W/UP

### 2. Configuration System (`evalpulse/config.py`)

**What**: A Pydantic-based configuration system that loads settings from `evalpulse.yml` files and environment variables, with sensible defaults for zero-config operation.

**Why**: EvalPulse must work out of the box with zero configuration (the "30-minute setup" promise), but also support customization for production deployments. The config system is the bridge between these two requirements.

**How it works**:
- `EvalPulseConfig` is a Pydantic `BaseModel` with nested models for modules, thresholds, and notifications
- `EvalPulseConfig.load(path)` searches for `evalpulse.yml` in the current directory, loads it with `yaml.safe_load()`, then applies environment variable overrides
- `get_config()` provides a singleton accessor (global `_config` variable with `reset_config()` for testing)
- Environment variables like `GROQ_API_KEY` and `EVALPULSE_APP_NAME` override file values

**Design pattern**: Singleton with lazy initialization. The config is loaded once on first access and cached. Tests can reset it via `reset_config()`.

**Validation**: Pydantic validates all fields at load time. Invalid values (e.g., `hallucination: "not_a_number"`) raise `ValidationError` immediately rather than failing silently at runtime.

### 3. Data Models (`evalpulse/models.py`)

**What**: Two Pydantic models that define the data contracts throughout the system:
- `EvalEvent`: Lightweight input captured by the SDK (query, context, response, metadata)
- `EvalRecord`: Full evaluation result with all module scores, stored in the database

**Why**: Separating the input (EvalEvent) from the output (EvalRecord) follows the Command/Result pattern. The SDK produces lightweight events quickly (~2ms), while the evaluation engine enriches them into full records asynchronously.

**Key fields in EvalRecord**:
- Module 1 (Hallucination): `hallucination_score` (0.0-1.0), `hallucination_method`, `flagged_claims`
- Module 2 (Drift): `embedding_vector` (384-dim), `drift_score`
- Module 3 (RAG): `faithfulness_score`, `context_relevance`, `answer_relevancy`, `groundedness_score`
- Module 4 (Quality): `sentiment_score`, `toxicity_score`, `response_length`, `language_detected`, `is_denial`
- Composite: `health_score` (0-100)

**`EvalRecord.from_event()`**: Factory method that converts an EvalEvent to an EvalRecord with zeroed scores. This is used by the worker when a module hasn't scored the event yet.

### 4. SQLite Storage Backend (`evalpulse/storage/`)

**What**: An abstract `StorageBackend` interface with a concrete `SQLiteStore` implementation. Provides CRUD operations, time-series queries, and concurrent write safety.

**Why**: SQLite was chosen as the default because it's built into Python's stdlib (zero dependencies), requires no server setup, and handles the typical single-developer use case perfectly. The abstract base class allows swapping to Postgres for production deployments.

**How it works**:
- **WAL mode** (`PRAGMA journal_mode=WAL`): Allows concurrent reads while writing, critical for the dashboard polling the DB while the worker writes
- **Thread-local connections** (`threading.local()`): Each thread gets its own SQLite connection, preventing the "SQLite objects created in a thread can only be used in that same thread" error
- **Mutex for writes** (`threading.Lock()`): Serializes write operations to prevent corruption
- **JSON serialization**: List fields (`tags`, `flagged_claims`, `embedding_vector`) are stored as JSON text since SQLite doesn't have array types
- **Indexes**: On `(app_name, timestamp)` for filtered queries, `(health_score)` for status queries, `(timestamp)` for time-series

**SQL injection prevention**: The `order_by` parameter is validated against a whitelist of column names before being interpolated into SQL. All user-supplied values use parameterized queries.

**Storage factory** (`get_storage()`): Singleton pattern that returns the correct backend based on config. Defaults to SQLite. Supports `reset_storage()` for testing.

### 5. SDK Layer (`evalpulse/sdk.py`)

**What**: The developer-facing API with three integration patterns:
- `@track` — sync decorator for any LLM function
- `@atrack` — async decorator for async LLM functions
- `EvalContext` — context manager for manual logging (especially RAG pipelines)

**Why**: The SDK must add minimal overhead to the developer's application. The <2ms target is achieved by only doing a queue push in the hot path — all evaluation happens asynchronously in the background.

**How `@track` works**:
1. Wraps the decorated function with `functools.wraps` (preserves `__name__`, `__doc__`)
2. Records `time.perf_counter()` before and after the function call
3. Extracts `query` from the first positional argument (or `query` kwarg)
4. Extracts `response` from the return value
5. Creates an `EvalEvent` with all captured data
6. Calls `_enqueue_event()` which does `queue.put_nowait()` — if the queue is full, the event is silently dropped (never blocks the user's app)

**Context propagation**: `contextvars.ContextVar` allows nested `EvalContext` blocks to propagate `app_name`, `model_name`, and `tags` to `@track` calls within their scope.

**Lifecycle management**:
- `init(config_path)`: Loads config, initializes storage, starts the background worker
- `shutdown(timeout)`: Signals the worker to stop, drains the queue, closes storage

### 6. Background Worker (`evalpulse/worker.py`)

**What**: `EvaluationWorker` runs in a daemon thread, continuously reading events from the queue, processing them through evaluation modules, and saving results to storage.

**Why**: Asynchronous processing is essential to the <2ms SDK overhead guarantee. The worker decouples the hot path (user's LLM call) from the expensive evaluation (embedding, API calls, scoring).

**How it works**:
- Runs in a `daemon=True` thread (auto-exits when the main program exits)
- Reads from `queue.Queue` with a 0.1s timeout (allows checking the stop event)
- **Batch writing**: Collects up to `batch_size` (default 10) events or waits up to `batch_timeout` (default 1.0s), whichever comes first, then writes the entire batch to storage in one call
- **Module dispatch**: Iterates over registered modules, calling `evaluate_sync()` on each, and merges results into the `EvalRecord`. (Week 1 has no modules registered — events are saved with zeroed scores.)
- **Graceful shutdown**: When `stop()` is called, sets a stop event, then drains all remaining events from the queue before exiting

**Error handling**: If a module fails, the error is logged and the record is saved with default scores. If storage fails, the error is logged but doesn't crash the worker.

### 7. Module Interface (`evalpulse/modules/base.py`)

**What**: `EvalModule` abstract base class defining the contract for all 5 evaluation modules.

**Why**: A common interface ensures all modules are interchangeable, independently testable, and can be registered/deregistered via configuration.

**Interface**:
- `name: str` — human-readable module name
- `evaluate(event: EvalEvent) -> dict` — async evaluation returning partial EvalRecord fields
- `evaluate_sync(event)` — synchronous wrapper (handles asyncio event loop edge cases)
- `is_available() -> bool` — checks if module dependencies are installed

### 8. Gradio Dashboard Shell (`dashboard/app.py`)

**What**: A Gradio Blocks application with 4 tabs (Overview, Hallucination Deep-Dive, Semantic Drift, RAG & Quality), each with placeholder content and empty chart widgets.

**Why**: Having the dashboard shell from Day 1 provides:
1. Immediate visual feedback as modules are implemented
2. A deployment target for HuggingFace Spaces
3. A validation surface for the storage layer (can query and display data)

**Dashboard structure**:
- Custom CSS with EvalPulse branding (purple gradient header)
- 4 metric cards on the Overview tab (Health Score, Hallucination Rate, Drift Status, Total Evaluations)
- Placeholder messages indicating when each feature will be available
- Footer with version info

### 9. CI Pipeline (`.github/workflows/ci.yml`)

**What**: GitHub Actions workflow that runs on every push/PR: ruff lint, ruff format check, and pytest unit tests on Python 3.11 and 3.12.

**Why**: Continuous integration catches regressions early. The dual Python version matrix ensures compatibility.

## Test Results

**51 tests total**, all passing:
- `test_config.py`: 10 tests (defaults, YAML loading, env vars, validation, singleton, edge cases)
- `test_models.py`: 5 tests (creation, serialization, from_event conversion)
- `test_sdk.py`: 11 tests (return values, query capture, latency, overhead <5ms, async, context manager)
- `test_sqlite_store.py`: 14 tests (CRUD, batch, pagination, JSON roundtrip, concurrent writes, time-series, SQL injection prevention)
- `test_worker.py`: 5 tests (start/stop, processing, batch writes, graceful shutdown, conversion)
- `test_smoke.py`: 2 tests (full end-to-end flow, dashboard creation)

**Key performance test**: `test_overhead_is_minimal` verifies that the `@track` decorator adds less than 5ms overhead per call by comparing 500 decorated vs undecorated function calls.

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| `queue.Queue` over `asyncio.Queue` | The SDK must work in sync Python code (most LLM apps). `queue.Queue` is thread-safe and doesn't require an event loop. |
| Daemon thread for worker | Auto-exits when the main program exits, preventing zombie processes. |
| Batch writes (10 events or 1s) | Reduces SQLite I/O by ~10x compared to per-event writes. |
| JSON for list fields in SQLite | SQLite has no array type. JSON text is simple, human-readable, and performs well for reads. |
| `functools.wraps` on decorators | Preserves the original function's `__name__`, `__doc__`, and signature for debugging and docs. |
| Singleton pattern for config/storage | Avoids passing config objects through every function call. Reset functions enable testing. |

## Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `pyproject.toml` | Package metadata, deps, tool config | ~60 |
| `evalpulse/__init__.py` | Public API exports | ~8 |
| `evalpulse/config.py` | YAML config loader + Pydantic validation | ~100 |
| `evalpulse/models.py` | EvalEvent + EvalRecord data models | ~90 |
| `evalpulse/sdk.py` | @track, @atrack, EvalContext, init/shutdown | ~200 |
| `evalpulse/worker.py` | Background EvaluationWorker | ~140 |
| `evalpulse/modules/base.py` | EvalModule ABC | ~45 |
| `evalpulse/storage/base.py` | StorageBackend ABC | ~55 |
| `evalpulse/storage/sqlite_store.py` | SQLite implementation (WAL, thread-safe) | ~280 |
| `dashboard/app.py` | Gradio 4-tab dashboard shell | ~130 |
| `dashboard/charts.py` | Plotly chart builder stubs | ~100 |
| `.github/workflows/ci.yml` | CI lint + test pipeline | ~25 |
| `tests/` (7 files) | 51 tests covering all components | ~500 |
