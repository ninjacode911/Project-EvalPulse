# Week 8: Launch — Examples, Demo Data, Docs

## Overview

Week 8 prepared EvalPulse for launch with example applications, demo data for the HuggingFace Spaces dashboard, contribution guidelines, a changelog, and project documentation. This week focused on making the project usable, demoable, and contributor-friendly.

## What Was Built

### 1. Demo Data Generator (`dashboard/demo_data.py`)

**What**: Generates 200 synthetic EvalRecords with realistic score distributions for the HuggingFace Spaces demo dashboard.

**Why**: The HuggingFace Spaces deployment needs data to display immediately — users visiting the demo should see a populated dashboard with realistic patterns, not an empty screen. Synthetic data also serves as a visual test: if the dashboard renders correctly with 200 records spanning various scenarios, it will handle real production data.

**Simulation model**: The generator models an LLM application that experiences realistic quality patterns over time:

- **Baseline health**: Records start with low hallucination scores (~0.05-0.15) and low drift (~0.02-0.05), representing a well-functioning system.
- **Gradual drift**: Drift scores increase linearly with record index, simulating model behavior shifting over time as input distributions change. By the last records, drift is noticeably elevated.
- **Hallucination spikes**: Random spikes in hallucination scores are injected to simulate intermittent failure modes (e.g., adversarial inputs, edge cases that confuse the model).
- **RAG distribution**: Approximately 70% of calls include context and RAG scores (context relevance, faithfulness, answer relevancy, groundedness). The remaining 30% are non-RAG calls with None for RAG metrics. This mirrors a typical production deployment where most but not all queries hit the retrieval pipeline.
- **Denial rate**: Approximately 5% of calls are flagged as denials (the LLM refused to answer), simulating safety guardrails triggering.
- **Model variety**: Records are distributed across 3 model names (e.g., "gpt-4", "gpt-3.5-turbo", "claude-3") to test the dashboard's model-filtering capabilities.
- **Query diversity**: 10 different query types/categories are rotated through to test per-category analytics.

**Deterministic output**: Uses `random.seed(42)` so that every run produces identical data. This is important for: (1) reproducible dashboard screenshots for documentation, (2) consistent CI test expectations, (3) predictable HuggingFace Spaces behavior.

**Score distributions**: Scores are generated using clipped normal distributions centered around realistic values. For example, hallucination scores use a mean of 0.1 with standard deviation 0.08, clipped to [0.0, 1.0]. This produces a natural-looking distribution with most values near the mean and occasional outliers.

### 2. Example Applications

Three example applications demonstrate different EvalPulse integration patterns. Each is a self-contained runnable script with its own README explaining setup and expected output.

#### Groq Chatbot (`examples/groq_chatbot/main.py`)

**What**: A real LLM application using the Groq API with EvalPulse's `@track` decorator for monitoring.

**How it works**:
1. Initializes EvalPulse with default configuration.
2. Defines a `chat()` function decorated with `@track`.
3. The function calls Groq's API (Llama model) with the user's message.
4. `@track` intercepts the call, captures input/output, runs evaluation modules, and stores results.
5. Prints the response and evaluation scores.

**Why Groq**: Groq provides free API access with fast inference, making it the lowest-friction way for users to try EvalPulse with a real LLM. No paid API key needed.

**Requirements**: `GROQ_API_KEY` environment variable (free from console.groq.com).

#### RAG Pipeline (`examples/rag_pipeline/main.py`)

**What**: An in-memory RAG pipeline demonstrating EvalPulse's RAG-specific evaluation.

**How it works**:
1. Defines a simple knowledge base as a list of text passages.
2. Implements a keyword-based retrieval function that finds relevant passages.
3. Combines retrieved context with the query into a prompt.
4. Uses `EvalContext` to pass the retrieved context to EvalPulse for RAG metric evaluation.
5. The `@track` decorator captures the query, response, and context, then evaluates context relevance, faithfulness, answer relevancy, and groundedness.

**Why in-memory retrieval**: The example is meant to demonstrate EvalPulse integration, not build a production RAG system. An in-memory keyword search avoids dependencies on vector databases (Pinecone, Weaviate, ChromaDB) while still producing meaningful context relevance and faithfulness scores.

**What it demonstrates**: How to use `EvalContext` to pass RAG-specific metadata (retrieved context) to EvalPulse's evaluation pipeline, and how RAG metrics differ from standard metrics.

#### FastAPI App (`examples/fastapi_app/main.py`)

**What**: An async FastAPI application using EvalPulse's `@atrack` decorator for non-blocking evaluation.

**How it works**:
1. Creates a FastAPI application with startup/shutdown lifecycle hooks.
2. On startup, initializes EvalPulse (loads config, starts background worker).
3. Defines an async endpoint `/chat` decorated with `@atrack`.
4. The endpoint calls a simulated async LLM function.
5. `@atrack` runs evaluation asynchronously, ensuring the HTTP response is not delayed by evaluation processing.
6. On shutdown, flushes pending evaluations and stops the background worker.

**Why async**: FastAPI is an async framework. Using the synchronous `@track` decorator would block the event loop during evaluation, degrading request throughput. `@atrack` runs evaluation modules in the background worker queue without blocking the response.

**What it demonstrates**: Proper async integration pattern including lifecycle management (startup initialization, shutdown flush), the `@atrack` decorator, and non-blocking evaluation in a web server context.

### 3. Contributing Guide (`CONTRIBUTING.md`)

**Contents**:

- **Development setup**: Clone, create virtualenv, install in editable mode with dev dependencies (`pip install -e ".[dev]"`).
- **Testing commands**: `pytest tests/unit/` for unit tests, `pytest --cov=evalpulse` for coverage, `pytest -x` for fail-fast during development.
- **Code style**: Enforced with ruff. Run `ruff check evalpulse/` before committing. Configuration in `pyproject.toml`.
- **Module creation guide**: A 6-step process for adding a new evaluation module:
  1. Create the module file in `evalpulse/modules/`.
  2. Implement the `BaseModule` interface (`name`, `description`, `version`, `evaluate(record)` method).
  3. Register the module in the module registry.
  4. Add configuration options to the config schema.
  5. Write unit tests in `tests/unit/`.
  6. Update the dashboard to display the new metrics.
- **PR process**: Fork, branch from main, write tests, ensure all tests pass, submit PR with description of changes.

**Why a module creation guide**: EvalPulse's value grows with each evaluation module added. Making it easy for contributors to add modules (clear interface, step-by-step guide, existing modules as examples) directly increases the project's scope and utility.

### 4. Changelog (`CHANGELOG.md`)

**v0.1.0 release notes** documenting all features built across 8 weeks:

- Core SDK with `@track` and `@atrack` decorators
- 5 evaluation modules (hallucination, toxicity, RAG quality, language detection, prompt regression)
- Drift monitoring with embedding-based detection
- Background worker with SQLite storage
- Gradio dashboard with time series, histograms, and per-model analytics
- Alert engine with cooldown deduplication
- Email and Slack notifications
- CLI with init, regression, and dashboard commands
- 3 example applications

**Known limitations** (transparent about what is not yet production-ready):
- SelfCheckGPT not yet integrated (hallucination module uses embedding similarity, not NLI)
- UMAP visualization is a placeholder (dimensionality reduction for embedding visualization not yet implemented)
- PostgreSQL backend not fully tested (SQLite is the primary storage; Postgres adapter exists but lacks production testing)
- Sliding window for drift detection not yet implemented (currently compares against full baseline, not a rolling window)

**Why document known limitations**: For interview discussions, acknowledging limitations shows engineering maturity. For contributors, it provides a roadmap of what to work on next. For users, it sets honest expectations.

### 5. Quick Start Example (`examples/quickstart.py`)

**What**: A minimal 10-line example demonstrating the simplest possible EvalPulse integration.

**How it works**:
1. Import EvalPulse and the `@track` decorator.
2. Initialize EvalPulse with defaults.
3. Define a simulated LLM function (returns a hardcoded response) decorated with `@track`.
4. Call the function.
5. Print the response and evaluation results.

**Why a separate quickstart**: The three full examples (Groq, RAG, FastAPI) each demonstrate specific integration patterns but require context to understand. The quickstart is the "hello world" — the absolute minimum code to get EvalPulse running. It is the first thing a new user should see.

## Test Results

5 unit tests for demo data:
- **Correct count**: Generator produces exactly 200 records.
- **Valid score ranges**: All numeric scores are between 0.0 and 1.0.
- **RAG/non-RAG partitioning**: Approximately 70% of records have RAG scores (not None), approximately 30% have None for RAG metrics.
- **Deterministic with seed**: Two runs with the same seed produce identical records.
- **Model distribution**: Records are distributed across all 3 model names.

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| 200 synthetic records | Enough to populate all dashboard visualizations (time series, histograms, tables) without being slow to generate |
| `random.seed(42)` determinism | Reproducible output for screenshots, CI, and consistent demo behavior |
| Three distinct example apps | Each demonstrates a different integration pattern (basic, RAG, async); together they cover the most common deployment scenarios |
| In-memory RAG retrieval | Avoids vector DB dependency in the example; demonstrates EvalContext usage without infrastructure setup |
| Groq as the real API example | Free API access lowers the barrier to trying EvalPulse with a real LLM |
| Separate quickstart from full examples | Different audiences: quickstart for "show me the minimum", full examples for "show me real patterns" |
| Known limitations in changelog | Engineering honesty; provides contributor roadmap; sets user expectations |

## Files Created

| File | Purpose |
|------|---------|
| `dashboard/demo_data.py` | Synthetic data generator producing 200 realistic EvalRecords |
| `examples/quickstart.py` | Minimal 10-line quick start example |
| `examples/groq_chatbot/main.py` | Groq chatbot with `@track` decorator |
| `examples/groq_chatbot/README.md` | Setup and usage instructions for Groq example |
| `examples/rag_pipeline/main.py` | In-memory RAG pipeline with `EvalContext` |
| `examples/rag_pipeline/README.md` | Setup and usage instructions for RAG example |
| `examples/fastapi_app/main.py` | Async FastAPI application with `@atrack` decorator |
| `examples/fastapi_app/README.md` | Setup and usage instructions for FastAPI example |
| `CONTRIBUTING.md` | Contribution guide with development setup, testing, code style, module creation guide, PR process |
| `CHANGELOG.md` | v0.1.0 release notes with all features and known limitations |
| `tests/unit/test_demo_data.py` | 5 unit tests for demo data generator |

## Interview Talking Points

- **Why synthetic demo data instead of anonymized real data?** Synthetic data has no privacy concerns, produces consistent results across deployments, and can be tuned to demonstrate specific patterns (drift, spikes, RAG vs non-RAG mix). Real data would need anonymization, might not exhibit interesting patterns, and would change with every deployment.

- **How would you structure examples for a production SDK?** The three examples follow a pattern: each has a `main.py` (runnable entry point), a `README.md` (setup instructions), and optionally a `requirements.txt` (example-specific dependencies). For a production SDK, you would add: automated testing of examples in CI (ensure they still run), version-pinned dependencies, and a gallery page in the docs.

- **What would v0.2.0 look like?** Based on the known limitations: (1) SelfCheckGPT integration for NLI-based hallucination detection, (2) sliding window drift detection for time-aware baselines, (3) PostgreSQL production testing and migration tooling, (4) UMAP embedding visualization in the dashboard, (5) webhook notification channel for generic integrations.
