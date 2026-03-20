# Changelog

All notable changes to EvalPulse are documented in this file.

## v0.1.0 (March 2026) — Initial Release

### Core Platform

- **SDK Layer**: Non-blocking instrumentation with ~2ms overhead
  - `@track` decorator for sync functions — captures query, response, latency, model name
  - `@atrack` async decorator for FastAPI/async applications
  - `EvalContext` context manager for RAG pipelines (query + context + response)
  - Bounded asyncio.Queue (10,000 events) with overflow warning
  - `init()` / `shutdown()` lifecycle management with singleton cleanup

- **Background Worker**: Daemon thread with batch processing
  - Configurable batch size (default: 10) and timeout (default: 1s)
  - ThreadPoolExecutor dispatches to evaluation modules in parallel
  - Graceful shutdown drains remaining events before stopping
  - Automatic health score computation after all modules run

### Evaluation Modules

- **Module 1 — Hallucination Scorer**
  - Embedding-based consistency scoring (cosine similarity between response and context)
  - Optional Groq LLM-as-judge with structured XML-delimited prompts (prompt injection resistant)
  - Per-claim extraction and flagging for sentences with low context support
  - Configurable scoring weights (0.6 embedding + 0.4 LLM-judge when both available)
  - Graceful fallback: works without Groq API key using embedding-only scoring

- **Module 2 — Semantic Drift Detector**
  - sentence-transformers (all-MiniLM-L6-v2, CPU-only) embeds every response
  - ChromaDB persistent vector store with collection-per-app isolation
  - Cosine distance between current embedding and rolling baseline centroid
  - Requires 10+ responses before scoring begins (returns None until then)
  - Sanitized collection names for ChromaDB compatibility

- **Module 3 — RAG Quality Evaluator**
  - Context Relevance: cosine similarity between query and retrieved context embeddings
  - Faithfulness: cosine similarity between response and context + sentence-level verification
  - Answer Relevancy: cosine similarity between query and response
  - Groundedness: weighted composite (40% faithfulness + 30% context relevance + 30% answer relevancy)
  - Automatically skips non-RAG calls (no context provided)
  - Input truncation at 10,000 characters to prevent OOM

- **Module 4 — Response Quality Scorer**
  - Sentiment analysis via NLTK VADER (normalized 0.0-1.0)
  - Toxicity scoring via detoxify local model (thread-safe lazy loading)
  - Language detection via langdetect
  - Denial/refusal detection via regex patterns ("I cannot", "As an AI", etc.)
  - Response length (word count) tracking

- **Module 5 — Prompt Regression Tester**
  - Golden dataset schema (JSON) with per-example thresholds
  - Batch regression runner evaluates LLM function against golden examples
  - Pass/fail per example with violation details
  - CLI: `evalpulse regression run --dataset path/to/golden.json`

### Health Score

- Composite formula: `(1-halluc)*0.35 + (1-drift)*0.25 + groundedness*0.20 + quality*0.15 + regression*0.05`
- None-handling with proportional weight redistribution
- Clamped to 0-100 integer range
- Pydantic `validate_assignment=True` catches out-of-range scores at write time

### Alert Engine

- Configurable thresholds per metric via `evalpulse.yml`
- Cooldown-based deduplication (default: 5 minutes) — persists across batches
- Severity classification: warning (slightly over threshold) vs critical (>0.2 over)
- Thread-safe alert persistence via SQLiteStore methods (no lock bypass)
- Notification dispatch: Email (SMTP) + Slack webhook (SSRF-protected URL validation)

### Storage

- SQLite with WAL mode, busy timeout, thread-local connections
- All connections tracked and closed on shutdown (no resource leaks)
- Parameterized queries throughout (SQL injection safe)
- Column names validated against allowlists before interpolation
- Alert table with separate save/query methods

### Dashboard

- 4-tab Gradio interface with dark "mission control" theme
  - **Overview**: Health gauge, 4 KPI cards, health trend (auto-scaled y-axis), alert table
  - **Hallucination**: Score timeline, distribution histogram, per-model comparison, flagged responses
  - **Semantic Drift**: Drift timeline with threshold, embedding space scatter
  - **RAG & Quality**: Quality metrics, RAG radar chart, language distribution, CSV export
- Full-width responsive layout with JetBrains Mono + Outfit typography
- Refresh buttons per tab + auto-load on startup

### CLI

- `evalpulse init` — creates `evalpulse.yml` from template
- `evalpulse regression run --dataset PATH` — runs regression suite
- `evalpulse dashboard` — launches Gradio dashboard

### CI/CD

- GitHub Actions: lint (ruff) + unit tests on push/PR for Python 3.11 and 3.12
- Regression test workflow (manual/scheduled triggers)

### Deployment

- HuggingFace Spaces: self-contained demo with 200 synthetic records
- Live at: https://huggingface.co/spaces/NinjainPJs/EvalPulse

### Security

- Prompt injection protection: XML-delimited content with separate system/user message roles
- Input validation: Pydantic `Field(ge=0.0, le=1.0)` on all scores, `max_length` on strings
- `validate_assignment=True` on EvalRecord catches out-of-range values at write time
- Worker clamps float scores before `setattr` to prevent floating-point edge cases
- `SecretStr` for API keys — prevents accidental logging/serialization
- SSRF protection: Slack webhook URLs validated against `https://hooks.slack.com` domain
- SQL injection prevention: parameterized queries + allowlist validation
- YAML safety: `yaml.safe_load()` exclusively
- Thread-safe singleton initialization with double-check locking (detoxify model)
- ChromaDB collection name sanitization (alphanumeric only)
- Bounded event queue with overflow logging

### Known Limitations

- SelfCheckGPT library integration not yet wired (using embedding consistency instead)
- UMAP 2D projection available in charts.py but not yet displayed in dashboard
- Postgres backend (`postgres_store.py`) not yet implemented — config will show helpful error
- Sliding window for drift baseline uses full history (auto-pruning not yet implemented)
- RTX 5070 (sm_120) and other unsupported CUDA GPUs — forced CPU mode for embeddings

### Test Coverage

- 150 unit and integration tests
- Config, models, storage, SDK, worker, all 5 modules, alerts, notifications, CLI, demo data, health score
- Concurrent write tests, edge case tests, integration flow tests

### Dependencies

- Python >= 3.11
- Core: pydantic>=2.0, pyyaml, gradio>=4.0, plotly>=5.0, evidently, sentence-transformers, chromadb, groq, detoxify, langdetect, httpx, numpy, pandas, scipy, umap-learn
- Dev: pytest, pytest-asyncio, pytest-timeout, ruff
- Optional: psycopg2-binary (Postgres backend)
