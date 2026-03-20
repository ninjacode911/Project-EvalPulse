# Contributing to EvalPulse

Thank you for your interest in contributing to EvalPulse! This guide covers everything you need to get started.

## Development Setup

### Prerequisites

- Python 3.11 or 3.12
- Git

### Clone and Install

```bash
git clone https://github.com/ninjacode911/Project-EvalPulse.git
cd Project-EvalPulse
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

### Configuration

```bash
# Copy the example config
cp evalpulse.yml.example evalpulse.yml

# Optional: Set Groq API key for LLM-as-judge hallucination scoring
export GROQ_API_KEY=gsk_your_key_here
```

## Running Tests

```bash
# All 150 tests
pytest tests/ -v --timeout=120

# Unit tests only (fast, no external deps)
pytest tests/unit/ -v

# Integration tests (requires model downloads on first run)
pytest tests/integration/ -v

# Specific module
pytest tests/unit/test_hallucination.py -v
```

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures (tmp_db, sample records, etc.)
├── unit/
│   ├── test_config.py       # Config loading, YAML parsing, env overrides
│   ├── test_models.py       # EvalEvent/EvalRecord validation, serialization
│   ├── test_sqlite_store.py # CRUD, concurrent writes, time-series queries
│   ├── test_sdk.py          # @track overhead, @atrack, EvalContext
│   ├── test_worker.py       # Batch processing, graceful shutdown
│   ├── test_quality.py      # Sentiment, toxicity, language, denial
│   ├── test_drift.py        # Embeddings, ChromaDB, drift scoring
│   ├── test_hallucination.py# Embedding consistency, claim extraction
│   ├── test_rag_eval.py     # Faithfulness, context relevance, groundedness
│   ├── test_regression.py   # Golden dataset, pass/fail logic
│   ├── test_health_score.py # Composite formula, None redistribution
│   ├── test_alerts.py       # Thresholds, cooldown, severity
│   ├── test_notifications.py# Email format, Slack URL validation
│   ├── test_cli.py          # CLI commands
│   └── test_demo_data.py    # Synthetic data generation
├── integration/
│   ├── test_smoke.py        # End-to-end: init -> track -> verify storage
│   └── test_quality_flow.py # Full quality module pipeline
└── benchmarks/
    └── (benchmark scripts)
```

## Code Style

We use `ruff` for linting and formatting:

```bash
# Check for errors
ruff check .

# Auto-fix
ruff check --fix .

# Format
ruff format .
```

Configuration is in `pyproject.toml`:
- Line length: 100
- Target: Python 3.11
- Rules: E, F, I, W, UP
- `dashboard/app.py` is exempt from E501 (CSS strings exceed line length)

## Architecture Overview

```
SDK (@track) → Queue → Worker Thread → [Modules in parallel] → EvalRecord → SQLite
                                                                    ↓
                                                          Health Score → Alerts → Notifications
                                                                    ↓
                                                          Dashboard (Gradio) reads from SQLite
```

### Key Design Decisions

1. **Non-blocking SDK**: `@track` adds ~2ms (queue push only). All evaluation is background.
2. **Module interface**: Every module implements `EvalModule.evaluate_sync(event) -> dict`. The worker merges results.
3. **Pydantic v2 models**: `EvalRecord` has `validate_assignment=True` — catches invalid scores at write time.
4. **Thread-safe storage**: SQLite WAL mode + explicit locking. All connections tracked for cleanup.
5. **Security-first**: XML-delimited prompts, parameterized SQL, SecretStr for API keys, SSRF validation.

## Adding a New Evaluation Module

1. **Create the module** at `evalpulse/modules/your_module.py`:

```python
from evalpulse.modules.base import EvalModule
from evalpulse.models import EvalEvent

class YourModule(EvalModule):
    @property
    def name(self) -> str:
        return "your_module"

    @classmethod
    def is_available(cls) -> bool:
        # Check if dependencies are installed
        try:
            import your_dependency
            return True
        except ImportError:
            return False

    async def evaluate(self, event: EvalEvent) -> dict:
        return self.evaluate_sync(event)

    def evaluate_sync(self, event: EvalEvent) -> dict:
        # Your scoring logic here
        return {"your_score_field": 0.5}
```

2. **Add fields** to `EvalRecord` in `evalpulse/models.py`:

```python
your_score_field: float = Field(default=0.0, ge=0.0, le=1.0)
```

3. **Register** in `evalpulse/modules/__init__.py`:

```python
from evalpulse.modules.your_module import YourModule
if YourModule.is_available():
    modules.append(YourModule(config=config))
```

4. **Add SQLite column** in `evalpulse/storage/sqlite_store.py` (both CREATE TABLE and INSERT)

5. **Write tests** at `tests/unit/test_your_module.py`

6. **Update health score** formula in `evalpulse/health_score.py` if the module contributes to overall health

## Pull Request Process

1. Create a feature branch from `main`
2. Write tests for new functionality
3. Ensure all tests pass: `pytest tests/ -v --timeout=120`
4. Ensure lint passes: `ruff check .`
5. Update CHANGELOG.md with your changes
6. Submit PR with a clear description of what and why

## Security Guidelines

When contributing, ensure:

- **No hardcoded secrets** — use environment variables or `SecretStr`
- **Parameterized SQL** — never interpolate user input into queries
- **Input validation** — all score fields must have `ge=0.0, le=1.0`
- **LLM prompts** — wrap user content in XML tags, use separate system/user roles
- **External URLs** — validate against expected domains before making requests
- **Thread safety** — use locks for shared mutable state, lazy-load with double-check pattern

## Project Structure

```
evalpulse/           # Core Python package
├── sdk.py           # @track, @atrack, EvalContext, init(), shutdown()
├── worker.py        # Background EvaluationWorker
├── models.py        # EvalEvent, EvalRecord (Pydantic v2)
├── config.py        # YAML config with env var overrides
├── health_score.py  # Composite health score formula
├── alerts.py        # Threshold checking + persistence
├── notifications.py # Email + Slack dispatch
├── cli.py           # CLI entry points
├── modules/         # 5 evaluation modules + shared services
└── storage/         # SQLite backend (+ future Postgres)

dashboard/           # Gradio dashboard (separate from package)
├── app.py           # 4-tab dark-themed interface
├── charts.py        # Plotly chart builders
└── demo_data.py     # Synthetic data for demos

hf_space/            # HuggingFace Spaces deployment (self-contained)
└── app.py           # Standalone demo app (no evalpulse dependency)
```

## Questions?

Open an issue at https://github.com/ninjacode911/Project-EvalPulse/issues
