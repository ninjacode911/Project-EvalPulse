# Contributing to EvalPulse

Thank you for your interest in contributing to EvalPulse!

## Development Setup

```bash
git clone https://github.com/ninjacode911/evalpulse
cd evalpulse
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# All tests
pytest tests/ -v
```

## Code Style

We use `ruff` for linting and formatting:

```bash
ruff check .
ruff format .
```

Configuration is in `pyproject.toml` (line-length=100, Python 3.11+).

## Adding a New Module

1. Create `evalpulse/modules/your_module.py`
2. Extend `EvalModule` from `evalpulse/modules/base.py`
3. Implement `name`, `evaluate()`, `is_available()`
4. Register in `evalpulse/modules/__init__.py`
5. Add fields to `EvalRecord` in `evalpulse/models.py`
6. Add tests in `tests/unit/test_your_module.py`

## Pull Request Process

1. Create a feature branch from `main`
2. Write tests for new functionality
3. Ensure all tests pass: `pytest tests/ -v`
4. Ensure lint passes: `ruff check .`
5. Submit PR with a clear description
