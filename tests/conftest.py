"""Shared test fixtures for EvalPulse."""

import pytest
import yaml

from evalpulse.config import reset_config
from evalpulse.models import EvalEvent, EvalRecord
from evalpulse.storage import reset_storage


@pytest.fixture(autouse=True)
def clean_singletons():
    """Reset global singletons before each test."""
    reset_config()
    reset_storage()
    yield
    reset_config()
    reset_storage()


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary SQLite database path."""
    return str(tmp_path / "test_evalpulse.db")


@pytest.fixture
def tmp_config(tmp_path):
    """Create a temporary evalpulse.yml config file and return its path."""
    config_data = {
        "app_name": "test-app",
        "storage_backend": "sqlite",
        "sqlite_path": str(tmp_path / "test.db"),
        "modules": {
            "hallucination": True,
            "drift": True,
            "rag_quality": True,
            "response_quality": True,
            "regression": True,
        },
        "thresholds": {
            "hallucination": 0.3,
            "drift": 0.15,
            "rag_groundedness_min": 0.65,
            "toxicity": 0.05,
            "regression_fail_rate": 0.10,
        },
    }
    config_path = tmp_path / "evalpulse.yml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return str(config_path)


@pytest.fixture
def sample_eval_event():
    """Create a sample EvalEvent for testing."""
    return EvalEvent(
        app_name="test-app",
        query="What is Python?",
        context="Python is a programming language created by Guido van Rossum.",
        response="Python is a high-level programming language known for its simplicity.",
        model_name="test-model",
        latency_ms=150,
        tags=["test", "unit"],
    )


@pytest.fixture
def sample_eval_record():
    """Create a sample EvalRecord for testing."""
    return EvalRecord(
        app_name="test-app",
        query="What is Python?",
        context="Python is a programming language created by Guido van Rossum.",
        response="Python is a high-level programming language known for its simplicity.",
        model_name="test-model",
        latency_ms=150,
        tags=["test", "unit"],
        hallucination_score=0.1,
        hallucination_method="selfcheck",
        sentiment_score=0.7,
        toxicity_score=0.01,
        response_length=10,
        language_detected="en",
        is_denial=False,
        health_score=85,
    )


@pytest.fixture
def make_eval_records():
    """Factory fixture to create multiple EvalRecords."""

    def _make(n: int, **overrides) -> list:
        records = []
        for i in range(n):
            data = {
                "app_name": "test-app",
                "query": f"Test query {i}",
                "response": f"Test response {i} with some content here.",
                "model_name": "test-model",
                "latency_ms": 100 + i,
                "hallucination_score": 0.1 + (i % 5) * 0.1,
                "sentiment_score": 0.5 + (i % 3) * 0.15,
                "toxicity_score": 0.01,
                "response_length": 8,
                "health_score": 80 - (i % 20),
            }
            data.update(overrides)
            records.append(EvalRecord(**data))
        return records

    return _make
