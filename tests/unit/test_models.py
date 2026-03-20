"""Tests for EvalPulse data models."""

from datetime import datetime

from evalpulse.models import EvalEvent, EvalRecord


class TestEvalEvent:
    """Test EvalEvent model."""

    def test_default_creation(self):
        """EvalEvent should create with sensible defaults."""
        event = EvalEvent()
        assert event.id is not None
        assert len(event.id) == 36  # UUID format
        assert event.app_name == "default"
        assert event.query == ""
        assert event.response == ""
        assert event.model_name == "unknown"
        assert event.latency_ms == 0
        assert event.tags == []
        assert event.context is None
        assert isinstance(event.timestamp, datetime)

    def test_custom_creation(self):
        """EvalEvent should accept custom values."""
        event = EvalEvent(
            app_name="test-app",
            query="hello",
            response="world",
            model_name="gpt-4",
            latency_ms=100,
            tags=["test"],
            context="some context",
        )
        assert event.app_name == "test-app"
        assert event.query == "hello"
        assert event.response == "world"
        assert event.context == "some context"
        assert event.tags == ["test"]

    def test_serialization_roundtrip(self):
        """EvalEvent should survive serialization/deserialization."""
        event = EvalEvent(app_name="test", query="q", response="r", tags=["a", "b"])
        data = event.model_dump()
        restored = EvalEvent.model_validate(data)
        assert restored.app_name == event.app_name
        assert restored.query == event.query
        assert restored.tags == event.tags


class TestEvalRecord:
    """Test EvalRecord model."""

    def test_default_creation(self):
        """EvalRecord should create with zeroed scores."""
        record = EvalRecord()
        assert record.hallucination_score == 0.0
        assert record.drift_score is None
        assert record.sentiment_score == 0.5
        assert record.toxicity_score == 0.0
        assert record.health_score == 0
        assert record.is_denial is False
        assert record.flagged_claims == []
        assert record.embedding_vector == []

    def test_optional_fields_accept_none(self):
        """Optional fields should accept None."""
        record = EvalRecord(
            drift_score=None,
            faithfulness_score=None,
            context_relevance=None,
            answer_relevancy=None,
            groundedness_score=None,
            context=None,
        )
        assert record.drift_score is None
        assert record.context is None

    def test_from_event(self):
        """EvalRecord.from_event should copy event fields and zero scores."""
        event = EvalEvent(
            app_name="test-app",
            query="hello",
            response="world is great",
            model_name="test-model",
            latency_ms=50,
            tags=["prod"],
        )
        record = EvalRecord.from_event(event)
        assert record.id == event.id
        assert record.app_name == "test-app"
        assert record.query == "hello"
        assert record.response == "world is great"
        assert record.model_name == "test-model"
        assert record.latency_ms == 50
        assert record.tags == ["prod"]
        assert record.response_length == 3  # "world is great" = 3 words
        assert record.hallucination_score == 0.0  # default

    def test_serialization_roundtrip(self):
        """EvalRecord should survive serialization/deserialization."""
        record = EvalRecord(
            app_name="test",
            query="q",
            response="r",
            hallucination_score=0.5,
            tags=["a"],
            flagged_claims=["claim1"],
            embedding_vector=[0.1, 0.2, 0.3],
        )
        data = record.model_dump()
        restored = EvalRecord.model_validate(data)
        assert restored.hallucination_score == 0.5
        assert restored.flagged_claims == ["claim1"]
        assert restored.embedding_vector == [0.1, 0.2, 0.3]

    def test_from_event_empty_response(self):
        """from_event should handle empty response."""
        event = EvalEvent(response="")
        record = EvalRecord.from_event(event)
        assert record.response_length == 0
