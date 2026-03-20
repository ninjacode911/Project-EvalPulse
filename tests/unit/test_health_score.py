"""Tests for health score computation."""

from evalpulse.health_score import compute_aggregate_health, compute_health_score
from evalpulse.models import EvalRecord


class TestHealthScore:
    """Test health score formula."""

    def test_perfect_record(self):
        """Perfect scores should give health score near 100."""
        record = EvalRecord(
            hallucination_score=0.0,
            drift_score=0.0,
            groundedness_score=1.0,
            sentiment_score=0.8,
            toxicity_score=0.0,
            is_denial=False,
        )
        score = compute_health_score(record)
        assert score >= 85

    def test_worst_record(self):
        """Worst scores should give health score near 0."""
        record = EvalRecord(
            hallucination_score=1.0,
            drift_score=1.0,
            groundedness_score=0.0,
            sentiment_score=0.0,
            toxicity_score=1.0,
            is_denial=True,
        )
        score = compute_health_score(record)
        assert score <= 20

    def test_none_drift_redistributes(self):
        """None drift score should redistribute weight."""
        record = EvalRecord(
            hallucination_score=0.0,
            drift_score=None,
            sentiment_score=0.8,
            toxicity_score=0.0,
        )
        score = compute_health_score(record)
        assert 70 <= score <= 100

    def test_none_rag_redistributes(self):
        """None RAG scores should redistribute weight."""
        record = EvalRecord(
            hallucination_score=0.1,
            drift_score=0.05,
            groundedness_score=None,
            faithfulness_score=None,
            sentiment_score=0.7,
            toxicity_score=0.02,
        )
        score = compute_health_score(record)
        assert 60 <= score <= 100

    def test_score_clamped_0_100(self):
        """Score should always be in 0-100 range."""
        record = EvalRecord(
            hallucination_score=0.0,
            sentiment_score=1.0,
            toxicity_score=0.0,
        )
        score = compute_health_score(record)
        assert 0 <= score <= 100

    def test_aggregate_health(self):
        """Aggregate should be the mean of individual scores."""
        records = [
            EvalRecord(health_score=80),
            EvalRecord(health_score=60),
            EvalRecord(health_score=90),
        ]
        agg = compute_aggregate_health(records)
        assert agg == 77  # (80 + 60 + 90) / 3

    def test_aggregate_empty(self):
        """Empty list should return neutral 50."""
        assert compute_aggregate_health([]) == 50

    def test_toxicity_penalty(self):
        """High toxicity should lower health score."""
        clean = EvalRecord(
            hallucination_score=0.1,
            toxicity_score=0.0,
            sentiment_score=0.7,
        )
        toxic = EvalRecord(
            hallucination_score=0.1,
            toxicity_score=0.9,
            sentiment_score=0.7,
        )
        assert compute_health_score(clean) > compute_health_score(toxic)

    def test_denial_penalty(self):
        """Denial should slightly lower health score."""
        normal = EvalRecord(
            hallucination_score=0.1,
            sentiment_score=0.7,
            toxicity_score=0.0,
            is_denial=False,
        )
        denied = EvalRecord(
            hallucination_score=0.1,
            sentiment_score=0.7,
            toxicity_score=0.0,
            is_denial=True,
        )
        assert compute_health_score(normal) >= compute_health_score(denied)
