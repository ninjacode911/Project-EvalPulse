"""Tests for the Hallucination Scorer module."""

import pytest

from evalpulse.models import EvalEvent
from evalpulse.modules.hallucination import HallucinationModule


@pytest.fixture
def module():
    return HallucinationModule()


class TestHallucinationModule:
    """Test hallucination detection."""

    def test_module_name(self, module):
        assert module.name == "hallucination"

    def test_is_available(self):
        assert HallucinationModule.is_available() is True

    def test_empty_response(self, module):
        event = EvalEvent(response="")
        result = module.evaluate_sync(event)
        assert result["hallucination_score"] == 0.0
        assert result["hallucination_method"] == "none"
        assert result["flagged_claims"] == []

    def test_grounded_response_low_score(self, module):
        """Response closely matching context should have low hallucination."""
        event = EvalEvent(
            query="What is Python?",
            context=(
                "Python is a high-level programming language created by Guido van Rossum in 1991."
            ),
            response=(
                "Python is a high-level programming language that was created by Guido van Rossum."
            ),
        )
        result = module.evaluate_sync(event)
        assert result["hallucination_score"] < 0.4
        assert result["hallucination_method"] == "embedding"

    def test_unrelated_response_higher_score(self, module):
        """Response unrelated to context should have higher hallucination score."""
        event = EvalEvent(
            query="What is Python?",
            context=("Python is a high-level programming language created by Guido van Rossum."),
            response=(
                "The Amazon rainforest covers approximately "
                "5.5 million square kilometers and is home to "
                "diverse wildlife."
            ),
        )
        result = module.evaluate_sync(event)
        assert result["hallucination_score"] > 0.2

    def test_no_context_uses_query(self, module):
        """Without context, should compare response to query."""
        event = EvalEvent(
            query="What is the capital of France?",
            response="Paris is the capital and largest city of France.",
        )
        result = module.evaluate_sync(event)
        assert result["hallucination_method"] == "embedding"
        # Should be reasonably low since response matches query topic
        assert result["hallucination_score"] < 0.5

    def test_flagged_claims_extraction(self, module):
        """Should extract potentially fabricated claims."""
        event = EvalEvent(
            query="Tell me about Python",
            context="Python is a programming language.",
            response=(
                "Python was created by James Gosling in 2005. It has 500 million users worldwide."
            ),
        )
        result = module.evaluate_sync(event)
        # Should flag claims with numbers/names not in context
        assert isinstance(result["flagged_claims"], list)

    def test_all_fields_returned(self, module):
        event = EvalEvent(query="test", response="test response with enough words to process.")
        result = module.evaluate_sync(event)
        assert "hallucination_score" in result
        assert "hallucination_method" in result
        assert "flagged_claims" in result
        assert 0.0 <= result["hallucination_score"] <= 1.0

    def test_without_groq_uses_embedding(self, module):
        """Without Groq API key, should use embedding method."""
        event = EvalEvent(
            query="What is ML?",
            context="Machine learning is a subset of artificial intelligence.",
            response="Machine learning is part of AI that focuses on learning from data.",
        )
        result = module.evaluate_sync(event)
        assert result["hallucination_method"] in ("embedding", "none")


class TestGroqClient:
    """Test Groq API client."""

    def test_unavailable_without_key(self):
        from evalpulse.modules.groq_client import GroqClient

        client = GroqClient(api_key=None)
        assert client.is_available() is False

    def test_available_with_key(self):
        from evalpulse.modules.groq_client import GroqClient

        client = GroqClient(api_key="test-key")
        assert client.is_available() is True

    def test_rate_limiter(self):
        from evalpulse.modules.groq_client import TokenBucketRateLimiter

        limiter = TokenBucketRateLimiter(rate=10.0, capacity=3.0)
        # Should succeed for first 3
        assert limiter.acquire(timeout=0.1) is True
        assert limiter.acquire(timeout=0.1) is True
        assert limiter.acquire(timeout=0.1) is True
        # 4th should fail with short timeout
        assert limiter.acquire(timeout=0.1) is False
