"""Tests for the RAG Quality Evaluator module."""

import pytest

from evalpulse.models import EvalEvent
from evalpulse.modules.rag_eval import RAGQualityModule


@pytest.fixture
def module():
    return RAGQualityModule()


class TestRAGQualityModule:
    """Test RAG quality evaluation."""

    def test_module_name(self, module):
        assert module.name == "rag_quality"

    def test_is_available(self):
        assert RAGQualityModule.is_available() is True

    def test_no_context_returns_none(self, module):
        """Non-RAG calls should return None for all RAG fields."""
        event = EvalEvent(
            query="What is Python?",
            response="Python is a programming language.",
            context=None,
        )
        result = module.evaluate_sync(event)
        assert result["faithfulness_score"] is None
        assert result["context_relevance"] is None
        assert result["answer_relevancy"] is None
        assert result["groundedness_score"] is None

    def test_empty_context_returns_none(self, module):
        """Empty context should be treated as non-RAG."""
        event = EvalEvent(
            query="What is Python?",
            response="Python is a programming language.",
            context="",
        )
        result = module.evaluate_sync(event)
        assert result["faithfulness_score"] is None

    def test_relevant_context_high_scores(self, module):
        """Relevant context + faithful response = high scores."""
        event = EvalEvent(
            query="What is machine learning?",
            context=(
                "Machine learning is a subset of artificial "
                "intelligence that enables systems to learn "
                "and improve from experience without being "
                "explicitly programmed."
            ),
            response=(
                "Machine learning is a branch of AI that "
                "allows systems to learn from experience "
                "without explicit programming."
            ),
        )
        result = module.evaluate_sync(event)
        assert result["faithfulness_score"] is not None
        assert result["faithfulness_score"] > 0.5
        assert result["context_relevance"] > 0.5
        assert result["answer_relevancy"] > 0.5
        assert result["groundedness_score"] > 0.5

    def test_irrelevant_context_lower_scores(self, module):
        """Irrelevant context should yield lower scores."""
        event = EvalEvent(
            query="What is machine learning?",
            context=(
                "The Eiffel Tower is a wrought-iron lattice "
                "tower in Paris, built in 1889. It is 330 "
                "meters tall and a famous landmark."
            ),
            response=(
                "Machine learning is a branch of AI that allows computers to learn from data."
            ),
        )
        result = module.evaluate_sync(event)
        assert result["context_relevance"] is not None
        # Context about Eiffel Tower is irrelevant to ML query
        assert result["context_relevance"] < 0.5

    def test_unfaithful_response_lower_faithfulness(self, module):
        """Response ignoring context should have lower faithfulness."""
        event = EvalEvent(
            query="What color is the sky?",
            context="The sky appears blue due to Rayleigh scattering.",
            response=(
                "The Amazon rainforest is home to millions "
                "of species and covers a vast area of South "
                "America."
            ),
        )
        result = module.evaluate_sync(event)
        assert result["faithfulness_score"] is not None
        assert result["faithfulness_score"] < 0.5

    def test_all_scores_in_range(self, module):
        """All scores should be in [0, 1] range."""
        event = EvalEvent(
            query="What is Python?",
            context="Python is a programming language.",
            response="Python is used for web development.",
        )
        result = module.evaluate_sync(event)
        for key in [
            "faithfulness_score",
            "context_relevance",
            "answer_relevancy",
            "groundedness_score",
        ]:
            val = result[key]
            assert val is not None
            assert 0.0 <= val <= 1.0, f"{key}={val} out of range"

    def test_groundedness_is_composite(self, module):
        """Groundedness should be a weighted average of other scores."""
        event = EvalEvent(
            query="What is deep learning?",
            context=(
                "Deep learning uses neural networks with multiple layers to learn representations."
            ),
            response=(
                "Deep learning is a technique using multi-layer "
                "neural networks for learning data representations."
            ),
        )
        result = module.evaluate_sync(event)
        f = result["faithfulness_score"]
        cr = result["context_relevance"]
        ar = result["answer_relevancy"]
        g = result["groundedness_score"]
        # Groundedness = 0.4*f + 0.3*cr + 0.3*ar
        expected = 0.4 * f + 0.3 * cr + 0.3 * ar
        assert abs(g - expected) < 0.02

    def test_empty_query_returns_none(self, module):
        """Empty query with context should return None."""
        event = EvalEvent(
            query="",
            context="Some context here.",
            response="Some response.",
        )
        result = module.evaluate_sync(event)
        assert result["faithfulness_score"] is None

    def test_empty_response_returns_none(self, module):
        """Empty response with context should return None."""
        event = EvalEvent(
            query="What is X?",
            context="X is something.",
            response="",
        )
        result = module.evaluate_sync(event)
        assert result["faithfulness_score"] is None
