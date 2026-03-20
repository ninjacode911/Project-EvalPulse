"""Tests for demo data generation."""

from dashboard.demo_data import generate_demo_records


class TestDemoData:
    """Test synthetic data generation."""

    def test_generates_correct_count(self):
        records = generate_demo_records(50)
        assert len(records) == 50

    def test_records_have_valid_scores(self):
        records = generate_demo_records(100)
        for r in records:
            assert 0 <= r.hallucination_score <= 1
            assert 0 <= r.sentiment_score <= 1
            assert 0 <= r.toxicity_score <= 1
            assert 0 <= r.health_score <= 100
            assert r.response_length > 0

    def test_rag_records_have_scores(self):
        records = generate_demo_records(100)
        rag_records = [r for r in records if r.faithfulness_score is not None]
        # ~70% should be RAG
        assert len(rag_records) > 40
        for r in rag_records:
            assert 0 <= r.faithfulness_score <= 1
            assert 0 <= r.groundedness_score <= 1

    def test_non_rag_records_have_none(self):
        records = generate_demo_records(100)
        non_rag = [r for r in records if r.faithfulness_score is None]
        assert len(non_rag) > 10
        for r in non_rag:
            assert r.context_relevance is None
            assert r.groundedness_score is None

    def test_deterministic_with_seed(self):
        r1 = generate_demo_records(10)
        r2 = generate_demo_records(10)
        assert r1[0].hallucination_score == r2[0].hallucination_score
