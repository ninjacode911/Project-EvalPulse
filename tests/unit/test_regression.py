"""Tests for the Prompt Regression Tester module."""

import pytest

from evalpulse.modules.golden_dataset import (
    GoldenDataset,
    GoldenExample,
    load_golden_dataset,
    save_golden_dataset,
)
from evalpulse.modules.regression import (
    PromptRegressionModule,
    RegressionResult,
)


class TestGoldenDataset:
    """Test golden dataset loading and saving."""

    def test_create_golden_example(self):
        ex = GoldenExample(
            query="What is Python?",
            expected_response="A programming language.",
        )
        assert ex.query == "What is Python?"
        assert ex.max_hallucination == 0.3
        assert ex.max_toxicity == 0.1

    def test_create_dataset(self):
        dataset = GoldenDataset(
            name="test",
            examples=[
                GoldenExample(query="q1", expected_response="r1"),
                GoldenExample(query="q2", expected_response="r2"),
            ],
        )
        assert len(dataset.examples) == 2
        assert dataset.name == "test"

    def test_save_and_load(self, tmp_path):
        dataset = GoldenDataset(
            name="save-test",
            version="2.0",
            examples=[
                GoldenExample(
                    query="q1",
                    expected_response="r1",
                    max_hallucination=0.5,
                ),
            ],
        )
        path = tmp_path / "test_golden.json"
        save_golden_dataset(dataset, path)
        loaded = load_golden_dataset(path)
        assert loaded.name == "save-test"
        assert loaded.version == "2.0"
        assert len(loaded.examples) == 1
        assert loaded.examples[0].max_hallucination == 0.5

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_golden_dataset("/nonexistent/path.json")

    def test_load_sample_golden(self):
        """Load the sample golden dataset from examples."""
        from pathlib import Path

        sample = Path("examples/golden_datasets/sample_golden.json")
        if sample.exists():
            dataset = load_golden_dataset(sample)
            assert len(dataset.examples) >= 3


class TestPromptRegressionModule:
    """Test regression test runner."""

    def test_module_name(self):
        module = PromptRegressionModule()
        assert module.name == "regression"

    def test_perfect_llm_passes(self):
        """A perfect LLM should pass all tests."""

        def perfect_llm(query: str) -> str:
            return f"Response to: {query}"

        dataset = GoldenDataset(
            name="test",
            examples=[
                GoldenExample(
                    query="q1",
                    expected_response="r1",
                    max_hallucination=1.0,
                    max_toxicity=1.0,
                ),
                GoldenExample(
                    query="q2",
                    expected_response="r2",
                    max_hallucination=1.0,
                    max_toxicity=1.0,
                ),
            ],
        )

        module = PromptRegressionModule()
        result = module.run_regression_suite(perfect_llm, dataset)
        assert result.total == 2
        assert result.passed == 2
        assert result.failed == 0
        assert result.pass_rate == 1.0

    def test_empty_response_fails(self):
        """LLM returning empty string should fail."""

        def empty_llm(query: str) -> str:
            return ""

        dataset = GoldenDataset(
            name="test",
            examples=[
                GoldenExample(query="q1", expected_response="r1"),
            ],
        )

        module = PromptRegressionModule()
        result = module.run_regression_suite(empty_llm, dataset)
        assert result.failed == 1
        assert "empty response" in result.failures[0].violations[0]

    def test_failing_llm_reports_failures(self):
        """LLM that raises should report failure."""

        def bad_llm(query: str) -> str:
            raise RuntimeError("API error")

        dataset = GoldenDataset(
            name="test",
            examples=[
                GoldenExample(query="q1", expected_response="r1"),
            ],
        )

        module = PromptRegressionModule()
        result = module.run_regression_suite(bad_llm, dataset)
        assert result.failed == 1
        assert "LLM call failed" in result.failures[0].violations[0]

    def test_pass_rate_calculation(self):
        """Pass rate should be correct."""

        call_count = 0

        def mixed_llm(query: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return "good response"
            return ""

        dataset = GoldenDataset(
            name="test",
            examples=[
                GoldenExample(
                    query=f"q{i}",
                    expected_response=f"r{i}",
                    max_hallucination=1.0,
                    max_toxicity=1.0,
                )
                for i in range(4)
            ],
        )

        module = PromptRegressionModule()
        result = module.run_regression_suite(mixed_llm, dataset)
        assert result.total == 4
        assert result.passed == 2
        assert result.failed == 2
        assert result.pass_rate == 0.5

    def test_result_serialization(self):
        """RegressionResult should serialize properly."""
        result = RegressionResult(
            dataset_name="test",
            total=10,
            passed=8,
            failed=2,
            pass_rate=0.8,
        )
        data = result.model_dump()
        restored = RegressionResult.model_validate(data)
        assert restored.pass_rate == 0.8

    def test_empty_dataset(self):
        """Empty dataset should return 0/0."""

        def llm(query: str) -> str:
            return "response"

        dataset = GoldenDataset(name="empty", examples=[])
        module = PromptRegressionModule()
        result = module.run_regression_suite(llm, dataset)
        assert result.total == 0
        assert result.pass_rate == 0.0
