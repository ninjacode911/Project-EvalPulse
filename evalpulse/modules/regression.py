"""Module 5: Prompt Regression Tester.

Runs automated regression tests against a golden dataset to catch
quality regressions when prompts or models change.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from evalpulse.models import EvalEvent
from evalpulse.modules.base import EvalModule

logger = logging.getLogger("evalpulse.modules.regression")


class RegressionFailure(BaseModel):
    """A single regression test failure."""

    query: str
    expected_response: str = ""
    actual_response: str = ""
    violations: list[str] = Field(default_factory=list)


class RegressionResult(BaseModel):
    """Result of a regression test suite run."""

    dataset_name: str = "default"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    total: int = 0
    passed: int = 0
    failed: int = 0
    pass_rate: float = 0.0
    failures: list[RegressionFailure] = Field(default_factory=list)


class PromptRegressionModule(EvalModule):
    """Module 5: Prompt Regression Tester.

    Unlike other modules, this runs in batch mode (not per-event).
    It tests an LLM function against a golden dataset and reports
    pass/fail results.
    """

    @property
    def name(self) -> str:
        return "regression"

    @classmethod
    def is_available(cls) -> bool:
        return True

    async def evaluate(self, event: EvalEvent) -> dict[str, Any]:
        # Regression module doesn't evaluate individual events
        return {}

    def evaluate_sync(self, event: EvalEvent) -> dict[str, Any]:
        return {}

    def run_regression_suite(
        self,
        llm_func: Callable[[str], str],
        dataset: Any,
        modules: list[EvalModule] | None = None,
    ) -> RegressionResult:
        """Run regression tests against a golden dataset.

        Args:
            llm_func: Function that takes a query and returns
                a response string.
            dataset: GoldenDataset with test examples.
            modules: Optional list of eval modules to run on
                each response.

        Returns:
            RegressionResult with pass/fail counts and failure details.
        """
        result = RegressionResult(
            dataset_name=dataset.name,
            total=len(dataset.examples),
        )

        for example in dataset.examples:
            try:
                # Call the LLM function
                actual_response = llm_func(example.query)

                # Create an eval event for module scoring
                event = EvalEvent(
                    id=str(uuid4()),
                    query=example.query,
                    context=example.context,
                    response=actual_response,
                )

                # Run evaluation modules if provided
                scores = {}
                if modules:
                    for module in modules:
                        try:
                            if module.is_available():
                                module_result = module.evaluate_sync(event)
                                scores.update(module_result)
                        except Exception as e:
                            logger.warning(f"Module {module.name} failed: {e}")

                # Check violations against thresholds
                violations = self._check_violations(example, scores, actual_response)

                if violations:
                    result.failed += 1
                    result.failures.append(
                        RegressionFailure(
                            query=example.query,
                            expected_response=example.expected_response,
                            actual_response=actual_response[:500],
                            violations=violations,
                        )
                    )
                else:
                    result.passed += 1

            except Exception as e:
                result.failed += 1
                result.failures.append(
                    RegressionFailure(
                        query=example.query,
                        violations=[f"LLM call failed: {e}"],
                    )
                )

        if result.total > 0:
            result.pass_rate = round(result.passed / result.total, 4)

        return result

    def _check_violations(
        self,
        example: Any,
        scores: dict,
        actual_response: str,
    ) -> list[str]:
        """Check if scores violate the golden example thresholds."""
        violations = []

        # Check hallucination threshold
        halluc = scores.get("hallucination_score")
        if halluc is not None and halluc > example.max_hallucination:
            violations.append(f"hallucination_score {halluc:.2f} > max {example.max_hallucination}")

        # Check toxicity threshold
        toxicity = scores.get("toxicity_score")
        if toxicity is not None and toxicity > example.max_toxicity:
            violations.append(f"toxicity_score {toxicity:.2f} > max {example.max_toxicity}")

        # Check faithfulness threshold (RAG)
        if example.min_faithfulness is not None:
            faith = scores.get("faithfulness_score")
            if faith is not None and faith < example.min_faithfulness:
                violations.append(
                    f"faithfulness_score {faith:.2f} < min {example.min_faithfulness}"
                )

        # Check language
        if example.expected_language:
            detected = scores.get("language_detected", "")
            if detected and detected != example.expected_language:
                violations.append(f"language {detected} != expected {example.expected_language}")

        # Check if response is empty
        if not actual_response or not actual_response.strip():
            violations.append("empty response")

        return violations
