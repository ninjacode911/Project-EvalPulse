"""Golden dataset schema and loader for prompt regression testing."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class GoldenExample(BaseModel):
    """A single golden test example with expected behavior."""

    query: str
    expected_response: str = ""
    context: str | None = None
    tags: list[str] = Field(default_factory=list)
    max_hallucination: float = 0.3
    min_faithfulness: float | None = None
    expected_language: str | None = None
    max_toxicity: float = 0.1


class GoldenDataset(BaseModel):
    """A collection of golden test examples for regression testing."""

    name: str = "default"
    version: str = "1.0"
    description: str = ""
    examples: list[GoldenExample] = Field(default_factory=list)


def load_golden_dataset(path: str | Path) -> GoldenDataset:
    """Load a golden dataset from a JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}")

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    return GoldenDataset.model_validate(data)


def save_golden_dataset(dataset: GoldenDataset, path: str | Path) -> None:
    """Save a golden dataset to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(
            dataset.model_dump(),
            f,
            indent=2,
            default=str,
        )
