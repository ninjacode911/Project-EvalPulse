"""EvalPulse data models — EvalEvent (SDK input) and EvalRecord (full evaluation result)."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

# --- Safety limits ---
_MAX_TEXT_LEN = 200_000  # Max characters for query/context/response
_MAX_SHORT_STR = 256  # Max characters for app_name, model_name, etc.
_MAX_TAGS = 50
_MAX_TAG_LEN = 100
_MAX_EMBEDDING_DIM = 2048  # Largest common embedding dimension
_MAX_FLAGGED_CLAIMS = 20


def _utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(UTC)


def _new_id() -> str:
    """Generate a new UUID string."""
    return str(uuid4())


def _truncate(value: str, max_len: int) -> str:
    """Truncate a string to max_len if it exceeds the limit."""
    if len(value) > max_len:
        return value[:max_len]
    return value


class EvalEvent(BaseModel):
    """Lightweight event produced by the SDK before evaluation.

    This is what the @track decorator pushes to the queue.
    Contains only the raw data captured from the LLM call.
    """

    id: str = Field(default_factory=_new_id)
    app_name: str = Field(default="default", max_length=_MAX_SHORT_STR)
    timestamp: datetime = Field(default_factory=_utc_now)
    query: str = ""
    context: str | None = None
    response: str = ""
    model_name: str = Field(default="unknown", max_length=_MAX_SHORT_STR)
    latency_ms: int = Field(default=0, ge=0)
    tags: list[str] = Field(default_factory=list)

    @field_validator("query", "response", mode="before")
    @classmethod
    def truncate_text(cls, v: str) -> str:
        if isinstance(v, str):
            return _truncate(v, _MAX_TEXT_LEN)
        return v

    @field_validator("context", mode="before")
    @classmethod
    def truncate_context(cls, v: str | None) -> str | None:
        if isinstance(v, str):
            return _truncate(v, _MAX_TEXT_LEN)
        return v

    @field_validator("tags", mode="before")
    @classmethod
    def validate_tags(cls, v: list) -> list:
        if isinstance(v, list):
            return [_truncate(str(t), _MAX_TAG_LEN) for t in v[:_MAX_TAGS]]
        return v


class EvalRecord(BaseModel):
    """Full evaluation result stored in the database.

    Contains all module scores computed by the evaluation engine.
    One EvalRecord per LLM call.
    """

    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(default_factory=_new_id)
    app_name: str = Field(default="default", max_length=_MAX_SHORT_STR)
    timestamp: datetime = Field(default_factory=_utc_now)
    query: str = ""
    context: str | None = None
    response: str = ""
    model_name: str = Field(default="unknown", max_length=_MAX_SHORT_STR)
    latency_ms: int = Field(default=0, ge=0)
    tags: list[str] = Field(default_factory=list)

    # Module 1: Hallucination
    hallucination_score: float = Field(default=0.0, ge=0.0, le=1.0)
    hallucination_method: str = "none"
    flagged_claims: list[str] = Field(default_factory=list)

    # Module 2: Drift
    embedding_vector: list[float] = Field(default_factory=list)
    drift_score: float | None = Field(default=None, ge=0.0, le=1.0)

    # Module 3: RAG Quality
    faithfulness_score: float | None = Field(default=None, ge=0.0, le=1.0)
    context_relevance: float | None = Field(default=None, ge=0.0, le=1.0)
    answer_relevancy: float | None = Field(default=None, ge=0.0, le=1.0)
    groundedness_score: float | None = Field(default=None, ge=0.0, le=1.0)

    # Module 4: Response Quality
    sentiment_score: float = Field(default=0.5, ge=0.0, le=1.0)
    toxicity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    response_length: int = Field(default=0, ge=0)
    language_detected: str = Field(default="en", max_length=10)
    is_denial: bool = False

    # Composite
    health_score: int = Field(default=0, ge=0, le=100)

    @field_validator("query", "response", mode="before")
    @classmethod
    def truncate_text(cls, v: str) -> str:
        if isinstance(v, str):
            return _truncate(v, _MAX_TEXT_LEN)
        return v

    @field_validator("context", mode="before")
    @classmethod
    def truncate_context(cls, v: str | None) -> str | None:
        if isinstance(v, str):
            return _truncate(v, _MAX_TEXT_LEN)
        return v

    @field_validator("tags", mode="before")
    @classmethod
    def validate_tags(cls, v: list) -> list:
        if isinstance(v, list):
            return [_truncate(str(t), _MAX_TAG_LEN) for t in v[:_MAX_TAGS]]
        return v

    @field_validator("embedding_vector", mode="before")
    @classmethod
    def validate_embedding_vector(cls, v: list) -> list:
        if isinstance(v, list) and len(v) > _MAX_EMBEDDING_DIM:
            return v[:_MAX_EMBEDDING_DIM]
        return v

    @field_validator("flagged_claims", mode="before")
    @classmethod
    def validate_flagged_claims(cls, v: list) -> list:
        if isinstance(v, list):
            return [_truncate(str(c), 500) for c in v[:_MAX_FLAGGED_CLAIMS]]
        return v

    @classmethod
    def from_event(cls, event: EvalEvent) -> EvalRecord:
        """Create an EvalRecord from an EvalEvent with default (zero) scores."""
        return cls(
            id=event.id,
            app_name=event.app_name,
            timestamp=event.timestamp,
            query=event.query,
            context=event.context,
            response=event.response,
            model_name=event.model_name,
            latency_ms=event.latency_ms,
            tags=event.tags,
            response_length=len(event.response.split()) if event.response else 0,
        )
