"""EvalPulse Health Score — composite quality metric (0-100).

Combines all module scores into a single at-a-glance health indicator.
"""

from __future__ import annotations

from evalpulse.models import EvalRecord

# Weight configuration for health score components
_WEIGHTS = {
    "hallucination": 0.35,  # Most critical
    "drift": 0.25,  # Silent degradation
    "rag": 0.20,  # RAG quality
    "quality": 0.15,  # Surface quality
    "regression": 0.05,  # Stability
}


def compute_health_score(record: EvalRecord) -> int:
    """Compute the health score for a single evaluation record.

    Formula:
        health = (
            (1 - hallucination_score) * 0.35 +
            (1 - drift_score)         * 0.25 +
            rag_groundedness_score    * 0.20 +
            response_quality_score    * 0.15 +
            regression_pass_rate      * 0.05
        ) * 100

    When a component is None (e.g., no RAG context), its weight
    is redistributed proportionally among available components.

    Returns an integer from 0 to 100.
    """
    components: dict[str, float | None] = {}

    # Hallucination: lower is better, invert
    components["hallucination"] = 1.0 - record.hallucination_score

    # Drift: lower is better, invert (None if insufficient data)
    if record.drift_score is not None:
        components["drift"] = 1.0 - record.drift_score
    else:
        components["drift"] = None

    # RAG: groundedness (None if not a RAG call)
    if record.groundedness_score is not None:
        components["rag"] = record.groundedness_score
    elif record.faithfulness_score is not None:
        # Fallback: use faithfulness if groundedness not computed
        components["rag"] = record.faithfulness_score
    else:
        components["rag"] = None

    # Quality: composite from toxicity and sentiment
    quality_score = _compute_quality_subscore(record)
    components["quality"] = quality_score

    # Regression: pass rate (not yet implemented, default to 1.0)
    components["regression"] = None

    # Compute weighted score, redistributing None weights
    available = {k: v for k, v in components.items() if v is not None}
    if not available:
        return 50  # No data, neutral score

    total_weight = sum(_WEIGHTS[k] for k in available)
    if total_weight == 0:
        return 50

    weighted_sum = sum(available[k] * (_WEIGHTS[k] / total_weight) for k in available)

    score = int(round(weighted_sum * 100))
    return max(0, min(100, score))


def _compute_quality_subscore(record: EvalRecord) -> float:
    """Compute the quality subscore from individual quality metrics.

    Combines toxicity (inverted) and sentiment into a single score.
    """
    # Toxicity: lower is better
    toxicity_component = 1.0 - record.toxicity_score

    # Sentiment: already 0-1 scale, higher is better
    # But neutral (0.5) is fine for factual responses
    # Normalize: treat 0.3-0.7 as "good" range
    sentiment_component = record.sentiment_score

    # Denial: penalize slightly
    denial_penalty = 0.1 if record.is_denial else 0.0

    quality = toxicity_component * 0.5 + sentiment_component * 0.4 + (1.0 - denial_penalty) * 0.1

    return max(0.0, min(1.0, quality))


def compute_aggregate_health(
    records: list[EvalRecord],
) -> int:
    """Compute aggregate health score over multiple records.

    Returns the mean of individual health scores.
    """
    if not records:
        return 50

    scores = [r.health_score for r in records if r.health_score > 0]
    if not scores:
        return 50

    return int(round(sum(scores) / len(scores)))
