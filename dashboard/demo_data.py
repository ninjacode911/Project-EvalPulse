"""Synthetic demo data generator for HuggingFace Spaces deployment."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

from evalpulse.models import EvalRecord


def generate_demo_records(n: int = 200) -> list[EvalRecord]:
    """Generate N synthetic EvalRecords with realistic distributions.

    Simulates an LLM app with:
    - Generally good performance (health 70-95)
    - Occasional hallucination spikes
    - Gradual drift over time
    - Some toxic/denial responses
    """
    random.seed(42)
    records = []
    now = datetime.now(UTC)

    queries = [
        "What is machine learning?",
        "Explain neural networks",
        "How does RAG work?",
        "What is Python used for?",
        "Describe transformer architecture",
        "What are embeddings?",
        "How do LLMs handle context?",
        "What is fine-tuning?",
        "Explain attention mechanism",
        "What is prompt engineering?",
    ]

    models = ["llama-3.1-70b", "gpt-4o-mini", "gemini-flash"]

    for i in range(n):
        ts = now - timedelta(hours=n - i)
        query = random.choice(queries)
        model = random.choice(models)

        # Simulate drift: later responses drift slightly
        drift_factor = i / n * 0.1

        # Base scores
        halluc = random.gauss(0.12, 0.08) + drift_factor * 0.5
        halluc = max(0.0, min(1.0, halluc))

        drift = random.gauss(0.05, 0.03) + drift_factor
        drift = max(0.0, min(1.0, drift))

        sentiment = random.gauss(0.7, 0.1)
        sentiment = max(0.0, min(1.0, sentiment))

        toxicity = abs(random.gauss(0.02, 0.02))
        toxicity = max(0.0, min(1.0, toxicity))

        is_denial = random.random() < 0.05
        length = random.randint(20, 200)

        # RAG scores (70% of calls are RAG)
        is_rag = random.random() < 0.7
        faith = None
        ctx_rel = None
        ans_rel = None
        ground = None
        context = None

        if is_rag:
            faith = random.gauss(0.75, 0.1)
            faith = max(0.0, min(1.0, faith))
            ctx_rel = random.gauss(0.8, 0.08)
            ctx_rel = max(0.0, min(1.0, ctx_rel))
            ans_rel = random.gauss(0.78, 0.09)
            ans_rel = max(0.0, min(1.0, ans_rel))
            ground = 0.4 * faith + 0.3 * ctx_rel + 0.3 * ans_rel
            context = f"Context for: {query}"

        # Compute health score
        components = [(1 - halluc) * 0.35, (1 - drift) * 0.25]
        if ground is not None:
            components.append(ground * 0.20)
        quality = (1 - toxicity) * 0.5 + sentiment * 0.4 + 0.1
        components.append(quality * 0.15)
        health = int(
            sum(components) / sum([0.35, 0.25] + ([0.20] if ground else []) + [0.15]) * 100
        )
        health = max(0, min(100, health))

        record = EvalRecord(
            app_name="demo-app",
            timestamp=ts,
            query=query,
            context=context,
            response=f"Demo response for: {query}",
            model_name=model,
            latency_ms=random.randint(50, 500),
            tags=["demo"],
            hallucination_score=round(halluc, 4),
            hallucination_method="embedding",
            drift_score=round(drift, 4),
            faithfulness_score=round(faith, 4) if faith else None,
            context_relevance=round(ctx_rel, 4) if ctx_rel else None,
            answer_relevancy=round(ans_rel, 4) if ans_rel else None,
            groundedness_score=round(ground, 4) if ground else None,
            sentiment_score=round(sentiment, 4),
            toxicity_score=round(toxicity, 4),
            response_length=length,
            language_detected="en",
            is_denial=is_denial,
            health_score=health,
        )
        records.append(record)

    return records
