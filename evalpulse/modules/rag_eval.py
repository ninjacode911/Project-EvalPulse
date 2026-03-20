"""Module 3: RAG Quality Evaluator.

Evaluates the full RAG pipeline — retrieval quality and generation faithfulness.
Measures whether retrieved context is relevant and whether the answer uses it.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from evalpulse.models import EvalEvent
from evalpulse.modules.base import EvalModule

logger = logging.getLogger("evalpulse.modules.rag_eval")


class RAGQualityModule(EvalModule):
    """Module 3: RAG Quality Evaluator.

    Evaluates:
    - Context Relevance: Is the retrieved context relevant to the query?
    - Faithfulness: Does the response stay within the context?
    - Answer Relevancy: Does the response actually address the query?
    - Groundedness: Composite score of the above three.

    Skips evaluation entirely when no context is provided (non-RAG call).
    """

    @property
    def name(self) -> str:
        return "rag_quality"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import sentence_transformers  # noqa: F401

            return True
        except ImportError:
            return False

    async def evaluate(self, event: EvalEvent) -> dict[str, Any]:
        return self.evaluate_sync(event)

    def evaluate_sync(self, event: EvalEvent) -> dict[str, Any]:
        """Evaluate RAG quality metrics."""
        # Skip if no context provided (not a RAG call)
        if not event.context or not event.context.strip():
            return {
                "faithfulness_score": None,
                "context_relevance": None,
                "answer_relevancy": None,
                "groundedness_score": None,
            }

        try:
            from evalpulse.modules.embeddings import get_embedding_service

            embedder = get_embedding_service()

            query = (event.query or "")[:10000]
            context = (event.context or "")[:10000]
            response = (event.response or "")[:10000]

            if not query.strip() or not response.strip():
                return {
                    "faithfulness_score": None,
                    "context_relevance": None,
                    "answer_relevancy": None,
                    "groundedness_score": None,
                }

            # Embed all three components
            query_emb = np.array(embedder.embed(query))
            context_emb = np.array(embedder.embed(context))
            response_emb = np.array(embedder.embed(response))

            # 1. Context Relevance: cosine similarity between query and context
            context_relevance = float(np.dot(query_emb, context_emb))
            context_relevance = max(0.0, min(1.0, context_relevance))

            # 2. Faithfulness: cosine similarity between response and context
            # High similarity = response stays close to context
            faithfulness = float(np.dot(response_emb, context_emb))
            faithfulness = max(0.0, min(1.0, faithfulness))

            # 3. Answer Relevancy: cosine similarity between query and response
            # High similarity = response addresses the query
            answer_relevancy = float(np.dot(query_emb, response_emb))
            answer_relevancy = max(0.0, min(1.0, answer_relevancy))

            # 4. Sentence-level faithfulness refinement
            # Reuse pre-computed context_emb to avoid redundant embedding
            sentence_faithfulness = self._sentence_level_faithfulness(
                response, context_emb, embedder
            )
            if sentence_faithfulness is not None:
                # Blend embedding-level and sentence-level
                faithfulness = 0.6 * faithfulness + 0.4 * sentence_faithfulness

            # 5. Groundedness: weighted composite
            groundedness = faithfulness * 0.40 + context_relevance * 0.30 + answer_relevancy * 0.30

            return {
                "faithfulness_score": round(faithfulness, 4),
                "context_relevance": round(context_relevance, 4),
                "answer_relevancy": round(answer_relevancy, 4),
                "groundedness_score": round(groundedness, 4),
            }

        except Exception as e:
            logger.warning(f"RAG evaluation failed: {e}")
            return {
                "faithfulness_score": None,
                "context_relevance": None,
                "answer_relevancy": None,
                "groundedness_score": None,
            }

    def _sentence_level_faithfulness(
        self, response: str, context_emb: np.ndarray, embedder: Any
    ) -> float | None:
        """Compute sentence-level faithfulness.

        Splits the response into sentences and checks each one
        against the pre-computed context embedding.
        Returns average faithfulness.
        """
        import re

        sentences = re.split(r"[.!?]+", response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

        if not sentences:
            return None

        scores = []
        for sentence in sentences[:10]:  # Limit to 10 sentences
            sent_emb = np.array(embedder.embed(sentence))
            sim = float(np.dot(sent_emb, context_emb))
            scores.append(max(0.0, min(1.0, sim)))

        return sum(scores) / len(scores) if scores else None
