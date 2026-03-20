"""Module 1: Hallucination Scorer.

Detects whether LLM responses contain claims not supported by the provided context.
Uses a combination of embedding-based consistency checking and optional Groq LLM-as-judge.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from evalpulse.models import EvalEvent
from evalpulse.modules.base import EvalModule

logger = logging.getLogger("evalpulse.modules.hallucination")


class HallucinationModule(EvalModule):
    """Module 1: Hallucination Scorer.

    Scoring methods:
    1. Embedding consistency: Compare response embedding against context embedding.
       Low similarity suggests the response strays from the context.
    2. LLM-as-judge (optional, requires Groq API): Ask an LLM to rate hallucination.
    3. Combined: Weighted average when both methods are available.
    """

    def __init__(self, config: Any = None):
        self._config = config
        self._groq_client = None
        self._groq_initialized = False

    @property
    def name(self) -> str:
        return "hallucination"

    @classmethod
    def is_available(cls) -> bool:
        """Always available — embedding-based scoring doesn't need extra deps."""
        return True

    def _get_groq_client(self):
        """Lazy-initialize the Groq client."""
        if not self._groq_initialized:
            self._groq_initialized = True
            try:
                import os

                from evalpulse.modules.groq_client import GroqClient

                api_key = None
                if self._config and hasattr(self._config, "groq_api_key"):
                    key = self._config.groq_api_key
                    # Handle SecretStr or plain str
                    if key is not None:
                        api_key = (
                            key.get_secret_value() if hasattr(key, "get_secret_value") else str(key)
                        )
                if not api_key:
                    api_key = os.environ.get("GROQ_API_KEY")
                if api_key:
                    self._groq_client = GroqClient(api_key=api_key)
                    logger.info("Groq client initialized for LLM-as-judge")
            except Exception:
                logger.info("Groq client not available")
        return self._groq_client

    async def evaluate(self, event: EvalEvent) -> dict[str, Any]:
        return self.evaluate_sync(event)

    def evaluate_sync(self, event: EvalEvent) -> dict[str, Any]:
        """Evaluate hallucination in a response."""
        response = event.response or ""
        if not response.strip():
            return {
                "hallucination_score": 0.0,
                "hallucination_method": "none",
                "flagged_claims": [],
            }

        # Method 1: Embedding-based consistency
        embedding_score = self._embedding_consistency_score(event)

        # Method 2: LLM-as-judge (if available)
        groq = self._get_groq_client()
        judge_score = None
        if groq and groq.is_available():
            judge_score = groq.judge_hallucination(
                query=event.query, context=event.context, response=response
            )

        # Combine scores
        if judge_score is not None and embedding_score is not None:
            # Weighted combination: 60% embedding, 40% judge
            hallucination_score = 0.6 * embedding_score + 0.4 * judge_score
            method = "both"
        elif judge_score is not None:
            hallucination_score = judge_score
            method = "llm_judge"
        elif embedding_score is not None:
            hallucination_score = embedding_score
            method = "embedding"
        else:
            hallucination_score = 0.0
            method = "none"

        # Extract flagged claims (sentences with potential issues)
        flagged = self._extract_flagged_claims(response, event.context)

        return {
            "hallucination_score": round(hallucination_score, 4),
            "hallucination_method": method,
            "flagged_claims": flagged,
        }

    def _embedding_consistency_score(self, event: EvalEvent) -> float | None:
        """Score hallucination using embedding similarity.

        Compares response embedding against context embedding (if available)
        or query embedding. Low similarity = potential hallucination.
        """
        try:
            from evalpulse.modules.embeddings import get_embedding_service

            embedder = get_embedding_service()
            response_emb = embedder.embed(event.response)

            if event.context:
                # Compare response against context
                context_emb = embedder.embed(event.context)
            else:
                # Compare response against query
                if not event.query:
                    return None
                context_emb = embedder.embed(event.query)

            # Cosine similarity (embeddings are normalized)
            import numpy as np

            similarity = float(np.dot(response_emb, context_emb))

            # Convert similarity to hallucination score
            # High similarity = low hallucination, low similarity = high hallucination
            # similarity ranges from -1 to 1, map to 0-1 hallucination score
            hallucination = max(0.0, min(1.0, (1.0 - similarity) / 2.0))
            return hallucination
        except Exception as e:
            logger.warning(f"Embedding consistency check failed: {e}")
            return None

    def _extract_flagged_claims(self, response: str, context: str | None) -> list[str]:
        """Extract sentences from the response that may be hallucinated.

        Simple heuristic: sentences containing specific factual claims
        (numbers, names, dates) that don't appear in the context.
        """
        if not context or not response:
            return []

        # Split response into sentences
        sentences = re.split(r"[.!?]+", response)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        flagged = []
        context_lower = context.lower()

        for sentence in sentences:
            # Check for specific factual patterns
            has_numbers = bool(re.search(r"\b\d{2,}\b", sentence))
            has_quotes = '"' in sentence or "'" in sentence
            has_names = bool(re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", sentence))

            if has_numbers or has_quotes or has_names:
                # Check if the factual content appears in context
                key_terms = re.findall(r"\b[A-Z][a-z]+\b|\b\d+\b", sentence)
                terms_in_context = sum(1 for t in key_terms if t.lower() in context_lower)
                if key_terms and terms_in_context < len(key_terms) / 2:
                    flagged.append(sentence[:200])

        return flagged[:5]  # Limit to 5 flagged claims
