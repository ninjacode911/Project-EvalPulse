"""Module 2: Semantic Drift Detector.

Detects when LLM output quality silently shifts over time by comparing
response embeddings against a rolling baseline using cosine distance.
"""

from __future__ import annotations

import logging
from typing import Any

from evalpulse.models import EvalEvent
from evalpulse.modules.base import EvalModule

logger = logging.getLogger("evalpulse.modules.drift")

_MIN_BASELINE_SIZE = 10  # Minimum embeddings needed for drift scoring


class SemanticDriftModule(EvalModule):
    """Module 2: Semantic Drift Detector.

    Embeds each response, stores in ChromaDB, and computes drift score
    as cosine distance between the current embedding and the baseline centroid.
    """

    def __init__(self, config: Any = None):
        self._config = config
        self._baseline_centroid: list[float] | None = None
        self._baseline_count = 0

    @property
    def name(self) -> str:
        return "semantic_drift"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import chromadb  # noqa: F401
            import sentence_transformers  # noqa: F401

            return True
        except ImportError:
            return False

    async def evaluate(self, event: EvalEvent) -> dict[str, Any]:
        return self.evaluate_sync(event)

    def evaluate_sync(self, event: EvalEvent) -> dict[str, Any]:
        """Evaluate semantic drift for a response."""
        response = event.response or ""
        if not response.strip():
            return {"embedding_vector": [], "drift_score": None}

        try:
            from evalpulse.modules.drift_store import get_drift_store
            from evalpulse.modules.embeddings import get_embedding_service

            embedder = get_embedding_service()
            store = get_drift_store()

            # Embed the response
            embedding = embedder.embed(response)

            # Store the embedding
            store.add_embedding(
                embedding_id=event.id,
                embedding=embedding,
                app_name=event.app_name,
                metadata={
                    "query": event.query[:200] if event.query else "",
                    "model_name": event.model_name,
                },
            )

            # Compute drift score
            drift_score = self._compute_drift(embedding, event.app_name, store)

            return {
                "embedding_vector": embedding,
                "drift_score": drift_score,
            }
        except Exception as e:
            logger.warning(f"Drift detection failed: {e}")
            return {"embedding_vector": [], "drift_score": None}

    def _compute_drift(
        self,
        current_embedding: list[float],
        app_name: str,
        store: Any,
    ) -> float | None:
        """Compute drift score against the baseline centroid.

        Returns None if insufficient baseline data.
        Returns 0.0 (no drift) to 1.0 (maximum drift).
        """
        count = store.get_embedding_count(app_name)
        if count < _MIN_BASELINE_SIZE:
            return None

        # Recompute baseline centroid periodically
        needs_recompute = self._baseline_centroid is None or self._baseline_count != count
        if needs_recompute:
            all_embeddings, _ = store.get_all_embeddings(app_name)
            if len(all_embeddings) < _MIN_BASELINE_SIZE:
                return None
            self._baseline_centroid = store.compute_centroid(all_embeddings)
            self._baseline_count = count

        # Cosine distance: 0 = identical, 2 = opposite
        raw_distance = store.cosine_distance(current_embedding, self._baseline_centroid)

        # Normalize to 0.0-1.0 range (divide by 2), clamp for float safety
        drift_score = max(0.0, min(raw_distance / 2.0, 1.0))

        return round(drift_score, 4)
