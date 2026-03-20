"""Shared embedding service using sentence-transformers.

Provides a singleton EmbeddingService that lazy-loads the all-MiniLM-L6-v2 model
and generates 384-dimensional embeddings for text.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("evalpulse.modules.embeddings")

_embedding_service: EmbeddingService | None = None


class EmbeddingService:
    """Sentence-transformers based embedding service.

    Lazy-loads the model on first use. Generates normalized 384-dim embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy-load the sentence-transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device="cpu")
            logger.info(f"Loaded embedding model: {self._model_name} (device=cpu)")

    def embed(self, text: str) -> list[float]:
        """Generate a normalized embedding for a single text.

        Returns a 384-dimensional unit vector.
        """
        self._load_model()
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate normalized embeddings for a batch of texts.

        Returns a list of 384-dimensional unit vectors.
        """
        if not texts:
            return []
        self._load_model()
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (384 for MiniLM)."""
        return 384


def get_embedding_service(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingService:
    """Get the global embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService(model_name)
    return _embedding_service


def reset_embedding_service() -> None:
    """Reset the embedding service singleton (for testing)."""
    global _embedding_service
    _embedding_service = None
