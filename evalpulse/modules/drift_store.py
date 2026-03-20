"""ChromaDB vector store for semantic drift detection.

Stores response embeddings with metadata for drift baseline comparison.
Uses on-disk persistence via ChromaDB PersistentClient.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("evalpulse.modules.drift_store")

_drift_store: DriftVectorStore | None = None


class DriftVectorStore:
    """ChromaDB-backed vector store for drift detection.

    Stores embeddings per app with timestamps for windowed comparison.
    """

    def __init__(self, persist_dir: str = "chroma_data"):
        self._persist_dir = persist_dir
        self._client = None
        self._collections: dict = {}

    def _get_client(self):
        """Lazy-initialize the ChromaDB client."""
        if self._client is None:
            import chromadb

            Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._persist_dir)
            logger.info(f"Initialized ChromaDB at {self._persist_dir}")
        return self._client

    @staticmethod
    def _sanitize_collection_name(app_name: str) -> str:
        """Sanitize app_name into a valid ChromaDB collection name."""
        import re

        # Strip to alphanumeric and underscores only
        safe = re.sub(r"[^a-zA-Z0-9_]", "_", app_name)
        safe = re.sub(r"_+", "_", safe).strip("_")  # collapse multiple underscores
        safe = f"drift_{safe}" if safe else "drift_default"
        # ChromaDB requires 3-63 chars
        if len(safe) < 3:
            safe = safe + "_ep"
        return safe[:63]

    def _get_collection(self, app_name: str):
        """Get or create a collection for the given app."""
        safe_name = self._sanitize_collection_name(app_name)
        if safe_name not in self._collections:
            client = self._get_client()
            self._collections[safe_name] = client.get_or_create_collection(
                name=safe_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[safe_name]

    def add_embedding(
        self,
        embedding_id: str,
        embedding: list[float],
        app_name: str = "default",
        metadata: dict | None = None,
    ) -> None:
        """Store an embedding with metadata."""
        collection = self._get_collection(app_name)
        meta = metadata or {}
        meta["timestamp"] = meta.get("timestamp", time.time())
        meta["app_name"] = app_name
        collection.add(
            ids=[embedding_id],
            embeddings=[embedding],
            metadatas=[meta],
        )

    def get_all_embeddings(
        self,
        app_name: str = "default",
        limit: int = 10000,
    ) -> tuple[list[list[float]], list[dict]]:
        """Get all embeddings for an app.

        Returns (embeddings, metadatas).
        """
        collection = self._get_collection(app_name)
        count = collection.count()
        if count == 0:
            return [], []
        result = collection.get(
            limit=min(count, limit),
            include=["embeddings", "metadatas"],
        )
        embeddings = result.get("embeddings", [])
        metadatas = result.get("metadatas", [])
        # Ensure embeddings are Python lists (ChromaDB may return numpy)
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        return embeddings, metadatas

    def get_embedding_count(self, app_name: str = "default") -> int:
        """Get the number of stored embeddings for an app."""
        collection = self._get_collection(app_name)
        return collection.count()

    @staticmethod
    def compute_centroid(embeddings: list[list[float]]) -> list[float]:
        """Compute the mean embedding (centroid) of a list of embeddings."""
        if embeddings is None or len(embeddings) == 0:
            return []
        arr = np.array(embeddings)
        centroid = arr.mean(axis=0)
        # Normalize the centroid
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return centroid.tolist()

    @staticmethod
    def cosine_distance(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine distance between two vectors.

        Returns 0.0 (identical) to 2.0 (opposite).
        """
        a = np.array(vec_a)
        b = np.array(vec_b)
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        similarity = dot / (norm_a * norm_b)
        return float(1.0 - similarity)

    def close(self) -> None:
        """Close the ChromaDB client."""
        self._collections.clear()
        self._client = None


def get_drift_store(persist_dir: str = "chroma_data") -> DriftVectorStore:
    """Get the global drift store singleton."""
    global _drift_store
    if _drift_store is None:
        _drift_store = DriftVectorStore(persist_dir)
    return _drift_store


def reset_drift_store() -> None:
    """Reset the drift store singleton."""
    global _drift_store
    if _drift_store is not None:
        _drift_store.close()
    _drift_store = None
