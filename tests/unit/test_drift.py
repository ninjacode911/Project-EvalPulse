"""Tests for the Semantic Drift Detector module."""

import numpy as np
import pytest

from evalpulse.models import EvalEvent


class TestEmbeddingService:
    """Test the shared embedding service."""

    def test_embed_returns_384_dim(self):
        from evalpulse.modules.embeddings import EmbeddingService

        service = EmbeddingService()
        vec = service.embed("Hello world")
        assert len(vec) == 384

    def test_embed_is_normalized(self):
        from evalpulse.modules.embeddings import EmbeddingService

        service = EmbeddingService()
        vec = service.embed("Hello world")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01

    def test_embed_batch(self):
        from evalpulse.modules.embeddings import EmbeddingService

        service = EmbeddingService()
        vecs = service.embed_batch(["Hello", "World"])
        assert len(vecs) == 2
        assert len(vecs[0]) == 384

    def test_similar_texts_similar_embeddings(self):
        from evalpulse.modules.embeddings import EmbeddingService

        service = EmbeddingService()
        vec1 = service.embed("The weather is sunny today")
        vec2 = service.embed("Today the weather is bright and sunny")
        sim = np.dot(vec1, vec2)
        assert sim > 0.7  # Similar texts should have high cosine similarity


class TestDriftVectorStore:
    """Test the ChromaDB drift store."""

    @pytest.fixture
    def store(self, tmp_path):
        from evalpulse.modules.drift_store import DriftVectorStore

        s = DriftVectorStore(persist_dir=str(tmp_path / "chroma"))
        yield s
        s.close()

    def test_add_and_count(self, store):
        vec = [0.1] * 384
        store.add_embedding("id1", vec, "test-app")
        assert store.get_embedding_count("test-app") == 1

    def test_add_multiple(self, store):
        for i in range(10):
            vec = [float(i) / 10] * 384
            store.add_embedding(f"id-{i}", vec, "test-app")
        assert store.get_embedding_count("test-app") == 10

    def test_get_all_embeddings(self, store):
        for i in range(5):
            vec = [0.1 * i] * 384
            store.add_embedding(f"id-{i}", vec, "test-app")
        embeddings, metadatas = store.get_all_embeddings("test-app")
        assert len(embeddings) == 5
        assert len(metadatas) == 5

    def test_compute_centroid(self, store):
        embs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        centroid = store.compute_centroid(embs)
        assert len(centroid) == 3
        # Centroid should be normalized
        norm = np.linalg.norm(centroid)
        assert abs(norm - 1.0) < 0.01

    def test_cosine_distance_identical(self, store):
        vec = [1.0, 0.0, 0.0]
        dist = store.cosine_distance(vec, vec)
        assert abs(dist) < 0.01

    def test_cosine_distance_orthogonal(self, store):
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [0.0, 1.0, 0.0]
        dist = store.cosine_distance(vec_a, vec_b)
        assert abs(dist - 1.0) < 0.01

    def test_collection_per_app(self, store):
        store.add_embedding("id1", [0.1] * 384, "app-a")
        store.add_embedding("id2", [0.2] * 384, "app-b")
        assert store.get_embedding_count("app-a") == 1
        assert store.get_embedding_count("app-b") == 1


class TestSemanticDriftModule:
    """Test the drift detection module."""

    def test_module_name(self):
        from evalpulse.modules.drift import SemanticDriftModule

        module = SemanticDriftModule()
        assert module.name == "semantic_drift"

    def test_empty_response_returns_none(self):
        from evalpulse.modules.drift import SemanticDriftModule

        module = SemanticDriftModule()
        event = EvalEvent(response="")
        result = module.evaluate_sync(event)
        assert result["drift_score"] is None
        assert result["embedding_vector"] == []

    def test_insufficient_baseline_returns_none(self, tmp_path):
        from unittest.mock import patch

        from evalpulse.modules.drift import SemanticDriftModule
        from evalpulse.modules.drift_store import DriftVectorStore, reset_drift_store
        from evalpulse.modules.embeddings import reset_embedding_service

        reset_drift_store()
        reset_embedding_service()

        # Use a fresh temp directory so no leftover ChromaDB data
        fresh_store = DriftVectorStore(persist_dir=str(tmp_path / "chroma_test"))
        with patch(
            "evalpulse.modules.drift_store.get_drift_store",
            return_value=fresh_store,
        ):
            module = SemanticDriftModule()
            # Only add 3 events (below minimum of 10)
            for i in range(3):
                event = EvalEvent(
                    app_name="drift-baseline-test",
                    query=f"query {i}",
                    response=f"The weather is nice today, test response {i}.",
                )
                result = module.evaluate_sync(event)

            assert result["drift_score"] is None

        fresh_store.close()
        reset_drift_store()
