"""Integration test for the quality module end-to-end flow."""

import time

import pytest
import yaml


class TestQualityFlow:
    """End-to-end test: init -> track diverse calls -> verify quality scores."""

    @pytest.fixture(autouse=True)
    def _offline_hf(self, monkeypatch):
        """Avoid HuggingFace network calls during tests."""
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    @pytest.fixture
    def setup_evalpulse(self, tmp_path):
        config_data = {
            "app_name": "quality-test",
            "sqlite_path": str(tmp_path / "quality.db"),
            "modules": {
                "hallucination": False,
                "drift": False,
                "rag_quality": False,
                "response_quality": True,
                "regression": False,
            },
        }
        config_path = tmp_path / "evalpulse.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        return str(config_path)

    def test_quality_scores_populated(self, setup_evalpulse):
        """Track diverse responses and verify quality scores are non-trivial."""
        from evalpulse.config import reset_config
        from evalpulse.storage import reset_storage

        reset_config()
        reset_storage()

        import evalpulse.sdk as sdk

        # Ensure clean SDK state
        if sdk._worker is not None:
            sdk._worker.stop()
        sdk._worker = None
        sdk._initialized = False

        from evalpulse.sdk import init, shutdown, track

        init(config_path=setup_evalpulse)

        @track(app="quality-test")
        def fake_llm(query):
            responses = {
                "positive": "This is an excellent and wonderful answer!",
                "negative": "This is terrible, awful, and bad.",
                "neutral": "The temperature is 72 degrees.",
                "denial": "I cannot provide medical advice as an AI.",
                "french": "Bonjour, comment allez-vous? Je suis content de vous rencontrer ici.",
            }
            return responses.get(query, "Default response.")

        test_cases = ["positive", "negative", "neutral", "denial", "french"]
        for case in test_cases:
            fake_llm(case)

        # Poll for records — detoxify model loading can take several seconds
        from evalpulse.storage import get_storage

        storage = get_storage()
        records = []
        for _ in range(60):
            time.sleep(1)
            records = storage.get_latest(10)
            if len(records) >= 5:
                break

        assert len(records) == 5, f"Expected 5 records, got {len(records)}"

        # Check that quality scores are populated (not all defaults)
        for r in records:
            assert r.response_length > 0, f"response_length should be > 0 for '{r.query}'"

        shutdown()
