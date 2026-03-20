"""End-to-end smoke test for EvalPulse Week 1."""

import time

import pytest
import yaml


class TestSmokeEndToEnd:
    """Full end-to-end test: init -> track -> verify storage -> shutdown."""

    @pytest.fixture
    def setup_evalpulse(self, tmp_path):
        """Set up a temporary EvalPulse environment with modules disabled for speed."""
        config_data = {
            "app_name": "smoke-test",
            "sqlite_path": str(tmp_path / "smoke.db"),
            "modules": {
                "hallucination": False,
                "drift": False,
                "rag_quality": False,
                "response_quality": False,
                "regression": False,
            },
        }
        config_path = tmp_path / "evalpulse.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        return str(config_path), str(tmp_path / "smoke.db")

    def test_full_flow(self, setup_evalpulse):
        """Test: init -> @track 10 calls -> verify records -> shutdown."""
        config_path, db_path = setup_evalpulse

        from evalpulse.config import reset_config
        from evalpulse.storage import reset_storage

        reset_config()
        reset_storage()

        from evalpulse.sdk import init, shutdown, track

        init(config_path=config_path)

        @track(app="smoke-test", model="test-model")
        def fake_llm(query):
            return f"Answer to: {query}"

        # Track 10 calls
        for i in range(10):
            result = fake_llm(f"Question {i}")
            assert result == f"Answer to: Question {i}"

        # Wait for worker to process
        time.sleep(3)

        # Verify records in storage
        from evalpulse.storage import get_storage

        storage = get_storage()
        count = storage.count()
        assert count == 10, f"Expected 10 records, got {count}"

        # Verify record contents
        records = storage.get_latest(10)
        assert len(records) == 10
        for r in records:
            assert r.app_name == "smoke-test"
            assert r.model_name == "test-model"
            assert "Answer to:" in r.response

        shutdown()

    def test_dashboard_creates_without_error(self):
        """Dashboard app should create without error."""
        from dashboard.app import create_app

        app = create_app()
        assert app is not None
