"""Tests for the EvaluationWorker."""

import queue
import time

import pytest

from evalpulse.models import EvalEvent
from evalpulse.storage.sqlite_store import SQLiteStore
from evalpulse.worker import EvaluationWorker


class TestEvaluationWorker:
    """Test background worker functionality."""

    @pytest.fixture
    def event_queue(self):
        return queue.Queue(maxsize=1000)

    @pytest.fixture
    def store(self, tmp_db):
        s = SQLiteStore(tmp_db)
        yield s
        s.close()

    @pytest.fixture
    def worker(self, event_queue, store, monkeypatch):
        """Create a worker and patch storage to use our test store."""
        import evalpulse.storage as storage_mod

        monkeypatch.setattr(storage_mod, "_store", store)

        w = EvaluationWorker(event_queue=event_queue, batch_size=5, batch_timeout=0.5)
        yield w
        if w.is_running:
            w.stop(timeout=2)

    def test_start_and_stop(self, worker):
        """Worker should start and stop cleanly."""
        worker.start()
        assert worker.is_running
        worker.stop(timeout=2)
        assert not worker.is_running

    def test_processes_events(self, worker, event_queue, store):
        """Worker should process events and save to storage."""
        worker.start()

        for i in range(10):
            event = EvalEvent(
                app_name="test",
                query=f"query-{i}",
                response=f"response-{i}",
            )
            event_queue.put(event)

        # Wait for processing
        time.sleep(2)
        worker.stop(timeout=3)

        count = store.count()
        assert count == 10, f"Expected 10 records, got {count}"

    def test_batch_processing(self, worker, event_queue, store):
        """Worker should batch writes efficiently."""
        worker.start()

        # Put exactly batch_size events
        for i in range(5):
            event = EvalEvent(
                app_name="test",
                query=f"batch-{i}",
                response=f"response-{i}",
            )
            event_queue.put(event)

        time.sleep(2)
        worker.stop(timeout=3)
        assert store.count() == 5

    def test_graceful_shutdown_drains_queue(self, worker, event_queue, store):
        """Shutdown should drain remaining events."""
        worker.start()

        # Add events
        for i in range(8):
            event_queue.put(EvalEvent(app_name="test", query=f"q-{i}", response=f"r-{i}"))

        # Small delay then stop
        time.sleep(0.5)
        worker.stop(timeout=5)

        # All events should be saved
        assert store.count() == 8

    def test_event_to_record_conversion(self, worker, event_queue, store):
        """Events should be converted to records with correct fields."""
        worker.start()

        event = EvalEvent(
            app_name="conversion-test",
            query="test query",
            response="test response words here",
            model_name="gpt-test",
            latency_ms=200,
            tags=["tag1"],
        )
        event_queue.put(event)

        time.sleep(2)
        worker.stop(timeout=3)

        records = store.get_latest(1)
        assert len(records) == 1
        r = records[0]
        assert r.app_name == "conversion-test"
        assert r.query == "test query"
        assert r.model_name == "gpt-test"
        assert r.latency_ms == 200
        assert r.tags == ["tag1"]
