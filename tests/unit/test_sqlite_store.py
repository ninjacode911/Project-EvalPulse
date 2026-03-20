"""Tests for SQLite storage backend."""

import threading
from datetime import UTC, datetime, timedelta

import pytest

from evalpulse.models import EvalRecord
from evalpulse.storage.sqlite_store import SQLiteStore


class TestSQLiteStore:
    """Test SQLite storage operations."""

    @pytest.fixture
    def store(self, tmp_db):
        """Create a fresh SQLiteStore for each test."""
        s = SQLiteStore(tmp_db)
        yield s
        s.close()

    def test_save_and_retrieve(self, store, sample_eval_record):
        """Should save a record and retrieve it by ID."""
        store.save(sample_eval_record)
        retrieved = store.get_by_id(sample_eval_record.id)
        assert retrieved is not None
        assert retrieved.id == sample_eval_record.id
        assert retrieved.app_name == sample_eval_record.app_name
        assert retrieved.query == sample_eval_record.query

    def test_save_batch(self, store, make_eval_records):
        """Should save multiple records in a batch."""
        records = make_eval_records(20)
        store.save_batch(records)
        assert store.count() == 20

    def test_save_empty_batch(self, store):
        """Saving an empty batch should not error."""
        store.save_batch([])
        assert store.count() == 0

    def test_count_with_filters(self, store, make_eval_records):
        """Should count records matching filters."""
        records = make_eval_records(10, app_name="app-a")
        records += make_eval_records(5, app_name="app-b")
        store.save_batch(records)
        assert store.count() == 15
        assert store.count({"app_name": "app-a"}) == 10
        assert store.count({"app_name": "app-b"}) == 5

    def test_get_latest(self, store, make_eval_records):
        """Should return the N most recent records."""
        records = make_eval_records(20)
        store.save_batch(records)
        latest = store.get_latest(5)
        assert len(latest) == 5

    def test_query_with_limit_offset(self, store, make_eval_records):
        """Should support pagination via limit and offset."""
        records = make_eval_records(30)
        store.save_batch(records)
        page1 = store.query(limit=10, offset=0)
        page2 = store.query(limit=10, offset=10)
        assert len(page1) == 10
        assert len(page2) == 10
        assert page1[0].id != page2[0].id

    def test_json_fields_roundtrip(self, store):
        """JSON list fields should survive save/load cycle."""
        record = EvalRecord(
            app_name="test",
            query="q",
            response="r",
            tags=["tag1", "tag2"],
            flagged_claims=["claim1", "claim2"],
            embedding_vector=[0.1, 0.2, 0.3, 0.4, 0.5],
        )
        store.save(record)
        retrieved = store.get_by_id(record.id)
        assert retrieved.tags == ["tag1", "tag2"]
        assert retrieved.flagged_claims == ["claim1", "claim2"]
        assert retrieved.embedding_vector == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_boolean_field_roundtrip(self, store):
        """Boolean is_denial should survive save/load."""
        record = EvalRecord(app_name="test", query="q", response="r", is_denial=True)
        store.save(record)
        retrieved = store.get_by_id(record.id)
        assert retrieved.is_denial is True

    def test_get_by_id_not_found(self, store):
        """Should return None for nonexistent ID."""
        assert store.get_by_id("nonexistent-id") is None

    def test_concurrent_writes(self, store):
        """Should handle concurrent writes from multiple threads."""
        errors = []

        def writer(thread_id):
            try:
                for i in range(25):
                    record = EvalRecord(
                        app_name=f"thread-{thread_id}",
                        query=f"query-{i}",
                        response=f"response-{i}",
                    )
                    store.save(record)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert store.count() == 100  # 4 threads x 25 records

    def test_get_time_series(self, store, make_eval_records):
        """Should return aggregated time-series data."""
        records = make_eval_records(10)
        store.save_batch(records)
        series = store.get_time_series("health_score")
        assert isinstance(series, list)
        # Should have at least one time bucket
        if series:
            assert "time_bucket" in series[0]
            assert "avg_value" in series[0]

    def test_invalid_metric_raises(self, store):
        """Should raise ValueError for invalid metric names."""
        with pytest.raises(ValueError):
            store.get_time_series("drop_table_students")

    def test_query_by_time_range(self, store):
        """Should filter records by time range."""
        now = datetime.now(UTC)
        old_record = EvalRecord(
            app_name="test",
            query="old",
            response="r",
            timestamp=now - timedelta(days=7),
        )
        new_record = EvalRecord(
            app_name="test",
            query="new",
            response="r",
            timestamp=now,
        )
        store.save(old_record)
        store.save(new_record)

        results = store.query(
            filters={
                "start_time": now - timedelta(days=1),
            }
        )
        assert len(results) == 1
        assert results[0].query == "new"

    def test_duplicate_id_upsert(self, store):
        """Should update existing record on duplicate ID (INSERT OR REPLACE)."""
        record = EvalRecord(app_name="test", query="q1", response="r1", health_score=50)
        store.save(record)

        updated = record.model_copy(update={"health_score": 90})
        store.save(updated)

        assert store.count() == 1
        retrieved = store.get_by_id(record.id)
        assert retrieved.health_score == 90
