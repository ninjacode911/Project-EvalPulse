"""SQLite storage backend for EvalPulse."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evalpulse.models import EvalRecord
from evalpulse.storage.base import StorageBackend

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS eval_records (
    id              TEXT PRIMARY KEY,
    app_name        TEXT NOT NULL,
    timestamp       TEXT NOT NULL,
    query           TEXT NOT NULL,
    context         TEXT,
    response        TEXT NOT NULL,
    model_name      TEXT,
    latency_ms      INTEGER,
    tags            TEXT,
    hallucination_score  REAL,
    hallucination_method TEXT,
    flagged_claims       TEXT,
    embedding_vector     TEXT,
    drift_score          REAL,
    faithfulness_score   REAL,
    context_relevance    REAL,
    answer_relevancy     REAL,
    groundedness_score   REAL,
    sentiment_score      REAL,
    toxicity_score       REAL,
    response_length      INTEGER,
    language_detected    TEXT,
    is_denial            INTEGER,
    health_score         INTEGER,
    created_at           TEXT DEFAULT (datetime('now'))
);
"""

_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_app_timestamp ON eval_records(app_name, timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_health_score ON eval_records(health_score);",
    "CREATE INDEX IF NOT EXISTS idx_timestamp ON eval_records(timestamp);",
]

_CREATE_ALERTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS alerts (
    id              TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    severity        TEXT NOT NULL,
    metric          TEXT NOT NULL,
    value           REAL,
    threshold       REAL,
    message         TEXT,
    record_id       TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);
"""

_INSERT_SQL = """
INSERT OR REPLACE INTO eval_records (
    id, app_name, timestamp, query, context, response, model_name, latency_ms,
    tags, hallucination_score, hallucination_method, flagged_claims,
    embedding_vector, drift_score, faithfulness_score, context_relevance,
    answer_relevancy, groundedness_score, sentiment_score, toxicity_score,
    response_length, language_detected, is_denial, health_score
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# Map of valid metric column names for time-series queries
_VALID_METRICS = {
    "hallucination_score",
    "drift_score",
    "faithfulness_score",
    "context_relevance",
    "answer_relevancy",
    "groundedness_score",
    "sentiment_score",
    "toxicity_score",
    "health_score",
    "response_length",
    "latency_ms",
}


class SQLiteStore(StorageBackend):
    """SQLite-based storage backend using WAL mode for concurrent access."""

    def __init__(self, db_path: str = "evalpulse.db") -> None:
        self._db_path = str(Path(db_path).resolve())
        self._local = threading.local()
        self._lock = threading.Lock()
        self._all_connections: list[sqlite3.Connection] = []
        self._conn_lock = threading.Lock()
        # Initialize tables on the calling thread
        conn = self._get_connection()
        self._init_tables(conn)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            conn = sqlite3.connect(self._db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.row_factory = sqlite3.Row
            self._local.connection = conn
            with self._conn_lock:
                self._all_connections.append(conn)
        return self._local.connection

    def _init_tables(self, conn: sqlite3.Connection) -> None:
        """Create tables and indexes if they don't exist."""
        conn.execute(_CREATE_TABLE_SQL)
        for idx_sql in _CREATE_INDEXES_SQL:
            conn.execute(idx_sql)
        conn.execute(_CREATE_ALERTS_TABLE_SQL)
        conn.commit()

    def _record_to_row(self, record: EvalRecord) -> tuple:
        """Convert an EvalRecord to a tuple for SQL insertion."""
        return (
            record.id,
            record.app_name,
            record.timestamp.isoformat(),
            record.query,
            record.context,
            record.response,
            record.model_name,
            record.latency_ms,
            json.dumps(record.tags),
            record.hallucination_score,
            record.hallucination_method,
            json.dumps(record.flagged_claims),
            json.dumps(record.embedding_vector) if record.embedding_vector else "[]",
            record.drift_score,
            record.faithfulness_score,
            record.context_relevance,
            record.answer_relevancy,
            record.groundedness_score,
            record.sentiment_score,
            record.toxicity_score,
            record.response_length,
            record.language_detected,
            1 if record.is_denial else 0,
            record.health_score,
        )

    def _row_to_record(self, row: sqlite3.Row) -> EvalRecord:
        """Convert a database row to an EvalRecord."""
        data = dict(row)
        # Parse JSON fields
        data["tags"] = json.loads(data.get("tags") or "[]")
        data["flagged_claims"] = json.loads(data.get("flagged_claims") or "[]")
        data["embedding_vector"] = json.loads(data.get("embedding_vector") or "[]")
        data["is_denial"] = bool(data.get("is_denial", 0))
        # Parse timestamp
        ts = data.get("timestamp", "")
        if ts:
            try:
                data["timestamp"] = datetime.fromisoformat(ts)
            except ValueError:
                data["timestamp"] = datetime.now(UTC)
        # Remove created_at (not in model)
        data.pop("created_at", None)
        return EvalRecord.model_validate(data)

    def save(self, record: EvalRecord) -> None:
        """Save a single evaluation record."""
        conn = self._get_connection()
        with self._lock:
            conn.execute(_INSERT_SQL, self._record_to_row(record))
            conn.commit()

    def save_batch(self, records: list[EvalRecord]) -> None:
        """Save multiple evaluation records in a batch."""
        if not records:
            return
        conn = self._get_connection()
        rows = [self._record_to_row(r) for r in records]
        with self._lock:
            conn.executemany(_INSERT_SQL, rows)
            conn.commit()

    def query(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "timestamp",
        order_desc: bool = True,
    ) -> list[EvalRecord]:
        """Query evaluation records with optional filters."""
        conn = self._get_connection()
        where_clauses = []
        params: list = []

        if filters:
            for key, value in filters.items():
                if key in ("app_name", "model_name", "language_detected"):
                    where_clauses.append(f"{key} = ?")
                    params.append(value)
                elif key == "min_health_score":
                    where_clauses.append("health_score >= ?")
                    params.append(value)
                elif key == "max_health_score":
                    where_clauses.append("health_score <= ?")
                    params.append(value)
                elif key == "start_time":
                    where_clauses.append("timestamp >= ?")
                    params.append(value.isoformat() if isinstance(value, datetime) else value)
                elif key == "end_time":
                    where_clauses.append("timestamp <= ?")
                    params.append(value.isoformat() if isinstance(value, datetime) else value)

        sql = "SELECT * FROM eval_records"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        # Validate order_by to prevent SQL injection
        valid_columns = {
            "timestamp",
            "health_score",
            "app_name",
            "hallucination_score",
            "drift_score",
            "created_at",
        }
        if order_by not in valid_columns:
            order_by = "timestamp"
        direction = "DESC" if order_desc else "ASC"
        sql += f" ORDER BY {order_by} {direction}"
        sql += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor = conn.execute(sql, params)
        return [self._row_to_record(row) for row in cursor.fetchall()]

    def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count evaluation records matching filters."""
        conn = self._get_connection()
        where_clauses = []
        params: list = []

        if filters:
            if "app_name" in filters:
                where_clauses.append("app_name = ?")
                params.append(filters["app_name"])

        sql = "SELECT COUNT(*) FROM eval_records"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)

        cursor = conn.execute(sql, params)
        return cursor.fetchone()[0]

    def get_by_id(self, record_id: str) -> EvalRecord | None:
        """Get a single record by ID."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM eval_records WHERE id = ?", (record_id,))
        row = cursor.fetchone()
        return self._row_to_record(row) if row else None

    def get_latest(self, n: int = 10, app_name: str | None = None) -> list[EvalRecord]:
        """Get the N most recent records."""
        conn = self._get_connection()
        if app_name:
            cursor = conn.execute(
                "SELECT * FROM eval_records WHERE app_name = ? ORDER BY timestamp DESC LIMIT ?",
                (app_name, n),
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM eval_records ORDER BY timestamp DESC LIMIT ?", (n,)
            )
        return [self._row_to_record(row) for row in cursor.fetchall()]

    def get_time_series(
        self,
        metric: str,
        start: datetime | None = None,
        end: datetime | None = None,
        app_name: str | None = None,
        granularity: str = "hour",
    ) -> list[dict[str, Any]]:
        """Get time-series data for a specific metric aggregated by time bucket."""
        if metric not in _VALID_METRICS:
            raise ValueError(f"Invalid metric: {metric}. Must be one of {_VALID_METRICS}")

        conn = self._get_connection()
        # SQLite time bucket formatting
        time_formats = {
            "minute": "%Y-%m-%d %H:%M",
            "hour": "%Y-%m-%d %H:00",
            "day": "%Y-%m-%d",
        }
        time_fmt = time_formats.get(granularity, "%Y-%m-%d %H:00")

        where_clauses = [f"{metric} IS NOT NULL"]
        params: list = []

        if start:
            where_clauses.append("timestamp >= ?")
            params.append(start.isoformat())
        if end:
            where_clauses.append("timestamp <= ?")
            params.append(end.isoformat())
        if app_name:
            where_clauses.append("app_name = ?")
            params.append(app_name)

        where_sql = " AND ".join(where_clauses)
        sql = f"""
            SELECT strftime('{time_fmt}', timestamp) as time_bucket,
                   AVG({metric}) as avg_value,
                   MIN({metric}) as min_value,
                   MAX({metric}) as max_value,
                   COUNT(*) as count
            FROM eval_records
            WHERE {where_sql}
            GROUP BY time_bucket
            ORDER BY time_bucket ASC
        """

        cursor = conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]

    def save_alert(self, alert_data: dict) -> None:
        """Save a single alert record (thread-safe)."""
        conn = self._get_connection()
        with self._lock:
            conn.execute(
                """INSERT OR REPLACE INTO alerts
                (id, timestamp, severity, metric, value,
                 threshold, message, record_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    alert_data["id"],
                    alert_data["timestamp"],
                    alert_data["severity"],
                    alert_data["metric"],
                    alert_data["value"],
                    alert_data["threshold"],
                    alert_data["message"],
                    alert_data["record_id"],
                ),
            )
            conn.commit()

    def save_alerts_batch(self, alerts: list[dict]) -> None:
        """Save multiple alert records in a batch (thread-safe)."""
        if not alerts:
            return
        conn = self._get_connection()
        with self._lock:
            for alert_data in alerts:
                conn.execute(
                    """INSERT OR REPLACE INTO alerts
                    (id, timestamp, severity, metric, value,
                     threshold, message, record_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        alert_data["id"],
                        alert_data["timestamp"],
                        alert_data["severity"],
                        alert_data["metric"],
                        alert_data["value"],
                        alert_data["threshold"],
                        alert_data["message"],
                        alert_data["record_id"],
                    ),
                )
            conn.commit()

    def get_recent_alerts(self, n: int = 20) -> list[dict]:
        """Get the N most recent alerts from storage (thread-safe)."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?",
            (n,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def close(self) -> None:
        """Close the storage backend and all thread-local connections."""
        with self._conn_lock:
            for conn in self._all_connections:
                try:
                    conn.close()
                except Exception:
                    pass
            self._all_connections.clear()
        # Also clear the calling thread's local reference
        if hasattr(self._local, "connection"):
            self._local.connection = None
