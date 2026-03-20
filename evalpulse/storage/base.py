"""Abstract base class for EvalPulse storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from evalpulse.models import EvalRecord


class StorageBackend(ABC):
    """Abstract storage backend for EvalPulse evaluation records."""

    @abstractmethod
    def save(self, record: EvalRecord) -> None:
        """Save a single evaluation record."""

    @abstractmethod
    def save_batch(self, records: list[EvalRecord]) -> None:
        """Save multiple evaluation records in a batch."""

    @abstractmethod
    def query(
        self,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "timestamp",
        order_desc: bool = True,
    ) -> list[EvalRecord]:
        """Query evaluation records with optional filters."""

    @abstractmethod
    def count(self, filters: dict[str, Any] | None = None) -> int:
        """Count evaluation records matching filters."""

    @abstractmethod
    def get_by_id(self, record_id: str) -> EvalRecord | None:
        """Get a single record by ID."""

    @abstractmethod
    def get_latest(self, n: int = 10, app_name: str | None = None) -> list[EvalRecord]:
        """Get the N most recent records."""

    @abstractmethod
    def get_time_series(
        self,
        metric: str,
        start: datetime | None = None,
        end: datetime | None = None,
        app_name: str | None = None,
        granularity: str = "hour",
    ) -> list[dict[str, Any]]:
        """Get time-series data for a specific metric."""

    @abstractmethod
    def save_alerts_batch(self, alerts: list[dict]) -> None:
        """Save multiple alert records in a batch."""

    @abstractmethod
    def get_recent_alerts(self, n: int = 20) -> list[dict]:
        """Get the N most recent alerts from storage."""

    @abstractmethod
    def close(self) -> None:
        """Close the storage backend and release resources."""
