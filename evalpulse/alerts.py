"""EvalPulse Alert Engine — threshold checking and alert management."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from evalpulse.models import EvalRecord

logger = logging.getLogger("evalpulse.alerts")


class Alert(BaseModel):
    """A single alert triggered by a threshold violation."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    severity: str = "warning"  # "warning" or "critical"
    metric: str = ""
    value: float = 0.0
    threshold: float = 0.0
    message: str = ""
    record_id: str = ""


class AlertEngine:
    """Checks evaluation records against configured thresholds.

    Features:
    - Configurable thresholds per metric
    - Cooldown-based deduplication (same metric won't fire twice
      within cooldown period)
    - Severity classification (warning vs critical)
    - Alert persistence via storage backend
    """

    def __init__(self, config: Any = None):
        self._config = config
        self._last_fired: dict[str, float] = {}

        # Load thresholds from config or use defaults
        if config and hasattr(config, "thresholds"):
            t = config.thresholds
            self._thresholds = {
                "hallucination": getattr(t, "hallucination", 0.3),
                "drift": getattr(t, "drift", 0.15),
                "rag_groundedness_min": getattr(t, "rag_groundedness_min", 0.65),
                "toxicity": getattr(t, "toxicity", 0.05),
                "regression_fail_rate": getattr(t, "regression_fail_rate", 0.10),
            }
        else:
            self._thresholds = {
                "hallucination": 0.3,
                "drift": 0.15,
                "rag_groundedness_min": 0.65,
                "toxicity": 0.05,
                "regression_fail_rate": 0.10,
            }

        cooldown = 300  # 5 minutes default
        if config and hasattr(config, "alert_cooldown_seconds"):
            cooldown = config.alert_cooldown_seconds
        self._cooldown = cooldown

    def check(self, record: EvalRecord) -> list[Alert]:
        """Check a record against all thresholds.

        Returns a list of triggered alerts (may be empty).
        Respects cooldown deduplication.
        """
        alerts: list[Alert] = []
        now = time.monotonic()

        # Check hallucination
        if record.hallucination_score > self._thresholds["hallucination"]:
            alert = self._maybe_fire(
                now,
                metric="hallucination_score",
                value=record.hallucination_score,
                threshold=self._thresholds["hallucination"],
                record_id=record.id,
            )
            if alert:
                alerts.append(alert)

        # Check drift
        if record.drift_score is not None and record.drift_score > self._thresholds["drift"]:
            alert = self._maybe_fire(
                now,
                metric="drift_score",
                value=record.drift_score,
                threshold=self._thresholds["drift"],
                record_id=record.id,
            )
            if alert:
                alerts.append(alert)

        # Check RAG groundedness (below minimum)
        if (
            record.groundedness_score is not None
            and record.groundedness_score < self._thresholds["rag_groundedness_min"]
        ):
            alert = self._maybe_fire(
                now,
                metric="groundedness_score",
                value=record.groundedness_score,
                threshold=self._thresholds["rag_groundedness_min"],
                record_id=record.id,
                is_below=True,
            )
            if alert:
                alerts.append(alert)

        # Check toxicity
        if record.toxicity_score > self._thresholds["toxicity"]:
            alert = self._maybe_fire(
                now,
                metric="toxicity_score",
                value=record.toxicity_score,
                threshold=self._thresholds["toxicity"],
                record_id=record.id,
            )
            if alert:
                alerts.append(alert)

        return alerts

    def _maybe_fire(
        self,
        now: float,
        metric: str,
        value: float,
        threshold: float,
        record_id: str,
        is_below: bool = False,
    ) -> Alert | None:
        """Fire an alert if cooldown has elapsed."""
        last = self._last_fired.get(metric, 0)
        if (now - last) < self._cooldown:
            return None

        self._last_fired[metric] = now

        # Determine severity
        if is_below:
            diff = threshold - value
            severity = "critical" if diff > 0.2 else "warning"
            direction = "below"
        else:
            diff = value - threshold
            severity = "critical" if diff > 0.2 else "warning"
            direction = "above"

        message = (
            f"{metric} is {direction} threshold: "
            f"{value:.3f} {'<' if is_below else '>'} {threshold:.3f}"
        )

        return Alert(
            severity=severity,
            metric=metric,
            value=value,
            threshold=threshold,
            message=message,
            record_id=record_id,
        )

    def save_alerts(self, alerts: list[Alert]) -> None:
        """Persist alerts to storage via the thread-safe storage API."""
        try:
            from evalpulse.storage import get_storage

            storage = get_storage()
            alert_dicts = [
                {
                    "id": alert.id,
                    "timestamp": alert.timestamp.isoformat(),
                    "severity": alert.severity,
                    "metric": alert.metric,
                    "value": alert.value,
                    "threshold": alert.threshold,
                    "message": alert.message,
                    "record_id": alert.record_id,
                }
                for alert in alerts
            ]
            storage.save_alerts_batch(alert_dicts)
        except Exception as e:
            logger.warning(f"Failed to save alerts: {e}")

    def get_recent_alerts(self, n: int = 20) -> list[Alert]:
        """Get the N most recent alerts from storage."""
        try:
            from evalpulse.storage import get_storage

            storage = get_storage()
            rows = storage.get_recent_alerts(n)
            alerts = []
            for d in rows:
                ts = d.get("timestamp", "")
                if isinstance(ts, str):
                    d["timestamp"] = datetime.fromisoformat(ts)
                d.pop("created_at", None)
                alerts.append(Alert.model_validate(d))
            return alerts
        except Exception as e:
            logger.warning(f"Failed to get alerts: {e}")
            return []
