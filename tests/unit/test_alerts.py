"""Tests for the Alert Engine."""

import pytest

from evalpulse.alerts import Alert, AlertEngine
from evalpulse.models import EvalRecord


@pytest.fixture
def engine():
    return AlertEngine()


class TestAlertEngine:
    """Test alert threshold checking."""

    def test_no_alerts_on_good_record(self, engine):
        """Clean record should produce no alerts."""
        record = EvalRecord(
            hallucination_score=0.1,
            toxicity_score=0.01,
            drift_score=0.05,
            groundedness_score=0.8,
        )
        alerts = engine.check(record)
        assert len(alerts) == 0

    def test_hallucination_alert(self, engine):
        """High hallucination should trigger alert."""
        record = EvalRecord(hallucination_score=0.5)
        alerts = engine.check(record)
        assert len(alerts) >= 1
        assert any(a.metric == "hallucination_score" for a in alerts)

    def test_toxicity_alert(self, engine):
        """High toxicity should trigger alert."""
        record = EvalRecord(toxicity_score=0.3)
        alerts = engine.check(record)
        assert any(a.metric == "toxicity_score" for a in alerts)

    def test_drift_alert(self, engine):
        """High drift should trigger alert."""
        record = EvalRecord(drift_score=0.3)
        alerts = engine.check(record)
        assert any(a.metric == "drift_score" for a in alerts)

    def test_groundedness_alert(self, engine):
        """Low groundedness should trigger alert."""
        record = EvalRecord(groundedness_score=0.4)
        alerts = engine.check(record)
        assert any(a.metric == "groundedness_score" for a in alerts)

    def test_none_drift_no_alert(self, engine):
        """None drift score should not trigger alert."""
        record = EvalRecord(drift_score=None)
        alerts = engine.check(record)
        assert not any(a.metric == "drift_score" for a in alerts)

    def test_cooldown_deduplication(self):
        """Same metric shouldn't fire twice within cooldown."""
        engine = AlertEngine()
        engine._cooldown = 10  # 10 seconds

        record = EvalRecord(hallucination_score=0.5)
        alerts1 = engine.check(record)
        alerts2 = engine.check(record)

        assert len(alerts1) >= 1
        assert len(alerts2) == 0  # Suppressed by cooldown

    def test_severity_classification_warning(self, engine):
        """Slightly above threshold should be warning."""
        record = EvalRecord(hallucination_score=0.35)
        alerts = engine.check(record)
        halluc_alerts = [a for a in alerts if a.metric == "hallucination_score"]
        if halluc_alerts:
            assert halluc_alerts[0].severity == "warning"

    def test_severity_classification_critical(self, engine):
        """Far above threshold should be critical."""
        record = EvalRecord(hallucination_score=0.8)
        alerts = engine.check(record)
        halluc_alerts = [a for a in alerts if a.metric == "hallucination_score"]
        if halluc_alerts:
            assert halluc_alerts[0].severity == "critical"

    def test_alert_message_format(self, engine):
        """Alert message should include metric details."""
        record = EvalRecord(hallucination_score=0.5)
        alerts = engine.check(record)
        assert len(alerts) >= 1
        assert "hallucination_score" in alerts[0].message
        assert "above" in alerts[0].message

    def test_multiple_violations(self, engine):
        """Record violating multiple thresholds should fire multiple alerts."""
        record = EvalRecord(
            hallucination_score=0.5,
            toxicity_score=0.3,
            drift_score=0.3,
        )
        alerts = engine.check(record)
        metrics = {a.metric for a in alerts}
        assert "hallucination_score" in metrics
        assert "toxicity_score" in metrics
        assert "drift_score" in metrics

    def test_alert_model(self):
        """Alert model should serialize properly."""
        alert = Alert(
            severity="critical",
            metric="hallucination_score",
            value=0.5,
            threshold=0.3,
            message="test",
        )
        data = alert.model_dump()
        restored = Alert.model_validate(data)
        assert restored.severity == "critical"
        assert restored.value == 0.5
