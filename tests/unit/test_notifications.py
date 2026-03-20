"""Tests for the Notification Dispatcher."""

from evalpulse.alerts import Alert
from evalpulse.notifications import NotificationDispatcher


class TestNotificationDispatcher:
    """Test notification dispatch."""

    def test_dispatcher_creation(self):
        """Should create dispatcher without config."""
        dispatcher = NotificationDispatcher()
        assert dispatcher._email is None
        assert dispatcher._slack_webhook is None

    def test_email_format(self):
        """Email notification should format correctly."""
        alert = Alert(
            severity="critical",
            metric="hallucination_score",
            value=0.5,
            threshold=0.3,
            message="test alert",
            record_id="test-id",
        )
        # send_email is a static method we can test formatting
        # (actual SMTP is not called)
        NotificationDispatcher.send_email(alert, "test@test.com")
        # Should not raise

    def test_dispatch_no_channels(self):
        """Dispatch with no channels configured should not error."""
        dispatcher = NotificationDispatcher()
        alert = Alert(
            severity="warning",
            metric="drift_score",
            value=0.2,
            threshold=0.15,
            message="test",
        )
        dispatcher.dispatch(alert)
        # Should not raise
