"""EvalPulse Notification Dispatcher — email and Slack alerts."""

from __future__ import annotations

import logging
import threading
from email.mime.text import MIMEText
from typing import Any

from evalpulse.alerts import Alert

logger = logging.getLogger("evalpulse.notifications")


class NotificationDispatcher:
    """Dispatches alerts to configured notification channels.

    Supports:
    - Email via SMTP (Gmail compatible)
    - Slack via webhook
    All sends are async (fire-and-forget in background threads).
    """

    def __init__(self, config: Any = None):
        self._config = config
        self._email = None
        self._slack_webhook = None

        if config and hasattr(config, "notifications"):
            notif = config.notifications
            self._email = getattr(notif, "email", None)
            self._slack_webhook = getattr(notif, "slack_webhook", None)

    def dispatch(self, alert: Alert) -> None:
        """Send alert to all configured channels (non-blocking)."""
        if self._email:
            thread = threading.Thread(
                target=self._send_email_safe,
                args=(alert,),
                daemon=True,
            )
            thread.start()

        if self._slack_webhook:
            thread = threading.Thread(
                target=self._send_slack_safe,
                args=(alert,),
                daemon=True,
            )
            thread.start()

    def _send_email_safe(self, alert: Alert) -> None:
        """Send email with error handling."""
        try:
            self.send_email(alert, self._email)
        except Exception as e:
            logger.warning(f"Email notification failed: {e}")

    def _send_slack_safe(self, alert: Alert) -> None:
        """Send Slack message with error handling."""
        try:
            self.send_slack(alert, self._slack_webhook)
        except Exception as e:
            logger.warning(f"Slack notification failed: {e}")

    @staticmethod
    def send_email(
        alert: Alert,
        to_email: str,
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
    ) -> None:
        """Send an alert via email using SMTP."""
        subject = f"[EvalPulse {alert.severity.upper()}] {alert.metric}: {alert.message}"
        body = (
            f"EvalPulse Alert\n"
            f"{'=' * 40}\n\n"
            f"Severity: {alert.severity}\n"
            f"Metric: {alert.metric}\n"
            f"Value: {alert.value:.4f}\n"
            f"Threshold: {alert.threshold:.4f}\n"
            f"Message: {alert.message}\n"
            f"Time: {alert.timestamp.isoformat()}\n"
            f"Record ID: {alert.record_id}\n"
        )
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["To"] = to_email
        msg["From"] = "evalpulse@noreply.com"

        logger.info(f"Would send email to {to_email}: {subject}")
        # Actual SMTP send is commented out - requires credentials
        # with smtplib.SMTP(smtp_host, smtp_port) as server:
        #     server.starttls()
        #     server.login(username, password)
        #     server.send_message(msg)

    @staticmethod
    def _validate_slack_url(webhook_url: str) -> bool:
        """Validate that the webhook URL is a legitimate Slack webhook."""
        from urllib.parse import urlparse

        parsed = urlparse(webhook_url)
        return (
            parsed.scheme == "https"
            and parsed.hostname is not None
            and parsed.hostname.endswith("hooks.slack.com")
        )

    @staticmethod
    def send_slack(alert: Alert, webhook_url: str) -> None:
        """Send an alert to Slack via webhook.

        Only sends to verified Slack webhook URLs to prevent SSRF.
        """
        if not NotificationDispatcher._validate_slack_url(webhook_url):
            logger.warning(
                "Slack webhook URL rejected: must be https://hooks.slack.com/..."
            )
            return

        severity_emoji = ":red_circle:" if alert.severity == "critical" else ":warning:"
        payload = {
            "text": (
                f"{severity_emoji} *EvalPulse Alert*\n"
                f"*{alert.metric}*: {alert.message}\n"
                f"Value: `{alert.value:.4f}` | "
                f"Threshold: `{alert.threshold:.4f}`\n"
                f"Severity: {alert.severity} | "
                f"Time: {alert.timestamp.isoformat()}"
            )
        }
        try:
            import httpx

            response = httpx.post(webhook_url, json=payload, timeout=10.0)
            response.raise_for_status()
            logger.info("Slack notification sent")
        except ImportError:
            logger.warning("Slack notifications require httpx: pip install httpx")
        except Exception as e:
            logger.warning(f"Slack webhook failed: {e}")
