# Week 7: Alerts + Notifications + Dashboard Polish

## Overview

Week 7 built the alert engine and notification system. The alert engine checks every evaluation record against configurable thresholds, fires alerts with cooldown-based deduplication, classifies severity, and dispatches notifications via email and Slack. This is the operational layer that turns passive monitoring into active incident response.

## What Was Built

### 1. Alert Engine (`evalpulse/alerts.py`)

**What**: `AlertEngine` checks evaluation records against configurable thresholds and fires alerts when metrics exceed acceptable ranges.

**Thresholds checked**:

| Metric | Default Threshold | Direction | What It Catches |
|--------|------------------|-----------|-----------------|
| hallucination_score | > 0.3 | Above = bad | LLM generating content not grounded in input/context |
| drift_score | > 0.15 | Above = bad | Model behavior shifting from baseline distribution |
| groundedness_score | < 0.65 | Below = bad | RAG responses not supported by retrieved context |
| toxicity_score | > 0.05 | Above = bad | Harmful, offensive, or inappropriate content |
| regression_fail_rate | > 0.10 | Above = bad | More than 10% of golden dataset tests failing |

**How threshold checking works**: For each evaluation record, the engine iterates through all configured thresholds. Each threshold defines a metric name, a limit value, and a direction (above or below). The engine reads the metric value from the record, compares it against the limit in the specified direction, and creates an Alert if the threshold is violated.

**Cooldown deduplication**: The same metric will not fire again within `alert_cooldown_seconds` (default: 300 seconds / 5 minutes). This prevents alert storms when a batch of bad responses comes through in quick succession.

Implementation: The engine maintains a dictionary mapping metric names to their last alert timestamp, using `time.monotonic()`. When a threshold is violated, the engine checks if `(current_time - last_alert_time) >= cooldown_seconds` before firing. `time.monotonic()` is used instead of `time.time()` because it is unaffected by system clock adjustments (NTP syncs, daylight saving changes, manual clock changes).

**Severity classification**: When an alert fires, it is classified as either Warning or Critical based on how far the metric value exceeds the threshold:

- **Warning**: The value exceeds the threshold by less than 0.2. Example: hallucination_score = 0.35 with threshold 0.3 (excess = 0.05 < 0.2).
- **Critical**: The value exceeds the threshold by 0.2 or more. Example: hallucination_score = 0.55 with threshold 0.3 (excess = 0.25 >= 0.2).

The 0.2 cutoff was chosen as a reasonable default — it roughly corresponds to a metric being "somewhat over" vs "significantly over" the acceptable range. This is configurable.

**`Alert` Pydantic model**:
- `id` (str): UUID for unique identification.
- `timestamp` (str): ISO 8601 timestamp of when the alert was created.
- `severity` (str): "warning" or "critical".
- `metric` (str): Which metric triggered the alert (e.g., "hallucination_score").
- `value` (float): The actual metric value that triggered the alert.
- `threshold` (float): The threshold value that was exceeded.
- `message` (str): Human-readable alert message (e.g., "hallucination_score = 0.45 exceeds threshold 0.3").
- `record_id` (str): The evaluation record ID that triggered this alert, for traceability.

**SQLite storage**: Alerts are stored in an `alerts` table in the same SQLite database as evaluation records. Schema: `id TEXT PRIMARY KEY, timestamp TEXT, severity TEXT, metric TEXT, value REAL, threshold REAL, message TEXT, record_id TEXT`. This means alerts can be queried alongside evaluation records without additional infrastructure.

### 2. Notification Dispatcher (`evalpulse/notifications.py`)

**What**: `NotificationDispatcher` sends alerts to configured notification channels (email via SMTP, Slack via webhook).

**Design principle**: All notification sends are fire-and-forget in background daemon threads. A failed notification (SMTP timeout, Slack webhook down, invalid credentials) is logged as a warning but never crashes the worker or blocks evaluation processing. The evaluation pipeline is the critical path; notifications are best-effort.

**Channels**:

#### Email (SMTP)
- Configured via `evalpulse.yml`: `smtp_host`, `smtp_port`, `smtp_user`, `smtp_password`, `alert_email_to`.
- Formats the alert as a plain-text email with: severity, metric name, metric value, threshold, human-readable message, timestamp, and record ID.
- Subject line includes severity and metric for quick scanning in inbox (e.g., "[CRITICAL] EvalPulse: hallucination_score alert").
- Uses Python's `smtplib` with STARTTLS. Compatible with Gmail (smtp.gmail.com:587), AWS SES, and standard SMTP servers.
- Connection is opened and closed per alert (no persistent connection). This is simpler and more reliable than maintaining a connection pool for what is expected to be low-volume alert traffic.

#### Slack (Webhook)
- Configured via `evalpulse.yml`: `slack_webhook_url`.
- Posts a formatted message to the webhook URL using `httpx.post()`.
- Message includes severity emoji (warning/critical indicators), metric name, value, threshold, and timestamp.
- Uses `httpx` (already a project dependency for async HTTP) rather than the official Slack SDK to avoid adding another dependency for a single POST request.

**Thread model**: When an alert is dispatched, the dispatcher spawns a daemon thread for each configured channel. Daemon threads are used so that if the main process exits, pending notifications do not prevent shutdown. The thread calls the channel-specific send function, catches all exceptions, and logs failures.

### 3. Worker Alert Pipeline

The worker (`evalpulse/worker.py`) was modified to integrate the alert system. After every batch of evaluation records is saved to SQLite, the worker runs `_check_alerts()`.

**Flow**:

1. Worker saves a batch of evaluation records to SQLite (existing behavior).
2. `_check_alerts()` is called with the batch of saved records.
3. Creates an `AlertEngine` instance with thresholds from the config.
4. For each record in the batch, calls `alert_engine.check(record)`.
5. The AlertEngine returns a list of `Alert` objects (possibly empty if no thresholds are violated or cooldown is active).
6. Each triggered Alert is saved to the SQLite `alerts` table.
7. Each triggered Alert is passed to `NotificationDispatcher.dispatch(alert)`, which spawns background threads for email/Slack.
8. Worker continues to process the next batch.

**Why check after save, not before**: The evaluation record must be persisted before alerting. If the worker crashes after sending a notification but before saving the record, the alert would reference a record that does not exist in the database. Saving first ensures data consistency.

**Why create AlertEngine per batch**: The AlertEngine is lightweight (just threshold config and a cooldown dictionary). Creating a fresh one per batch means cooldown state is not preserved across worker restarts. This is an acceptable tradeoff: the alternative (persisting cooldown state to SQLite) adds complexity for marginal benefit. In practice, cooldown matters within a burst of bad records, and those bursts happen within a single worker lifetime.

## Test Results

**Alert tests** (12 unit tests):
- Clean record (all metrics within thresholds) produces no alerts
- Individual metric violations (one test per metric: hallucination, drift, groundedness, toxicity, regression fail rate)
- None metric handling (metrics that are None, e.g., RAG metrics on non-RAG calls, do not trigger alerts)
- Cooldown deduplication (same metric violated twice within cooldown window only fires once)
- Cooldown expiry (same metric violated after cooldown window fires again)
- Severity classification (warning vs critical based on excess amount)
- Message format (human-readable message contains metric name, value, and threshold)
- Multiple violations (single record can trigger multiple alerts if multiple metrics are violated)
- Alert model serialization (Pydantic model serializes to/from JSON correctly)

**Notification tests** (3 unit tests):
- Dispatcher creation with and without configured channels
- Email formatting produces expected subject and body
- Dispatch with no channels configured does not crash

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| Cooldown via `time.monotonic()` | Prevents alert storms from bursts of bad records; monotonic clock is immune to system time changes (NTP, DST, manual adjustments) |
| Fire-and-forget daemon threads for notifications | Failed Slack/email must never block evaluation processing; daemon threads auto-terminate on process exit |
| Fresh AlertEngine per batch | Stateless design simplifies reasoning; cooldown resets on worker restart is an acceptable tradeoff vs persistence complexity |
| SQLite alerts table | Same database as evaluation records; zero additional infrastructure; enables joined queries between alerts and records |
| 0.2 excess threshold for severity classification | Simple, interpretable heuristic; "somewhat over" vs "significantly over" the acceptable range |
| `httpx.post()` for Slack instead of Slack SDK | Avoids adding a dependency for a single HTTP POST; httpx is already in the dependency tree for async HTTP |
| Per-alert email connections (no pool) | Alert volume is expected to be low (minutes between alerts due to cooldown); connection pooling complexity not justified |

## Files Created

| File | Purpose |
|------|---------|
| `evalpulse/alerts.py` | AlertEngine (threshold checking, cooldown, severity) + Alert Pydantic model |
| `evalpulse/notifications.py` | NotificationDispatcher with email (SMTP) and Slack (webhook) channels |
| `evalpulse/worker.py` (modified) | Added `_check_alerts()` integration after record saving |
| `tests/unit/test_alerts.py` | 12 unit tests for alert engine |
| `tests/unit/test_notifications.py` | 3 unit tests for notification dispatcher |

## Interview Talking Points

- **Why not use a dedicated alerting system like PagerDuty or OpsGenie?** EvalPulse's built-in alerting is designed for self-contained deployment. Adding PagerDuty integration would require API keys, account setup, and a dependency on an external service. The built-in email/Slack channels cover the most common notification needs. PagerDuty/OpsGenie integration could be added as an additional channel without changing the AlertEngine architecture.

- **How would you handle alert fatigue?** The cooldown mechanism is the first line of defense. Beyond that, you could add: (1) alert grouping (combine multiple metric violations from the same time window into a single notification), (2) escalation policies (only page on-call for critical alerts, send warnings to a low-priority Slack channel), (3) auto-resolve (send a "resolved" notification when a metric returns to normal).

- **Why not use a message queue (Redis, RabbitMQ) for notifications?** For the current scale (alerts per minute, not per second), background threads are simpler and require no additional infrastructure. A message queue would be appropriate if EvalPulse needed guaranteed delivery, retry with backoff, or was processing thousands of alerts per second.

- **How does cooldown interact with multiple worker instances?** Currently, cooldown is per-worker-instance (in-memory dictionary). If multiple workers are running, each maintains its own cooldown state, and duplicate alerts can occur. For multi-worker deployments, cooldown state would need to move to a shared store (Redis or the SQLite database with atomic check-and-set).
