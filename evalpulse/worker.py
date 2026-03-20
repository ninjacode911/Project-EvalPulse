"""Background EvaluationWorker — consumes events from the queue and processes them."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any

from evalpulse.models import EvalEvent, EvalRecord

logger = logging.getLogger("evalpulse.worker")


class EvaluationWorker:
    """Background worker that processes EvalEvents through evaluation modules.

    Runs in a daemon thread. Reads events from the queue, dispatches them
    to registered evaluation modules, and saves the results to storage.

    Supports batch writing: collects up to `batch_size` events or waits
    up to `batch_timeout` seconds before flushing to storage.
    """

    def __init__(
        self,
        event_queue: queue.Queue,
        config: Any = None,
        batch_size: int = 10,
        batch_timeout: float = 1.0,
    ):
        self._queue = event_queue
        self._config = config
        self._batch_size = batch_size
        self._batch_timeout = batch_timeout
        self._modules: list = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False
        self._alert_engine = None
        self._notification_dispatcher = None

    def register_module(self, module: Any) -> None:
        """Register an evaluation module to process events."""
        self._modules.append(module)
        logger.info(f"Registered module: {module.name}")

    def _register_default_modules(self) -> None:
        """Auto-register evaluation modules based on config."""
        if self._modules or self._config is None:
            return  # Already registered, or no config provided
        try:
            from evalpulse.modules import get_default_modules

            for module in get_default_modules(self._config):
                self.register_module(module)
        except Exception as e:
            logger.warning(f"Failed to register default modules: {e}")

    def start(self) -> None:
        """Start the background worker thread."""
        if self._started:
            return
        self._register_default_modules()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="evalpulse-worker",
            daemon=True,
        )
        self._thread.start()
        self._started = True
        logger.info("EvaluationWorker started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the worker, draining remaining events first."""
        if not self._started:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._started = False
        logger.info("EvaluationWorker stopped")

    def _run(self) -> None:
        """Main worker loop: collect events in batches and process them."""
        batch: list[EvalEvent] = []
        last_flush = time.monotonic()

        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=0.1)
                batch.append(event)
            except queue.Empty:
                pass

            # Flush if batch is full or timeout reached
            now = time.monotonic()
            if batch and (
                len(batch) >= self._batch_size or (now - last_flush) >= self._batch_timeout
            ):
                self._process_batch(batch)
                batch = []
                last_flush = now

        # Drain remaining events on shutdown
        while True:
            try:
                event = self._queue.get_nowait()
                batch.append(event)
            except queue.Empty:
                break

        if batch:
            self._process_batch(batch)

    def _process_batch(self, events: list[EvalEvent]) -> None:
        """Process a batch of events through modules and save to storage."""
        records = []
        for event in events:
            try:
                record = self._evaluate_event(event)
                records.append(record)
            except Exception as e:
                logger.error(f"Error evaluating event {event.id}: {e}")
                # Still save the record with default scores
                records.append(EvalRecord.from_event(event))

        if records:
            self._save_records(records)

    def _evaluate_event(self, event: EvalEvent) -> EvalRecord:
        """Run all registered modules on a single event."""
        record = EvalRecord.from_event(event)

        for module in self._modules:
            try:
                if not module.is_available():
                    continue
                result = module.evaluate_sync(event)
                if result:
                    # Merge module results into the record
                    for key, value in result.items():
                        if hasattr(record, key):
                            # Clamp float scores to valid range
                            # to prevent Pydantic validation errors
                            # from floating-point imprecision
                            if isinstance(value, float) and key.endswith("_score"):
                                value = max(0.0, min(1.0, value))
                            setattr(record, key, value)
            except Exception as e:
                logger.warning(f"Module {module.name} failed for event {event.id}: {e}")

        # Compute health score after all modules have run
        try:
            from evalpulse.health_score import compute_health_score

            record.health_score = compute_health_score(record)
        except Exception as e:
            logger.warning(f"Health score computation failed: {e}")

        return record

    def _save_records(self, records: list[EvalRecord]) -> None:
        """Save processed records to storage and check alerts."""
        try:
            from evalpulse.storage import get_storage

            storage = get_storage()
            storage.save_batch(records)
            logger.debug(f"Saved batch of {len(records)} records")
        except Exception as e:
            logger.error(f"Failed to save batch: {e}")

        # Check alerts for each record
        self._check_alerts(records)

    def _get_alert_engine(self):
        """Get or create the alert engine singleton for this worker."""
        if self._alert_engine is None:
            from evalpulse.alerts import AlertEngine

            self._alert_engine = AlertEngine(self._config)
        return self._alert_engine

    def _get_notification_dispatcher(self):
        """Get or create the notification dispatcher for this worker."""
        if self._notification_dispatcher is None:
            from evalpulse.notifications import NotificationDispatcher

            self._notification_dispatcher = NotificationDispatcher(self._config)
        return self._notification_dispatcher

    def _check_alerts(self, records: list[EvalRecord]) -> None:
        """Run alert engine on records and dispatch notifications."""
        try:
            engine = self._get_alert_engine()
            all_alerts = []
            for record in records:
                alerts = engine.check(record)
                all_alerts.extend(alerts)

            if all_alerts:
                engine.save_alerts(all_alerts)
                logger.info(f"Fired {len(all_alerts)} alert(s)")

                try:
                    dispatcher = self._get_notification_dispatcher()
                    for alert in all_alerts:
                        dispatcher.dispatch(alert)
                except Exception as e:
                    logger.warning(f"Notification dispatch failed: {e}")
        except Exception as e:
            logger.warning(f"Alert check failed: {e}")

    @property
    def is_running(self) -> bool:
        """Check if the worker is currently running."""
        return self._started and self._thread is not None and self._thread.is_alive()

    @property
    def pending_count(self) -> int:
        """Get the number of events waiting in the queue."""
        return self._queue.qsize()
