"""EvalPulse SDK — @track decorator, @atrack async decorator, EvalContext context manager."""

from __future__ import annotations

import functools
import logging
import queue
import threading
import time
from collections.abc import Callable
from contextvars import ContextVar

from evalpulse.models import EvalEvent

logger = logging.getLogger("evalpulse.sdk")

# Context variables for EvalContext scope propagation
_ctx_app_name: ContextVar[str | None] = ContextVar("evalpulse_app_name", default=None)
_ctx_model_name: ContextVar[str | None] = ContextVar("evalpulse_model_name", default=None)
_ctx_tags: ContextVar[list[str] | None] = ContextVar("evalpulse_tags", default=None)

# Global event queue — bounded to prevent memory issues
_event_queue: queue.Queue = queue.Queue(maxsize=10000)

# Global state
_initialized = False
_worker = None
_lock = threading.Lock()


class EvalContext:
    """Context manager for setting evaluation metadata scope.

    Usage:
        with EvalContext(app='my-rag', query=user_query, context=docs) as ctx:
            response = llm.generate(user_query)
            ctx.log(response)
    """

    def __init__(
        self,
        app: str | None = None,
        model: str | None = None,
        tags: list[str] | None = None,
        query: str | None = None,
        context: str | None = None,
    ):
        self.app = app
        self.model = model
        self.tags = tags
        self.query = query
        self.context = context
        self._tokens = []
        self._start_time = None

    def __enter__(self) -> EvalContext:
        if self.app:
            self._tokens.append(_ctx_app_name.set(self.app))
        if self.model:
            self._tokens.append(_ctx_model_name.set(self.model))
        if self.tags:
            self._tokens.append(_ctx_tags.set(self.tags))
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for token in reversed(self._tokens):
            token.var.reset(token)
        self._tokens.clear()
        return None

    def log(self, response: str, **kwargs) -> None:
        """Log an LLM response within this context."""
        latency_ms = 0
        if self._start_time:
            latency_ms = int((time.perf_counter() - self._start_time) * 1000)

        event = EvalEvent(
            app_name=self.app or _ctx_app_name.get() or "default",
            query=self.query or "",
            context=self.context,
            response=response,
            model_name=self.model or _ctx_model_name.get() or "unknown",
            latency_ms=latency_ms,
            tags=self.tags or _ctx_tags.get() or [],
        )
        _enqueue_event(event)


def _enqueue_event(event: EvalEvent) -> None:
    """Push an event to the queue (non-blocking, drops if full)."""
    try:
        _event_queue.put_nowait(event)
    except queue.Full:
        logger.warning("EvalPulse event queue full — event dropped.")


def track(
    func: Callable | None = None,
    *,
    app: str | None = None,
    model: str | None = None,
    tags: list[str] | None = None,
) -> Callable:
    """Decorator to track LLM function calls.

    Usage:
        @track(app='my-chatbot')
        def ask_llm(query):
            return llm.generate(query)

    The decorated function's first positional argument is captured as 'query',
    and the return value is captured as 'response'.
    Adds ~1-2ms overhead (just timestamp + queue push).
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            # Extract query from first argument
            query = ""
            if args:
                query = str(args[0])
            elif "query" in kwargs:
                query = str(kwargs["query"])

            # Extract context if provided
            context = kwargs.get("context", None)
            if context is None and len(args) > 1:
                second_arg = args[1]
                if isinstance(second_arg, str):
                    context = second_arg

            event = EvalEvent(
                app_name=app or _ctx_app_name.get() or "default",
                query=query,
                context=context,
                response=str(result) if result is not None else "",
                model_name=model or _ctx_model_name.get() or "unknown",
                latency_ms=elapsed_ms,
                tags=tags or _ctx_tags.get() or [],
            )
            _enqueue_event(event)
            return result

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def atrack(
    func: Callable | None = None,
    *,
    app: str | None = None,
    model: str | None = None,
    tags: list[str] | None = None,
) -> Callable:
    """Async decorator to track async LLM function calls.

    Usage:
        @atrack(app='my-chatbot')
        async def ask_llm(query):
            return await async_llm.generate(query)
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await fn(*args, **kwargs)
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            query = ""
            if args:
                query = str(args[0])
            elif "query" in kwargs:
                query = str(kwargs["query"])

            context = kwargs.get("context", None)
            if context is None and len(args) > 1:
                second_arg = args[1]
                if isinstance(second_arg, str):
                    context = second_arg

            event = EvalEvent(
                app_name=app or _ctx_app_name.get() or "default",
                query=query,
                context=context,
                response=str(result) if result is not None else "",
                model_name=model or _ctx_model_name.get() or "unknown",
                latency_ms=elapsed_ms,
                tags=tags or _ctx_tags.get() or [],
            )
            _enqueue_event(event)
            return result

        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def get_event_queue() -> queue.Queue:
    """Get the global event queue (used by the worker)."""
    return _event_queue


def init(config_path: str | None = None) -> None:
    """Initialize EvalPulse: load config, start storage, launch background worker.

    Call this once at application startup before any @track calls.
    """
    global _initialized, _worker

    with _lock:
        if _initialized:
            return

        from evalpulse.config import get_config
        from evalpulse.storage import get_storage
        from evalpulse.worker import EvaluationWorker

        config = get_config(config_path)
        get_storage(config)
        _worker = EvaluationWorker(event_queue=_event_queue, config=config)
        _worker.start()
        _initialized = True


def shutdown(timeout: float = 5.0) -> None:
    """Shutdown EvalPulse: drain the queue, stop the worker, close storage.

    Call this at application exit for clean shutdown.
    """
    global _initialized, _worker

    with _lock:
        if not _initialized:
            return

        if _worker is not None:
            _worker.stop(timeout=timeout)
            _worker = None

        from evalpulse.config import reset_config
        from evalpulse.storage import reset_storage

        reset_storage()
        reset_config()

        # Clean up module singletons to release memory
        try:
            from evalpulse.modules.drift_store import reset_drift_store

            reset_drift_store()
        except ImportError:
            pass
        try:
            from evalpulse.modules.embeddings import reset_embedding_service

            reset_embedding_service()
        except ImportError:
            pass

        _initialized = False


def is_initialized() -> bool:
    """Check if EvalPulse has been initialized."""
    return _initialized
