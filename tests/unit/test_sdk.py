"""Tests for EvalPulse SDK — @track decorator, @atrack, EvalContext."""

import asyncio
import time

import pytest

from evalpulse.models import EvalEvent
from evalpulse.sdk import (
    EvalContext,
    _event_queue,
    atrack,
    track,
)


@pytest.fixture(autouse=True)
def clear_queue():
    """Clear the event queue before each test."""
    while not _event_queue.empty():
        try:
            _event_queue.get_nowait()
        except Exception:
            break
    yield
    while not _event_queue.empty():
        try:
            _event_queue.get_nowait()
        except Exception:
            break


class TestTrackDecorator:
    """Test the @track decorator."""

    def test_preserves_return_value(self):
        """Decorated function should return the original value."""

        @track(app="test")
        def my_func(query):
            return f"Answer to: {query}"

        result = my_func("hello")
        assert result == "Answer to: hello"

    def test_captures_query_and_response(self):
        """Should capture query (first arg) and response (return value)."""

        @track(app="test")
        def my_func(query):
            return "response text"

        my_func("my query")
        event = _event_queue.get_nowait()
        assert isinstance(event, EvalEvent)
        assert event.query == "my query"
        assert event.response == "response text"
        assert event.app_name == "test"

    def test_captures_latency(self):
        """Should measure function execution latency."""

        @track(app="test")
        def slow_func(query):
            time.sleep(0.05)
            return "done"

        slow_func("q")
        event = _event_queue.get_nowait()
        assert event.latency_ms >= 40  # Allow some tolerance

    def test_overhead_is_minimal(self):
        """Decorator overhead should be < 5ms per call."""

        @track(app="test")
        def fast_func(query):
            return "fast"

        def bare_func(query):
            return "fast"

        # Warm up
        for _ in range(10):
            fast_func("q")
        while not _event_queue.empty():
            _event_queue.get_nowait()

        n = 500
        start = time.perf_counter()
        for i in range(n):
            fast_func(f"q{i}")
        tracked_time = time.perf_counter() - start

        while not _event_queue.empty():
            _event_queue.get_nowait()

        start = time.perf_counter()
        for i in range(n):
            bare_func(f"q{i}")
        bare_time = time.perf_counter() - start

        overhead_per_call_ms = ((tracked_time - bare_time) / n) * 1000
        assert overhead_per_call_ms < 5, f"Overhead {overhead_per_call_ms:.2f}ms exceeds 5ms limit"

    def test_handles_none_return(self):
        """Should handle functions that return None."""

        @track(app="test")
        def void_func(query):
            pass

        result = void_func("q")
        assert result is None
        event = _event_queue.get_nowait()
        assert event.response == ""

    def test_captures_kwargs(self):
        """Should capture query from kwargs."""

        @track(app="test")
        def my_func(query="default"):
            return "ok"

        my_func(query="kwarg query")
        event = _event_queue.get_nowait()
        assert event.query == "kwarg query"

    def test_preserves_function_metadata(self):
        """Decorator should preserve __name__ and __doc__."""

        @track(app="test")
        def documented_func(query):
            """This is documented."""
            return "ok"

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is documented."

    def test_tags_propagation(self):
        """Tags should be included in the event."""

        @track(app="test", tags=["prod", "v2"])
        def my_func(query):
            return "ok"

        my_func("q")
        event = _event_queue.get_nowait()
        assert event.tags == ["prod", "v2"]


class TestAtrackDecorator:
    """Test the @atrack async decorator."""

    def test_async_function(self):
        """Should work with async functions."""

        @atrack(app="test-async")
        async def async_func(query):
            return f"async: {query}"

        result = asyncio.run(async_func("hello"))
        assert result == "async: hello"
        event = _event_queue.get_nowait()
        assert event.app_name == "test-async"
        assert event.query == "hello"


class TestEvalContext:
    """Test the EvalContext context manager."""

    def test_log_response(self):
        """EvalContext.log() should push an event to the queue."""
        with EvalContext(app="ctx-app", query="ctx query") as ctx:
            ctx.log("ctx response")

        event = _event_queue.get_nowait()
        assert event.app_name == "ctx-app"
        assert event.query == "ctx query"
        assert event.response == "ctx response"

    def test_latency_measurement(self):
        """EvalContext should measure latency between enter and log."""
        with EvalContext(app="test") as ctx:
            time.sleep(0.05)
            ctx.log("done")

        event = _event_queue.get_nowait()
        assert event.latency_ms >= 40

    def test_context_with_rag(self):
        """Should capture context for RAG pipelines."""
        with EvalContext(app="rag-app", query="What is X?", context="X is defined as...") as ctx:
            ctx.log("X is something based on the context.")

        event = _event_queue.get_nowait()
        assert event.context == "X is defined as..."
