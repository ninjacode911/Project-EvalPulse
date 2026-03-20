"""Abstract base class for EvalPulse evaluation modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from evalpulse.models import EvalEvent


class EvalModule(ABC):
    """Base class for all evaluation modules.

    Each module implements evaluate() which takes an EvalEvent
    and returns a dict of EvalRecord field updates.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable module name."""

    @abstractmethod
    async def evaluate(self, event: EvalEvent) -> dict[str, Any]:
        """Evaluate an event asynchronously.

        Returns a dict of EvalRecord fields to update.
        Only include fields this module is responsible for.
        """

    def evaluate_sync(self, event: EvalEvent) -> dict[str, Any]:
        """Synchronous wrapper for evaluate().

        Subclasses can override this for purely synchronous modules.
        Uses asyncio.run() which is safe in Python 3.11+.
        """
        import asyncio

        return asyncio.run(self.evaluate(event))

    @classmethod
    def is_available(cls) -> bool:
        """Check if this module's dependencies are available.

        Override this to check for optional dependencies.
        """
        return True
