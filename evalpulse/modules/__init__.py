"""EvalPulse evaluation modules."""

from __future__ import annotations

from typing import Any

from evalpulse.modules.base import EvalModule


def get_default_modules(config: Any = None) -> list[EvalModule]:
    """Get the list of enabled evaluation modules based on config.

    Returns only modules whose dependencies are available
    and whose config flag is enabled.
    """
    modules: list[EvalModule] = []

    # Module 4: Response Quality (always available)
    if config is None or config.modules.response_quality:
        from evalpulse.modules.quality import ResponseQualityModule

        if ResponseQualityModule.is_available():
            modules.append(ResponseQualityModule())

    # Module 2: Drift (requires sentence-transformers + chromadb)
    if config is None or config.modules.drift:
        try:
            from evalpulse.modules.drift import SemanticDriftModule

            if SemanticDriftModule.is_available():
                modules.append(SemanticDriftModule(config=config))
        except ImportError:
            pass

    # Module 1: Hallucination (requires selfcheckgpt or groq)
    if config is None or config.modules.hallucination:
        try:
            from evalpulse.modules.hallucination import HallucinationModule

            if HallucinationModule.is_available():
                modules.append(HallucinationModule(config=config))
        except ImportError:
            pass

    # Module 3: RAG Quality (requires embeddings)
    if config is None or config.modules.rag_quality:
        try:
            from evalpulse.modules.rag_eval import RAGQualityModule

            if RAGQualityModule.is_available():
                modules.append(RAGQualityModule())
        except ImportError:
            pass

    return modules


__all__ = ["EvalModule", "get_default_modules"]
