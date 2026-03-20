"""Module 4: Response Quality Scorer.

Evaluates surface-level quality attributes of LLM responses:
sentiment, toxicity, length, language detection, and denial/refusal rate.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

from evalpulse.models import EvalEvent
from evalpulse.modules.base import EvalModule

logger = logging.getLogger("evalpulse.modules.quality")

# Denial/refusal patterns
_DENIAL_PATTERNS = [
    r"\bi\s+cannot\b",
    r"\bi\'?m\s+unable\b",
    r"\bi\s+can\'?t\b",
    r"\bas\s+an?\s+ai\b",
    r"\bi\s+don\'?t\s+have\s+access\b",
    r"\bi\'?m\s+not\s+able\b",
    r"\bi\s+am\s+not\s+able\b",
    r"\bi\s+apologize,?\s+but\s+i\b",
    r"\bunfortunately,?\s+i\s+(cannot|can\'?t)\b",
    r"\bi\s+do\s+not\s+have\s+(the\s+ability|enough\s+information)\b",
]
_DENIAL_RE = re.compile("|".join(_DENIAL_PATTERNS), re.IGNORECASE)


class ResponseQualityModule(EvalModule):
    """Module 4: Response Quality Scorer.

    Evaluates:
    - Sentiment (0.0 = negative, 0.5 = neutral, 1.0 = positive)
    - Toxicity (0.0 = benign, 1.0 = toxic)
    - Response length (word count)
    - Language detection
    - Denial/refusal detection
    """

    _detoxify_model = None  # Lazy-loaded singleton
    _detoxify_lock = threading.Lock()

    @property
    def name(self) -> str:
        return "response_quality"

    @classmethod
    def is_available(cls) -> bool:
        """Check if required dependencies are available."""
        try:
            import langdetect  # noqa: F401

            return True
        except ImportError:
            return False

    async def evaluate(self, event: EvalEvent) -> dict[str, Any]:
        """Evaluate response quality metrics."""
        response = event.response or ""

        sentiment = self._score_sentiment(response)
        toxicity = self._score_toxicity(response)
        length = len(response.split()) if response.strip() else 0
        language = self._detect_language(response)
        is_denial = self._detect_denial(response)

        return {
            "sentiment_score": sentiment,
            "toxicity_score": toxicity,
            "response_length": length,
            "language_detected": language,
            "is_denial": is_denial,
        }

    def evaluate_sync(self, event: EvalEvent) -> dict[str, Any]:
        """Synchronous evaluation (no async needed for this module)."""
        response = event.response or ""

        sentiment = self._score_sentiment(response)
        toxicity = self._score_toxicity(response)
        length = len(response.split()) if response.strip() else 0
        language = self._detect_language(response)
        is_denial = self._detect_denial(response)

        return {
            "sentiment_score": sentiment,
            "toxicity_score": toxicity,
            "response_length": length,
            "language_detected": language,
            "is_denial": is_denial,
        }

    def _score_sentiment(self, text: str) -> float:
        """Score sentiment using a simple lexicon-based approach.

        Returns 0.0 (very negative) to 1.0 (very positive), 0.5 = neutral.
        Uses VADER-like approach with a built-in word list.
        """
        if not text.strip():
            return 0.5

        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer

            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text)
            # compound ranges from -1 to 1, normalize to 0-1
            return (scores["compound"] + 1) / 2
        except Exception:
            # Fallback: simple positive/negative word counting
            return self._simple_sentiment(text)

    def _simple_sentiment(self, text: str) -> float:
        """Fallback sentiment scoring using keyword matching."""
        positive_words = {
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "fantastic",
            "helpful",
            "useful",
            "clear",
            "accurate",
            "correct",
            "perfect",
            "happy",
            "love",
            "best",
            "beautiful",
            "brilliant",
            "outstanding",
        }
        negative_words = {
            "bad",
            "terrible",
            "awful",
            "horrible",
            "wrong",
            "error",
            "fail",
            "poor",
            "worst",
            "ugly",
            "hate",
            "stupid",
            "useless",
            "broken",
            "confused",
            "disappointed",
            "frustrating",
            "annoying",
        }
        words = set(text.lower().split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)
        total = pos_count + neg_count
        if total == 0:
            return 0.5
        return 0.5 + 0.5 * (pos_count - neg_count) / total

    def _score_toxicity(self, text: str) -> float:
        """Score toxicity using the detoxify model.

        Returns 0.0 (benign) to 1.0 (toxic).
        Lazy-loads the model on first call.
        """
        if not text.strip():
            return 0.0

        try:
            if ResponseQualityModule._detoxify_model is None:
                with ResponseQualityModule._detoxify_lock:
                    # Double-check after acquiring lock
                    if ResponseQualityModule._detoxify_model is None:
                        import detoxify

                        ResponseQualityModule._detoxify_model = detoxify.Detoxify("original")
                        logger.info("Loaded detoxify model")

            results = ResponseQualityModule._detoxify_model.predict(text)
            return float(results.get("toxicity", 0.0))
        except Exception as e:
            logger.warning(f"Toxicity scoring failed: {e}")
            return 0.0

    def _detect_language(self, text: str) -> str:
        """Detect the language of the text.

        Returns ISO 639-1 language code (e.g., 'en', 'fr', 'es').
        Defaults to 'en' on failure.
        """
        if not text.strip() or len(text.strip()) < 10:
            return "en"

        try:
            from langdetect import detect

            return detect(text)
        except Exception:
            return "en"

    def _detect_denial(self, text: str) -> bool:
        """Detect if the response is a denial/refusal.

        Matches common LLM refusal patterns like 'I cannot', 'As an AI', etc.
        """
        if not text.strip():
            return False
        return bool(_DENIAL_RE.search(text))
