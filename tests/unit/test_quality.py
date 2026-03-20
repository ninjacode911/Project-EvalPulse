"""Tests for the Response Quality Scorer module."""

import pytest

from evalpulse.models import EvalEvent
from evalpulse.modules.quality import ResponseQualityModule


@pytest.fixture
def module():
    return ResponseQualityModule()


class TestResponseQualityModule:
    """Test quality module scoring."""

    def test_module_name(self, module):
        assert module.name == "response_quality"

    def test_is_available(self):
        assert ResponseQualityModule.is_available() is True

    def test_positive_sentiment(self, module):
        event = EvalEvent(response="This is an amazing, wonderful, and excellent response!")
        result = module.evaluate_sync(event)
        assert result["sentiment_score"] > 0.6

    def test_negative_sentiment(self, module):
        event = EvalEvent(response="This is terrible, awful, and horrible. Everything is wrong.")
        result = module.evaluate_sync(event)
        assert result["sentiment_score"] < 0.4

    def test_neutral_sentiment(self, module):
        event = EvalEvent(response="The temperature today is 72 degrees Fahrenheit.")
        result = module.evaluate_sync(event)
        assert 0.3 <= result["sentiment_score"] <= 0.7

    def test_toxicity_benign(self, module):
        event = EvalEvent(response="Python is a great programming language for beginners.")
        result = module.evaluate_sync(event)
        assert result["toxicity_score"] < 0.3

    def test_toxicity_toxic(self, module):
        event = EvalEvent(response="You are a complete idiot and I hate you, you worthless fool.")
        result = module.evaluate_sync(event)
        assert result["toxicity_score"] > 0.3

    def test_response_length(self, module):
        event = EvalEvent(response="one two three four five")
        result = module.evaluate_sync(event)
        assert result["response_length"] == 5

    def test_response_length_empty(self, module):
        event = EvalEvent(response="")
        result = module.evaluate_sync(event)
        assert result["response_length"] == 0

    def test_language_detection_english(self, module):
        event = EvalEvent(
            response=(
                "This is a long enough English sentence for language detection to work properly."
            )
        )
        result = module.evaluate_sync(event)
        assert result["language_detected"] == "en"

    def test_language_detection_french(self, module):
        event = EvalEvent(
            response="Bonjour, comment allez-vous aujourd'hui? Je suis tres content de vous voir."
        )
        result = module.evaluate_sync(event)
        assert result["language_detected"] == "fr"

    def test_denial_detection_cannot(self, module):
        event = EvalEvent(response="I cannot provide medical advice.")
        result = module.evaluate_sync(event)
        assert result["is_denial"] is True

    def test_denial_detection_as_ai(self, module):
        event = EvalEvent(response="As an AI language model, I don't have personal opinions.")
        result = module.evaluate_sync(event)
        assert result["is_denial"] is True

    def test_denial_detection_unable(self, module):
        event = EvalEvent(response="I'm unable to access external websites or databases.")
        result = module.evaluate_sync(event)
        assert result["is_denial"] is True

    def test_no_denial_normal_response(self, module):
        event = EvalEvent(response="Python was created by Guido van Rossum in 1991.")
        result = module.evaluate_sync(event)
        assert result["is_denial"] is False

    def test_empty_response_defaults(self, module):
        event = EvalEvent(response="")
        result = module.evaluate_sync(event)
        assert result["sentiment_score"] == 0.5
        assert result["toxicity_score"] == 0.0
        assert result["response_length"] == 0
        assert result["language_detected"] == "en"
        assert result["is_denial"] is False

    def test_all_fields_returned(self, module):
        event = EvalEvent(response="Hello world, this is a test.")
        result = module.evaluate_sync(event)
        assert "sentiment_score" in result
        assert "toxicity_score" in result
        assert "response_length" in result
        assert "language_detected" in result
        assert "is_denial" in result
