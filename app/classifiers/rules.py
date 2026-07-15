"""Lightweight rule classifier used for offline comparison."""

from app.domain.intents import IntentLabel
from app.domain.models import IntentResult


class RuleClassifier:
    def classify(self, text: str) -> IntentResult:
        if "问卷" in text:
            return IntentResult(label=IntentLabel.survey_request, confidence=0.9)
        return IntentResult(label=IntentLabel.supportive_conversation, confidence=0.55)
