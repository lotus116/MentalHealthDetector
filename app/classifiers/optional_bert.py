"""Isolated optional BERT adapter.

This module intentionally does not load model weights by default. The old BERT
training/inference code is preserved in legacy/ and must not be treated as a
clinical risk prediction model.
"""

from app.domain.models import IntentResult


class OptionalBertClassifier:
    """Disabled-by-default placeholder for interview-safe migration."""

    def __init__(self, enabled: bool, threshold: float):
        self.enabled = enabled
        self.threshold = threshold

    def classify(self, text: str) -> IntentResult | None:
        if not self.enabled:
            return None
        return None
