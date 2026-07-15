"""Optional local classifier interface."""

from abc import ABC, abstractmethod

from app.domain.models import IntentResult


class OptionalClassifier(ABC):
    @abstractmethod
    def classify(self, text: str) -> IntentResult | None:
        """Return a high-confidence intent candidate, or None."""
