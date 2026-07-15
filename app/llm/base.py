"""LLM provider interface."""

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Small provider interface for structured JSON outputs."""

    @abstractmethod
    def structured(self, prompt_name: str, variables: dict, schema: type[T]) -> T:
        """Return a schema-validated model."""
