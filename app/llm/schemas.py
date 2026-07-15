"""Structured LLM outputs validated by Pydantic."""

from pydantic import BaseModel, Field

from app.domain.intents import IntentLabel
from app.domain.safety import SafetyAction


class IntentClassification(BaseModel):
    label: IntentLabel
    confidence: float = Field(ge=0.0, le=1.0)


class SafetyClassification(BaseModel):
    action: SafetyAction
    confidence: float = Field(ge=0.0, le=1.0)


class GeneratedAnswer(BaseModel):
    answer: str

