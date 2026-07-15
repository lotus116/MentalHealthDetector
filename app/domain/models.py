"""Shared Pydantic models for API and service boundaries."""

from typing import Literal

from pydantic import BaseModel, Field

from app.domain.intents import IntentLabel
from app.domain.safety import SafetyAction


class Source(BaseModel):
    title: str
    source_id: str
    snippet: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    mode: Literal["auto", "knowledge", "survey", "resources", "support"] = "auto"
    session_id: str = "default"
    region: str | None = None


class ChatResponse(BaseModel):
    answer: str
    intent: IntentLabel
    safety_action: SafetyAction
    sources: list[Source] = []
    disclaimer: str
    latency_ms: int


class IntentResult(BaseModel):
    label: IntentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class SafetyResult(BaseModel):
    action: SafetyAction
    matched_rule: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class FeedbackIn(BaseModel):
    session_id: str = "default"
    rating: Literal["helpful", "not_helpful", "inaccurate", "unsafe"]
    comment: str | None = Field(default=None, max_length=1000)
