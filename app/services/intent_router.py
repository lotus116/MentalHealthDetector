"""Cascaded intent routing with rules, optional classifier and LLM fallback."""

import re

from app.domain.intents import IntentLabel
from app.domain.models import IntentResult
from app.llm.base import LLMProvider
from app.llm.schemas import IntentClassification


RULES: list[tuple[IntentLabel, str]] = [
    (IntentLabel.out_of_scope, r"股票|彩票|作业答案|黑客|黑进|入侵|写病毒"),
    (IntentLabel.survey_request, r"问卷|量表|压力测试|自我了解|测一测"),
    (IntentLabel.resource_request, r"求助|资源|心理咨询|医生|热线|专业支持"),
    (IntentLabel.knowledge_query, r"什么是|如何|为什么|区别|知识|解释|资料|会.*吗|能.*吗"),
]


class IntentRouter:
    """Intent router using high-confidence rules then structured LLM."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def route(self, message: str, forced_mode: str = "auto") -> IntentResult:
        forced = {
            "knowledge": IntentLabel.knowledge_query,
            "survey": IntentLabel.survey_request,
            "resources": IntentLabel.resource_request,
            "support": IntentLabel.supportive_conversation,
        }.get(forced_mode)
        if forced:
            return IntentResult(label=forced, confidence=1.0, rationale="forced_mode")
        for label, pattern in RULES:
            if re.search(pattern, message, flags=re.IGNORECASE):
                return IntentResult(label=label, confidence=0.96, rationale=f"rule:{pattern}")
        try:
            result = self.llm.structured("intent_router", {"message": message}, IntentClassification)
        except Exception:
            return IntentResult(label=IntentLabel.supportive_conversation, confidence=0.5, rationale="llm_fallback")
        if result.confidence < 0.5:
            return IntentResult(
                label=IntentLabel.clarification_needed, confidence=result.confidence, rationale="low_confidence"
            )
        return IntentResult(label=result.label, confidence=result.confidence, rationale="llm")
