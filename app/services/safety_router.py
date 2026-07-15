"""Deterministic first-pass safety routing."""

import re

from app.domain.models import SafetyResult
from app.domain.safety import SafetyAction
from app.llm.base import LLMProvider
from app.llm.schemas import SafetyClassification


CRISIS_PATTERNS = [
    r"自杀|自伤|结束(自己|生命|这一切)|不想活|想死|遗书|今晚.*死",
    r"kill myself|suicide|end my life|want to die|hurt myself",
    r"吃.*药.*(死|结束)|跳楼|割腕|上吊|一氧化碳",
]
MEDICAL_ADVICE_PATTERNS = [r"吃什么药|剂量|停药|加药|换药", r"diagnose me|what medication|dosage"]


class SafetyRouter:
    """Routes high-risk or medical-advice requests before normal chat."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def route(self, message: str) -> SafetyResult:
        for pattern in CRISIS_PATTERNS:
            if re.search(pattern, message, flags=re.IGNORECASE):
                return SafetyResult(action=SafetyAction.fixed_crisis_response, matched_rule=pattern, confidence=1.0)
        for pattern in MEDICAL_ADVICE_PATTERNS:
            if re.search(pattern, message, flags=re.IGNORECASE):
                return SafetyResult(action=SafetyAction.refuse_medical_advice, matched_rule=pattern, confidence=0.95)
        try:
            llm_result = self.llm.structured("safety_classifier", {"message": message}, SafetyClassification)
            return SafetyResult(action=llm_result.action, confidence=llm_result.confidence)
        except Exception:
            return SafetyResult(action=SafetyAction.continue_normal, matched_rule="llm_fallback", confidence=0.5)
