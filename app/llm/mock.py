"""Deterministic mock LLM for offline demos and tests."""

from app.domain.intents import IntentLabel
from app.domain.safety import SafetyAction
from app.llm.base import LLMProvider, T
from app.llm.schemas import GeneratedAnswer, IntentClassification, SafetyClassification


class MockLLM(LLMProvider):
    """Rule-backed deterministic responses."""

    def structured(self, prompt_name: str, variables: dict, schema: type[T]) -> T:
        text = str(variables.get("message") or variables.get("question") or "").lower()
        if schema is IntentClassification:
            label = IntentLabel.supportive_conversation
            if any(x in text for x in ["问卷", "量表", "压力测试", "自我了解"]):
                label = IntentLabel.survey_request
            elif any(x in text for x in ["资源", "求助", "心理咨询", "医生", "热线"]):
                label = IntentLabel.resource_request
            elif any(x in text for x in ["什么是", "如何", "为什么", "知识", "解释"]):
                label = IntentLabel.knowledge_query
            return schema(label=label, confidence=0.7)  # type: ignore[return-value]
        if schema is SafetyClassification:
            return schema(action=SafetyAction.continue_normal, confidence=0.55)  # type: ignore[return-value]
        if schema is GeneratedAnswer:
            answer = (
                "我可以提供一般性信息和下一步建议，但不能诊断。"
                "你可以描述当前困扰、持续时间以及希望了解的方向。"
            )
            return schema(answer=answer)  # type: ignore[return-value]
        raise ValueError(f"Unsupported schema for MockLLM: {schema}")

