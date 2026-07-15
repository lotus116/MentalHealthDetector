from app.core.config import Settings
from app.domain.safety import SafetyAction
from app.llm.openai_compatible import OpenAICompatibleLLM
from app.llm.schemas import GeneratedAnswer, SafetyClassification


def test_coerces_safe_boolean_response():
    provider = OpenAICompatibleLLM(Settings(openai_api_key="test"))
    result = provider._coerce_content('{"safe": true}', SafetyClassification)
    assert result is not None
    assert result.action == SafetyAction.continue_normal


def test_coerces_safety_string_response():
    provider = OpenAICompatibleLLM(Settings(openai_api_key="test"))
    result = provider._coerce_content('{"safety": "safe"}', SafetyClassification)
    assert result is not None
    assert result.action == SafetyAction.continue_normal


def test_coerces_plain_text_answer():
    provider = OpenAICompatibleLLM(Settings(openai_api_key="test"))
    result = provider._coerce_content("这是一段普通回答", GeneratedAnswer)
    assert result is not None
    assert result.answer == "这是一段普通回答"
