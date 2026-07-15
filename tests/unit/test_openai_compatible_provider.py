from app.core.config import Settings
from app.domain.intents import IntentLabel
from app.domain.safety import SafetyAction
from app.llm.openai_compatible import OpenAICompatibleLLM
from app.llm.schemas import GeneratedAnswer, IntentClassification, SafetyClassification


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


def test_safety_schema_instruction_uses_json_enum_strings():
    provider = OpenAICompatibleLLM(Settings(openai_api_key="test"))
    instruction = provider._schema_instruction(SafetyClassification)
    assert f'"{SafetyAction.fixed_crisis_response.value}"' in instruction
    assert "fixed_crisis_response," not in instruction


def test_intent_schema_instruction_uses_json_enum_strings():
    provider = OpenAICompatibleLLM(Settings(openai_api_key="test"))
    instruction = provider._schema_instruction(IntentClassification)
    assert f'"{IntentLabel.knowledge_query.value}"' in instruction
    assert "knowledge_query," not in instruction
