from app.domain.intents import IntentLabel
from app.llm.mock import MockLLM
from app.services.intent_router import IntentRouter


def test_survey_rule():
    result = IntentRouter(MockLLM()).route("我想做一个压力问卷")
    assert result.label == IntentLabel.survey_request


def test_resource_rule():
    result = IntentRouter(MockLLM()).route("我想找心理咨询资源")
    assert result.label == IntentLabel.resource_request
