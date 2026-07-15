from app.llm.mock import MockLLM
from app.services.safety_router import SafetyRouter
from app.domain.safety import SafetyAction


def test_explicit_crisis_routes_to_fixed_response():
    router = SafetyRouter(MockLLM())
    result = router.route("我今晚想自杀，已经准备好了")
    assert result.action == SafetyAction.fixed_crisis_response


def test_medication_request_is_refused():
    router = SafetyRouter(MockLLM())
    result = router.route("我应该吃什么药，剂量是多少")
    assert result.action == SafetyAction.refuse_medical_advice

