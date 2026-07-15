"""Application orchestration for chat requests."""

import time

from app.domain.intents import IntentLabel
from app.domain.models import ChatRequest, ChatResponse, Source
from app.domain.safety import CRISIS_RESPONSE, NON_DIAGNOSTIC_DISCLAIMER, SafetyAction
from app.llm.base import LLMProvider
from app.llm.schemas import GeneratedAnswer
from app.repositories.session_repository import SessionRepository
from app.services.intent_router import IntentRouter
from app.services.rag_service import RagService
from app.services.resource_service import ResourceService
from app.services.response_policy import ResponsePolicy
from app.services.safety_router import SafetyRouter


class ConversationService:
    """Coordinates safety, intent, RAG, resources and response policy."""

    def __init__(
        self,
        safety_router: SafetyRouter,
        intent_router: IntentRouter,
        rag: RagService,
        resources: ResourceService,
        policy: ResponsePolicy,
        sessions: SessionRepository,
        llm: LLMProvider,
    ):
        self.safety_router = safety_router
        self.intent_router = intent_router
        self.rag = rag
        self.resources = resources
        self.policy = policy
        self.sessions = sessions
        self.llm = llm

    def handle(self, request: ChatRequest) -> ChatResponse:
        start = time.perf_counter()
        safety = self.safety_router.route(request.message)
        sources: list[Source] = []
        if safety.action == SafetyAction.fixed_crisis_response:
            answer = CRISIS_RESPONSE
            intent = IntentLabel.crisis_signal
        elif safety.action == SafetyAction.refuse_medical_advice:
            answer = "我不能提供药品、剂量、停药或换药建议。请联系开药医生、药师或当地医疗服务。"
            intent = IntentLabel.resource_request
        else:
            intent_result = self.intent_router.route(request.message, request.mode)
            intent = intent_result.label
            if intent == IntentLabel.knowledge_query:
                answer, sources = self.rag.answer(request.message)
            elif intent == IntentLabel.survey_request:
                answer = "你可以进入“压力/情绪自我了解”问卷页。问卷由程序确定性计分，结果只作自我了解参考。"
            elif intent == IntentLabel.resource_request:
                answer = self.resources.get_resources(request.region or "generic")
            elif intent == IntentLabel.out_of_scope:
                answer = "这个问题超出本助手范围。我可以帮助查找心理健康信息、问卷参考和求助资源。"
            elif intent == IntentLabel.clarification_needed:
                answer = "我还不确定你希望获得哪类帮助。你可以选择知识问答、问卷、自助资源或普通支持性对话。"
            else:
                generated = self.llm.structured("supportive_response", {"message": request.message}, GeneratedAnswer)
                answer = generated.answer
        answer = self.policy.validate(answer)
        self.sessions.append(request.session_id, "user", request.message[:400])
        self.sessions.append(request.session_id, "assistant", answer[:800])
        latency_ms = int((time.perf_counter() - start) * 1000)
        return ChatResponse(
            answer=answer,
            intent=intent,
            safety_action=safety.action,
            sources=sources,
            disclaimer=NON_DIAGNOSTIC_DISCLAIMER,
            latency_ms=latency_ms,
        )
