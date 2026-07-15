"""FastAPI dependency container."""

from functools import lru_cache

from app.core.config import get_settings
from app.llm import build_llm
from app.repositories.feedback_repository import FeedbackRepository
from app.repositories.session_repository import SessionRepository
from app.services.conversation_service import ConversationService
from app.services.intent_router import IntentRouter
from app.services.rag_service import RagService
from app.services.resource_service import ResourceService
from app.services.response_policy import ResponsePolicy
from app.services.safety_router import SafetyRouter
from app.services.survey_service import SurveyService


@lru_cache
def get_sessions() -> SessionRepository:
    return SessionRepository()


@lru_cache
def get_conversation_service() -> ConversationService:
    settings = get_settings()
    llm = build_llm()
    return ConversationService(
        SafetyRouter(llm),
        IntentRouter(llm),
        RagService(settings),
        ResourceService(),
        ResponsePolicy(),
        get_sessions(),
        llm,
    )


@lru_cache
def get_survey_service() -> SurveyService:
    return SurveyService(get_settings().survey_path)


@lru_cache
def get_feedback_repo() -> FeedbackRepository:
    return FeedbackRepository(get_settings().sqlite_path)

