"""Chat API routes."""

from fastapi import APIRouter, Depends

from app.services.conversation_service import ConversationService
from app.api.deps import get_conversation_service, get_sessions
from app.domain.models import ChatRequest, ChatResponse
from app.repositories.session_repository import SessionRepository

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest, service: ConversationService = Depends(get_conversation_service)) -> ChatResponse:
    return service.handle(request)


@router.delete("/{session_id}")
def clear_session(session_id: str, sessions: SessionRepository = Depends(get_sessions)) -> dict:
    sessions.clear(session_id)
    return {"cleared": True, "session_id": session_id}
