"""Feedback endpoint."""

from fastapi import APIRouter, Depends

from app.api.deps import get_feedback_repo
from app.domain.models import FeedbackIn
from app.repositories.feedback_repository import FeedbackRepository

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("")
def add_feedback(feedback: FeedbackIn, repo: FeedbackRepository = Depends(get_feedback_repo)) -> dict:
    repo.add(feedback)
    return {"stored": True}

