"""Survey endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.deps import get_survey_service
from app.services.survey_service import SurveyService

router = APIRouter(prefix="/survey", tags=["survey"])


class SurveyAnswers(BaseModel):
    answers: dict[str, int]


@router.get("")
def get_survey(service: SurveyService = Depends(get_survey_service)) -> dict:
    return service.get_survey()


@router.post("/score")
def score_survey(payload: SurveyAnswers, service: SurveyService = Depends(get_survey_service)) -> dict:
    try:
        return service.score(payload.answers)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

