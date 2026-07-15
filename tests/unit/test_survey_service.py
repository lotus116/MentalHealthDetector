from pathlib import Path

import pytest

from app.services.survey_service import SurveyService


def test_survey_score_is_deterministic():
    service = SurveyService(Path("surveys/example_wellbeing_survey.json"))
    answers = {question["id"]: idx % 4 for idx, question in enumerate(service.get_survey()["questions"])}
    result = service.score(answers)
    assert result["score"] == 13
    assert result["max_score"] == 30
    assert "不是医学诊断" in result["disclaimer"]


def test_survey_rejects_invalid_values():
    service = SurveyService(Path("surveys/example_wellbeing_survey.json"))
    with pytest.raises(ValueError):
        answers = {question["id"]: 1 for question in service.get_survey()["questions"]}
        answers["sleep"] = 9
        service.score(answers)
