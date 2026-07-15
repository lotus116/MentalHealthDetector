from pathlib import Path

import pytest

from app.services.survey_service import SurveyService


def test_survey_score_is_deterministic():
    service = SurveyService(Path("surveys/example_wellbeing_survey.json"))
    result = service.score({"sleep": 1, "focus": 2, "support": 3})
    assert result["score"] == 6
    assert result["max_score"] == 9
    assert "不是医学诊断" in result["disclaimer"]


def test_survey_rejects_invalid_values():
    service = SurveyService(Path("surveys/example_wellbeing_survey.json"))
    with pytest.raises(ValueError):
        service.score({"sleep": 9, "focus": 2, "support": 3})
