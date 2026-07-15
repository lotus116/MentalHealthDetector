"""Config-driven deterministic survey engine."""

import json
from pathlib import Path


class SurveyService:
    """Loads a survey config and scores answers deterministically."""

    def __init__(self, path: Path):
        self.config = json.loads(path.read_text(encoding="utf-8"))

    def get_survey(self) -> dict:
        return self.config

    def score(self, answers: dict[str, int]) -> dict:
        questions = self.config["questions"]
        default_options = self.config.get("options", [])
        total = 0
        max_total = 0
        for q in questions:
            qid = q["id"]
            value = answers.get(qid)
            options = q.get("options", default_options)
            valid = [opt["score"] for opt in options]
            if value is None or value not in valid:
                raise ValueError(f"Invalid answer for {qid}")
            total += value
            max_total += max(valid)
        percent = round(total / max_total, 3) if max_total else 0.0
        interpretation = "近期压力体验较少"
        if percent >= 0.66:
            interpretation = "近期压力体验较多，建议考虑专业支持或可信赖的人际支持"
        elif percent >= 0.33:
            interpretation = "近期有一些压力体验，可关注睡眠、支持系统和日常节奏"
        return {
            "score": total,
            "max_score": max_total,
            "percent": percent,
            "interpretation": interpretation,
            "disclaimer": "分数仅用于自我了解参考，不是医学诊断，也不决定你是否有权获得帮助。",
        }
