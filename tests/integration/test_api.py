from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_crisis_fixed_response():
    response = client.post("/chat", json={"message": "我不想活了，想自杀", "session_id": "t"})
    assert response.status_code == 200
    data = response.json()
    assert data["safety_action"] == "fixed_crisis_response"
    assert "当地紧急服务" in data["answer"]


def test_chat_uses_session_history_for_supportive_followup():
    session_id = "memory-test"
    client.delete(f"/chat/{session_id}")
    first = client.post("/chat", json={"message": "最近有点烦，想聊聊", "session_id": session_id})
    second = client.post("/chat", json={"message": "还有点睡不好", "session_id": session_id})
    assert first.status_code == 200
    assert second.status_code == 200
    assert "当前会话" in second.json()["answer"]


def test_survey_score_api():
    survey = client.get("/survey").json()
    answers = {question["id"]: 1 for question in survey["questions"]}
    response = client.post("/survey/score", json={"answers": answers})
    assert response.status_code == 200
    assert response.json()["score"] == 10


def test_survey_score_rejects_incomplete_answers():
    response = client.post("/survey/score", json={"answers": {"sleep": 1}})
    assert response.status_code == 422
