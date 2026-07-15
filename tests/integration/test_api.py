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


def test_survey_score_api():
    response = client.post("/survey/score", json={"answers": {"sleep": 1, "focus": 1, "support": 1}})
    assert response.status_code == 200
    assert response.json()["score"] == 3
