from app.repositories.session_repository import SessionRepository


def test_session_repository_keeps_recent_messages_only():
    repo = SessionRepository(max_messages_per_session=3)
    for idx in range(5):
        repo.append("s", "user", str(idx))
    assert [message["content"] for message in repo.history("s")] == ["2", "3", "4"]


def test_recent_history_limit():
    repo = SessionRepository()
    for idx in range(10):
        repo.append("s", "user", str(idx))
    assert len(repo.recent_history("s", limit=4)) == 4
