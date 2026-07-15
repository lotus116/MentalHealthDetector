"""In-memory session store for current-process demos."""


class SessionRepository:
    """Stores only current runtime session state with bounded memory."""

    def __init__(self, max_messages_per_session: int = 20):
        self._store: dict[str, list[dict[str, str]]] = {}
        self.max_messages_per_session = max_messages_per_session

    def append(self, session_id: str, role: str, content: str) -> None:
        messages = self._store.setdefault(session_id, [])
        messages.append({"role": role, "content": content})
        if len(messages) > self.max_messages_per_session:
            del messages[: len(messages) - self.max_messages_per_session]

    def history(self, session_id: str) -> list[dict[str, str]]:
        return self._store.get(session_id, [])

    def recent_history(self, session_id: str, limit: int = 8) -> list[dict[str, str]]:
        return self.history(session_id)[-limit:]

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)
