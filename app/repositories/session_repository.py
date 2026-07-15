"""In-memory session store for current-process demos."""


class SessionRepository:
    """Stores only current runtime session state."""

    def __init__(self):
        self._store: dict[str, list[dict[str, str]]] = {}

    def append(self, session_id: str, role: str, content: str) -> None:
        self._store.setdefault(session_id, []).append({"role": role, "content": content})

    def history(self, session_id: str) -> list[dict[str, str]]:
        return self._store.get(session_id, [])

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)
