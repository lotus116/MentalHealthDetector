"""SQLite-backed feedback storage without full message persistence."""

import sqlite3
from pathlib import Path

from app.core.privacy import redact_text, text_digest
from app.domain.models import FeedbackIn


class FeedbackRepository:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY, session_id TEXT, rating TEXT, comment_preview TEXT, comment_digest TEXT)"
            )

    def add(self, feedback: FeedbackIn) -> None:
        preview = redact_text(feedback.comment or "")
        digest = text_digest(feedback.comment or "")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO feedback (session_id, rating, comment_preview, comment_digest) VALUES (?, ?, ?, ?)",
                (feedback.session_id, feedback.rating, preview, digest),
            )

