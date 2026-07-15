"""Privacy helpers used before logging or persistence."""

import hashlib
import re

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d -]{7,}\d)(?!\d)")


def redact_text(text: str, max_preview: int = 80) -> str:
    """Return a short, redacted preview suitable for logs."""

    redacted = EMAIL_RE.sub("[email]", text)
    redacted = PHONE_RE.sub("[phone]", redacted)
    return redacted[:max_preview]


def text_digest(text: str) -> str:
    """Stable digest for debugging without storing sensitive text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
