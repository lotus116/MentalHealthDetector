from app.core.privacy import redact_text, text_digest


def test_redact_text_masks_email_and_phone():
    redacted = redact_text("请联系 me@example.com 或 +852 9123 4567，我最近压力很大", max_preview=200)
    assert "me@example.com" not in redacted
    assert "9123 4567" not in redacted
    assert "[email]" in redacted
    assert "[phone]" in redacted


def test_text_digest_does_not_expose_source_text():
    digest = text_digest("一段敏感对话")
    assert "敏感" not in digest
    assert len(digest) == 16
