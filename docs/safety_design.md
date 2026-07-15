# Safety Design

## Product Boundary

The assistant provides general information, structured self-understanding and help-seeking navigation. It is not a doctor, therapist, diagnostic system or medication advisor.

## Enforced Rules

- No diagnosis claims.
- No disease probabilities.
- No future suicide/self-harm prediction.
- No simple future-danger labels.
- No medication, dose or discontinuation advice.
- Crisis expressions trigger fixed response before ordinary chat.
- Sensitive text is not persisted by default.
- Feedback comments are redacted and stored with a digest.

## SafetyRouter

Layer 1 is deterministic pattern matching for explicit crisis and medication requests. Layer 2 is an LLM structured classifier interface. Mock mode returns safe low-confidence continuation unless rules match.

## ResponsePolicy

Final answers are checked for diagnostic and medication patterns. If a violation is detected, the response is replaced with a safe boundary message.

## Crisis Template

The crisis response does not show internal reasoning and does not attempt prediction. It directs the user to local emergency services, nearby safe places, trusted people and professional support.

## Known Limits

The MVP uses a small deterministic rule set. It is suitable for demos and regression tests, not production crisis detection.

