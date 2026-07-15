# Agent Instructions For This Repository

This repository implements a non-diagnostic mental health information support assistant. Preserve that boundary in code, UI, prompts, docs, and tests.

## Safety Rules

- Do not add diagnosis, disease probability, future self-harm prediction, medication dose, stop-medication, or treatment-decision features.
- Crisis language must route to a fixed safety response before ordinary chat.
- Survey scores are only self-understanding references and must be computed deterministically by code.
- Do not store full sensitive conversation text by default.
- Do not add real user data, private health data, or unclear-license datasets.
- Do not claim clinical validation, clinician involvement, production deployment, or real user metrics unless evidence is committed.

## Engineering Rules

- Keep business logic out of FastAPI route handlers.
- Prefer small service modules and tests over monolithic scripts.
- Keep MockLLM mode working without API keys.
- Read configuration from environment variables and `.env`; never hardcode secrets.
- Use `ruff`, `mypy`, `pytest`, and the evaluation scripts before claiming readiness.
- Document any skipped or failed check in QA docs instead of implying it passed.

