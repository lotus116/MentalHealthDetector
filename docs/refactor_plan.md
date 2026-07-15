# Refactor Plan

## Must Have

- Non-diagnostic positioning across README, API, prompts, UI and tests.
- FastAPI backend with health, chat, survey and feedback endpoints.
- MockLLM mode that runs without API keys.
- Deterministic SafetyRouter before normal conversation.
- IntentRouter with clear categories.
- RAG answers with source metadata and refusal when evidence is missing.
- Config-driven survey with deterministic scoring.
- Streamlit demo with disclaimer, chat, survey, resources, feedback and clear-session controls.
- Minimal evaluation datasets and scripts.
- Automated tests for safety, intent, survey, RAG, API and response policy.

## Should Have

- SQLite feedback repository storing only redacted comments/digests.
- OpenAI-compatible provider interface for OpenAI and DashScope-compatible endpoints.
- Optional BERT adapter isolated and disabled by default.
- Dockerfile, docker-compose and Makefile.
- Interview and resume materials grounded in actual implementation.

## Could Have

- Persistent authenticated user accounts.
- Postgres repository implementation.
- Larger curated knowledge base with explicit redistribution rights.
- Real FAISS/Chroma backend behind the retrieval interface.

## Not Implemented in MVP

- Clinical validation.
- Diagnosis, disease probability, medication advice or treatment decision support.
- Real crisis hotline lookup by location.
- Multi-agent architecture, Kafka, Kubernetes or microservices.
- Production-grade authentication, audit logging and monitoring.

