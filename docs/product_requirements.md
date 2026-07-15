# Product Requirements

## Background

The project is refactored from a diagnostic-style prototype into a non-diagnostic Mental Health Information Support Assistant.

## Target Users

- Students and early-career users seeking general mental-health information.
- Demo reviewers evaluating AI application architecture, product thinking and safety design.

## User Pain Points

- Hard to distinguish general information from diagnosis.
- Need a low-friction way to organize stress/emotion experiences.
- Need safe redirection when language suggests immediate danger.
- Need cited answers rather than unsupported LLM claims.

## As-Is Flow

User input goes through a monolithic CLI pipeline that combines RAG, BERT and LLM report generation. The result can resemble diagnosis and depends on local models/API keys.

## To-Be Flow

User input -> disclaimer/privacy boundary -> SafetyRouter -> IntentRouter -> RAG/survey/resources/support -> ResponsePolicy -> answer with boundaries and sources where relevant.

## Functional Requirements

- First screen shows product boundary and privacy notice.
- Chat supports knowledge Q&A, survey guidance, resource navigation and support conversation.
- Crisis language triggers fixed safety response.
- Survey is configuration-driven and deterministically scored.
- Feedback supports helpful, not helpful, inaccurate and unsafe.
- Developer evaluation scripts report actual small-set results.

## Non-Functional Requirements

- Python 3.11+, FastAPI, Streamlit, Pydantic v2.
- No GPU required for MVP.
- Config via environment variables.
- Sensitive text is not permanently stored by default.
- Tests are repeatable in Mock mode.

## User Stories

- As a user, I want to ask a knowledge question and see cited source snippets.
- As a user, I want to complete a short self-understanding survey and see a non-diagnostic interpretation.
- As a user in possible danger, I want the system to stop normal conversation and direct me to real-world help.
- As a reviewer, I want to run the project without an API key.

## Acceptance Criteria

- `pytest` passes.
- `/health` returns status ok.
- `/docs` exposes OpenAPI docs when the API is running.
- Crisis examples return the fixed crisis template.
- RAG unknown questions refuse rather than invent.
- Survey invalid input returns validation error.

## Out of Scope

- Medical diagnosis.
- Medication recommendations.
- Clinical-grade triage.
- Real user analytics or claims of deployment scale.

