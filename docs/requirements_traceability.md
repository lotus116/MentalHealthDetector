# Requirements Traceability Check

Date: 2026-07-15

This document checks the v2 implementation against the original refactor instructions. It separates implemented behavior from MVP tradeoffs so the project does not overclaim.

## Safety Boundary

| Requirement | Status | Evidence |
| --- | --- | --- |
| No diagnosis claims | Implemented | `ResponsePolicy`, `tests/safety/test_response_policy.py`, README disclaimer |
| No disease probabilities | Implemented | Response policy and product copy prohibit probability claims |
| No future suicide/self-harm prediction | Implemented | Safety flow uses fixed response, no prediction labels |
| No low/medium/high future-risk labels | Implemented | API returns safety action, not user-facing future-risk label |
| No medication/dose/stop-medication advice | Implemented | `SafetyRouter`, `ResponsePolicy`, tests |
| Not described as doctor/therapist | Implemented | README, UI disclaimer, prompts |
| Survey scores not diagnosis | Implemented | `SurveyService`, survey copy, tests |
| Crisis expression triggers fixed flow | Implemented | `SafetyRouter`, `CRISIS_RESPONSE`, integration test |
| No permanent sensitive conversation storage by default | Implemented | `SessionRepository` is in-memory; feedback stores redacted preview/digest |
| Logs do not record full sensitive text | Implemented for app code | `privacy.py`; app code does not log full chat text |
| UI pages show disclaimer | Implemented | Streamlit top-level disclaimer |
| Demo data synthetic/project-authored | Implemented | `knowledge/sources/`; legacy data excluded |
| No fabricated clinical validation/users/deployment | Implemented | README and docs explicitly avoid such claims |

## Architecture

| Requirement | Status | Evidence / Note |
| --- | --- | --- |
| FastAPI backend | Implemented | `app/main.py`, `app/api/*` |
| Pydantic v2 models | Implemented | `app/domain/models.py`, `app/llm/schemas.py` |
| SQLite or repository pattern | Implemented | `FeedbackRepository`, `SessionRepository` |
| Streamlit frontend | Implemented | `ui/streamlit_app.py` |
| Unified LLM provider interface | Implemented | `app/llm/base.py`, Mock and OpenAI-compatible providers |
| API keys from environment | Implemented | `app/core/config.py`, `.env.example`, Compose `env_file` |
| Mock mode without API key | Implemented | `MockLLM`, default settings |
| Structured JSON/Pydantic LLM outputs | Implemented | `LLMProvider.structured`, schema validation |
| Prompt/model names not hardcoded in business services | Mostly implemented | Model names are settings-driven; prompt names are constants passed to provider |
| RAG loading/splitting/retrieval separated | Implemented | `app/retrieval/*`, `RagService` |
| Default FAISS/Chroma | Not implemented in MVP | Current backend is local TF-IDF for deterministic demos; documented as known limit |
| Source metadata and cited answers | Implemented | `Source` model, UI source expander |
| SafetyRouter deterministic + LLM layer | Implemented | deterministic rules plus structured LLM fallback |
| Region-configurable emergency resources | Implemented for generic guidance | `resources/support_resources.json`; no hardcoded hotline numbers |
| SurveyEngine config-driven | Implemented | 10-question JSON survey |
| Pause/resume/clear survey | Partially implemented | Streamlit session state preserves page choices; clear button implemented |
| Optional BERT isolated/disabled | Implemented | `app/classifiers/optional_bert.py`, legacy preserved |

## MVP Features

| Feature | Status |
| --- | --- |
| First-screen boundary/privacy notice | Implemented |
| User modes: knowledge, survey, resources, support | Implemented |
| Multi-turn conversation, session-only by default | Implemented; history is bounded in memory and passed to supportive LLM calls |
| RAG answer source title and chunk id | Implemented in API and Streamlit |
| Structured survey engine | Implemented |
| Programmatic survey scoring | Implemented |
| LLM cannot change raw score | Implemented by service boundary |
| Explicit crisis expression detection | Implemented |
| Fixed crisis template | Implemented |
| Clear session and survey data | Implemented in UI/API |
| Feedback buttons | Implemented |
| Admin/developer evaluation page or CLI | Implemented: Streamlit evaluation tab and CLI scripts |
| Latency/citation/safety evaluation | Implemented in evaluation scripts; UI shows per-response latency |
| Health check API | Implemented |
| Swagger/OpenAPI | Implemented by FastAPI |
| No API key demo | Implemented |

## Evaluation

| Area | Status |
| --- | --- |
| Intent accuracy, macro F1, confusion matrix, recall | Implemented |
| RAG 30 synthetic questions | Implemented |
| RAG hit rate, citation completeness, latency | Implemented |
| Groundedness manual scoring | Not executed; template provided |
| Safety dataset categories | Implemented as small synthetic set; can be expanded |
| Survey deterministic/boundary/invalid tests | Implemented |
| Log leakage tests | Partially implemented via privacy helpers; dedicated automated log capture can be added |

## Current Gaps Before Claiming Production Readiness

- Replace local TF-IDF retrieval with FAISS or Chroma if strict adherence to the recommended RAG backend is required.
- Expand safety evaluation and reduce false positives for academic/news mentions.
- Expand `resources/support_resources.json` if the demo needs more regions.
- Add a dedicated automated log-capture test for sensitive text leakage.
- Add full UI screenshot regression testing if this becomes a portfolio artifact with visual QA requirements.
