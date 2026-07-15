# Final Acceptance Matrix

Date: 2026-07-15
Branch: `qa/final-acceptance-v2`

Status values:

- PASS: implemented and verified by code, test, command, or manual smoke check.
- PARTIAL: implemented with a documented MVP limitation.
- GAP: not implemented; listed in `docs/remaining_gaps.md`.
- NOT_EXECUTED: planned check not run in this QA cycle.

## Summary

| Area | PASS | PARTIAL | GAP | NOT_EXECUTED |
| --- | ---: | ---: | ---: | ---: |
| Product boundary and safety | 14 | 1 | 0 | 0 |
| Architecture and configuration | 18 | 4 | 1 | 0 |
| Backend API | 10 | 1 | 0 | 0 |
| Streamlit demo | 9 | 1 | 0 | 0 |
| Evaluation and tests | 14 | 3 | 0 | 1 |
| Docker and operations | 8 | 2 | 1 | 0 |
| Documentation and portfolio material | 12 | 1 | 0 | 0 |

Overall verdict: READY_WITH_LIMITATIONS.

The project is runnable and demoable, but it should not be described as production-ready or clinically validated. The main limitations are the lightweight TF-IDF retrieval backend, conservative safety false positives on academic/news crisis mentions, no automated browser visual regression, and Docker not yet hardened with a non-root runtime user.

## Product Boundary And Safety

| ID | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| S1 | Non-diagnostic positioning | PASS | README, UI disclaimer, prompts, response policy |
| S2 | No disease probability claims | PASS | `ResponsePolicy`, safety tests |
| S3 | No future self-harm prediction | PASS | Fixed crisis flow, no predictive labels |
| S4 | No low/medium/high future risk label | PASS | API exposes safety action only; UI does not label user risk |
| S5 | No medication dose or stop-medication advice | PASS | Safety router and red-team tests |
| S6 | Not described as doctor or therapist | PASS | README, UI, prompts |
| S7 | Survey score not diagnosis | PASS | Survey copy, service boundary, tests |
| S8 | Crisis flow fixed before ordinary chat | PASS | Unit tests, integration tests, Docker smoke |
| S9 | Default no permanent sensitive chat storage | PASS | In-memory session repository |
| S10 | Feedback does not store full sensitive text by default | PASS | Redacted feedback repository |
| S11 | Logs avoid full sensitive text | PASS | Privacy helpers and no app-level full-text chat logs |
| S12 | Demo data synthetic/project-authored | PASS | `knowledge/sources`, evaluation datasets |
| S13 | No fabricated clinical validation or deployment claims | PASS | README and docs |
| S14 | Prompt injection cannot bypass crisis routing | PASS | `tests/safety/test_safety_red_team.py` |
| S15 | Academic/news crisis mentions handled precisely | PARTIAL | Current deterministic router over-triggers by design |

## Architecture And Configuration

| ID | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| A1 | FastAPI backend | PASS | `app/main.py`, `app/api/*` |
| A2 | Streamlit frontend | PASS | `ui/streamlit_app.py` |
| A3 | Pydantic v2 schemas | PASS | `app/domain/models.py`, `app/llm/schemas.py` |
| A4 | Repository pattern / SQLite demo store | PASS | Feedback and session repositories |
| A5 | Central config | PASS | `app/core/config.py` |
| A6 | Sensitive values from environment | PASS | `.env.example`, Docker `env_file` |
| A7 | OpenAI-compatible provider | PASS | `app/llm/openai_compatible.py` |
| A8 | DashScope-compatible mode possible via base URL/model | PASS | Provider uses OpenAI-compatible endpoint settings |
| A9 | Mock mode without API key | PASS | Default `LLM_PROVIDER=mock`, tests |
| A10 | Structured LLM output with Pydantic validation | PASS | Provider tests cover coercion/fallback |
| A11 | No business hardcoded API key/model | PASS | Secret scan clean |
| A12 | Prompt text separated from core services | PASS | `prompts/`, provider interface |
| A13 | RAG load/split/retrieve separated | PASS | `app/retrieval/*` |
| A14 | Source metadata in answers | PASS | `Source`, RAG tests, UI source expander |
| A15 | FAISS or Chroma default backend | GAP | Current backend is TF-IDF for MVP determinism |
| A16 | SafetyRouter deterministic plus LLM fallback | PASS | `SafetyRouter` |
| A17 | Region-configurable support resource copy | PARTIAL | Generic resources exist; no full hotline database |
| A18 | Survey engine config-driven | PASS | `surveys/example_wellbeing_survey.json` |
| A19 | Optional BERT isolated and disabled by default | PASS | `app/classifiers/optional_bert.py`, tests |
| A20 | Unified error handling | PARTIAL | Pydantic/FastAPI validation active; no custom global exception envelope |
| A21 | Bounded memory | PASS | Session history limit |
| A22 | CORS configurable | PASS | `CORS_ALLOW_ORIGINS` |
| A23 | Retry/backoff for real LLM calls | PARTIAL | Timeout and structured fallback exist; no retry/backoff loop |

## MVP Features

| ID | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| M1 | First-page boundary/privacy notice | PASS | Streamlit top-level `st.info` |
| M2 | Knowledge Q&A mode | PASS | `/chat` + RAG |
| M3 | Stress/emotion self-understanding survey | PASS | 10-question survey |
| M4 | Professional support navigation | PASS | Resource service |
| M5 | Supportive conversation mode | PASS | Mock/LLM fallback |
| M6 | Multi-turn current-session memory | PASS | In-memory session history |
| M7 | Clear conversation and survey data | PASS | UI buttons and DELETE API |
| M8 | Feedback buttons | PASS | UI and `/feedback` |
| M9 | Latency exposed per response | PASS | `latency_ms` |
| M10 | Swagger/OpenAPI | PASS | `/docs` smoke 200 |
| M11 | Health check | PASS | `/health` smoke 200 |

## Evaluation

| ID | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| E1 | Intent accuracy/macro F1/confusion matrix/recall | PASS | `evaluation/evaluate_intent.py` |
| E2 | RAG 30 synthetic questions | PASS | `evaluation/datasets/rag_questions.json` |
| E3 | RAG hit/citation/latency metrics | PASS | `evaluation/evaluate_rag.py` |
| E4 | Manual groundedness scoring | NOT_EXECUTED | Template exists; not scored in this QA cycle |
| E5 | Safety crisis recall and false-positive rate | PASS | `evaluation/evaluate_safety.py` |
| E6 | Diagnosis and medication violation checks | PASS | Evaluation and tests |
| E7 | Prompt injection safety cases | PASS | Red-team test |
| E8 | Survey deterministic/boundary/invalid tests | PASS | Survey tests |
| E9 | Log leakage automated test | PARTIAL | Privacy helper tests and scans; no full log capture harness |
| E10 | Optional BERT nonblocking behavior | PASS | Optional classifier tests |

## Operations

| ID | Requirement | Status | Evidence |
| --- | --- | --- | --- |
| O1 | Python 3.11+ runnable | PASS | Docker uses Python 3.11; QA venv used Python 3.13 |
| O2 | Cross-platform local commands | PASS | Makefile plus documented PowerShell commands |
| O3 | Dockerfile | PASS | Docker build succeeded |
| O4 | docker-compose | PASS | Services `api`, `ui` start |
| O5 | `.dockerignore` excludes local env/caches | PASS | Build context reduced to about 108 kB after QA docs were added |
| O6 | Non-root Docker runtime user | GAP | Dockerfile still runs as root |
| O7 | Dependency audit | PASS | Clean venv `pip-audit -l`: no known vulnerabilities |
| O8 | Bandit scan | PASS | No findings in `app` |
| O9 | UI automated browser regression | PARTIAL | HTTP smoke only; no Playwright screenshots |
| O10 | Containers stopped after QA | PARTIAL | Services left running for user demo |
