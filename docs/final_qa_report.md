# Final QA Report

Date: 2026-07-15
Branch: `qa/final-acceptance-v2`

## Verdict

READY_WITH_LIMITATIONS.

The v2 application is runnable, testable, and demoable through Docker and local Python. It satisfies the non-diagnostic product boundary, safety routing, cited knowledge Q&A, deterministic survey, feedback, health check, OpenAPI, and MockLLM requirements. It should not be presented as production-ready or clinically validated.

## Fixes Made In This QA Pass

- Added configurable CORS origins instead of permissive wildcard CORS.
- Hid the raw API URL from the Streamlit demo; the UI now shows only system health status.
- Added `.dockerignore` exclusions for local virtual environments, coverage data, caches, `.env`, and legacy data.
- Added final red-team, optional classifier, RAG injection, and integration tests.
- Added `bandit` and `pip-audit` to development dependencies.
- Verified Docker build context reduced from about 210 MB to about 108 kB after excluding `.qa_venv`.

## Verified Commands

| Command | Result |
| --- | --- |
| `python -m compileall app tests evaluation` | PASS |
| `ruff format --check .` | PASS |
| `ruff check .` | PASS |
| `mypy app` | PASS |
| `pytest -q` | PASS, 33 passed |
| `pytest --cov=app --cov-report=term-missing --cov-report=html -q` | PASS, 86% total coverage |
| `python evaluation/evaluate_intent.py` | PASS |
| `python evaluation/evaluate_safety.py` | PASS |
| `python evaluation/evaluate_rag.py` | PASS |
| `pip check` in clean QA venv | PASS |
| `pip-audit -l` in clean QA venv | PASS, no known vulnerabilities; local package skipped because it is not on PyPI |
| `bandit -q -r app tests -x tests` | PASS |
| Secret-like text scan | PASS |
| Local absolute path scan | PASS |
| `docker compose build` | PASS |
| `docker compose up -d --build` | PASS |
| `GET /health` through Docker | PASS |
| `GET /docs` through Docker | PASS |
| `GET Streamlit :8501` through Docker | PASS |
| `POST /feedback` through Docker | PASS |
| `POST /chat` crisis smoke through Docker | PASS when UTF-8 request bytes are used |

## Actual Evaluation Results

- Intent synthetic set: 7 cases, accuracy 1.0, macro F1 0.7143. Some classes have no sample in the tiny set, so macro F1 is lower than accuracy.
- Safety synthetic set: 6 cases, accuracy 0.8333, crisis recall 1.0, non-crisis false positive rate 0.25, diagnostic expression violations 0, medication advice violations 0.
- RAG synthetic set: 30 questions, retrieval hit rate 0.9333, citation completeness 0.9333. Manual groundedness scoring was not executed.
- Coverage: 86% total line coverage for `app`.

## Important Limitations

- Retrieval uses a deterministic TF-IDF MVP backend, not FAISS or Chroma.
- Safety routing intentionally over-triggers on some academic/news crisis mentions.
- Real LLM behavior is schema-validated and fallback-protected, but content quality is still provider-dependent.
- Dockerfile does not yet run as a non-root user; Docker build logs show pip running as root.
- UI verification was HTTP smoke only; no automated screenshot or browser interaction regression was run.
- PowerShell string bodies can corrupt Chinese request text unless sent as UTF-8 bytes. Browser and Python clients do not have this issue in the tested paths.

## Demo State

Docker services were left running for the user:

- API: `http://127.0.0.1:8000`
- Streamlit: `http://127.0.0.1:8501`
- Swagger/OpenAPI: `http://127.0.0.1:8000/docs`
