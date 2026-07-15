# Test Evidence

Date: 2026-07-15

## Static Checks

```text
python -m compileall app tests evaluation
Result: PASS

ruff format --check .
Result: PASS, 58 files already formatted

ruff check .
Result: PASS, all checks passed

mypy app
Result: PASS, no issues in 41 source files
```

## Test Suite

```text
pytest -q
Result: PASS, 33 passed, 1 warning
```

Warnings:

- Starlette/FastAPI TestClient deprecation warning from dependency internals.

## Coverage

```text
pytest --cov=app --cov-report=term-missing --cov-report=html -q
Result: PASS, 33 passed, 86% total coverage
```

Lower-coverage areas:

- `app/llm/openai_compatible.py`: real external HTTP paths are not fully covered.
- `app/classifiers/base.py` and `app/classifiers/rules.py`: small abstractions/rule helpers not used heavily by the core path.

## Evaluation Scripts

```text
python evaluation/evaluate_intent.py
Result: PASS
accuracy: 1.0
macro_f1: 0.7142857142857143
```

```text
python evaluation/evaluate_safety.py
Result: PASS
accuracy: 0.8333333333333334
crisis_recall: 1.0
non_crisis_false_positive_rate: 0.25
diagnostic_expression_violations: 0
medication_advice_violations: 0
```

```text
python evaluation/evaluate_rag.py
Result: PASS
count: 30
retrieval_hit_rate: 0.9333333333333333
citation_completeness: 0.9333333333333333
groundedness_manual_score: not executed
```

## Docker Smoke

```text
docker compose config --services
Result: PASS, services: api, ui

docker compose up -d --build
Result: PASS

docker compose build
Result: PASS, build context about 108 kB after .dockerignore fix and QA docs were added

GET http://127.0.0.1:8000/health
Result: PASS, status ok

GET http://127.0.0.1:8501
Result: PASS, HTTP 200

GET http://127.0.0.1:8000/docs
Result: PASS, HTTP 200

POST http://127.0.0.1:8000/feedback
Result: PASS, {"stored": true}

POST http://127.0.0.1:8000/chat crisis smoke
Result: PASS with UTF-8 request bytes
```

## Security Scans

```text
pip check
Result: PASS

pip-audit -l
Result: PASS, no known vulnerabilities found

bandit -q -r app tests -x tests
Result: PASS

secret-like source scan
Result: PASS

local absolute path scan
Result: PASS
```
