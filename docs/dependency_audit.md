# Dependency Audit

Date: 2026-07-15

## Environment

- Clean QA virtual environment: `.qa_venv`
- Python: 3.13.5
- Docker image runtime: Python 3.11 slim

## Results

| Check | Result | Notes |
| --- | --- | --- |
| `python -m pip install -e ".[dev]"` | PASS | Clean venv installation completed |
| `python -m pip check` | PASS | No broken requirements |
| `pip-audit -l` | PASS | No known vulnerabilities found |
| Local project package audit | SKIPPED BY TOOL | `mental-health-support` is local and not on PyPI |
| `bandit -q -r app tests -x tests` | PASS | No findings |

## Global Environment Note

An earlier audit against the global Anaconda environment produced many unrelated vulnerabilities from packages outside this project. That result is not used as the project dependency verdict. The project verdict is based on the clean `.qa_venv` environment installed from `pyproject.toml`.

## Follow-Up

- Pin exact versions or add a lock file before production deployment.
- Re-run `pip-audit` in CI after dependency updates.
- Consider Docker image scanning in CI.

