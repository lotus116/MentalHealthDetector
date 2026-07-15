.PHONY: install run-api run-ui test lint typecheck evaluate

install:
	python -m pip install -e ".[dev]"

run-api:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	streamlit run ui/streamlit_app.py

test:
	pytest

lint:
	ruff check .

typecheck:
	mypy app

evaluate:
	python evaluation/evaluate_intent.py && python evaluation/evaluate_safety.py && python evaluation/evaluate_rag.py

