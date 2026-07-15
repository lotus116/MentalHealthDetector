FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY app app
COPY ui ui
COPY knowledge knowledge
COPY resources resources
COPY surveys surveys
COPY prompts prompts
RUN pip install --no-cache-dir .
EXPOSE 8000 8501
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
