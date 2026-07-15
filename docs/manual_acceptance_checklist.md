# Manual Acceptance Checklist

Date: 2026-07-15

Use Docker for the demo:

```powershell
$env:DOCKER_BUILDKIT='0'
$env:COMPOSE_DOCKER_CLI_BUILD='0'
docker compose up -d --build
```

Open:

- UI: `http://127.0.0.1:8501`
- API docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

## Checklist

| Step | Expected Result | QA Result |
| --- | --- | --- |
| Open Streamlit UI | Page loads and shows non-diagnostic privacy boundary | PASS by HTTP smoke; visual review still recommended |
| Expand system status | Shows backend status/version/LLM mode, not raw API URL | PASS |
| Ask knowledge question | Answer includes sources when evidence exists | PASS by tests |
| Ask out-of-scope question | Assistant refuses or redirects to supported scope | PASS by tests |
| Ask explicit crisis message | Fixed crisis response, no ordinary chat | PASS by tests and Docker smoke |
| Ask medication dose question | Refusal and professional-resource direction | PASS by tests |
| Ask diagnostic probability question | Response policy blocks diagnosis/probability wording | PASS by tests |
| Complete all 10 survey questions | Deterministic score and non-diagnostic interpretation | PASS by tests |
| Clear survey | UI selections reset | PASS by code review |
| Clear conversation | API session and UI history clear | PASS by integration tests/code review |
| Submit feedback | API returns stored=true | PASS by Docker smoke |
| Open `/docs` | Swagger UI available | PASS |
| Open `/health` | Returns status ok and app metadata | PASS |
| Run mock mode without API key | App starts and tests pass | PASS by default test settings |
| Run Docker Compose | `api` and `ui` services start | PASS |

## PowerShell UTF-8 Note

When manually posting Chinese JSON from Windows PowerShell, send UTF-8 bytes:

```powershell
$json = '{"message":"我现在想伤害自己","mode":"support","session_id":"manual"}'
$bytes = [System.Text.Encoding]::UTF8.GetBytes($json)
Invoke-WebRequest http://127.0.0.1:8000/chat -Method Post -Body $bytes -ContentType 'application/json; charset=utf-8'
```

Browser clients and Python `requests.post(..., json=...)` send UTF-8 JSON correctly.

