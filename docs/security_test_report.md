# Security And Safety Test Report

Date: 2026-07-15

This report covers application safety boundaries, not clinical effectiveness.

## Automated Safety Coverage

| Area | Status | Evidence |
| --- | --- | --- |
| Explicit crisis expression routes to fixed template | PASS | Unit, integration, Docker smoke |
| Prompt injection cannot bypass crisis routing | PASS | `tests/safety/test_safety_red_team.py` |
| Medication dose/stop-medication requests refused | PASS | Safety tests |
| Diagnostic and disease-probability phrases blocked | PASS | Response policy tests |
| Low mood without immediate danger continues normal support | PASS | Red-team test |
| Survey score cannot be changed by LLM | PASS | Survey service tests |
| Feedback avoids storing full sensitive text | PASS | Feedback repository behavior |
| Secret-like values absent from source/docs scope | PASS | QA scan |
| Local absolute paths absent from committed source/docs scope | PASS | QA scan |

## Safety Evaluation Results

Dataset: tiny synthetic safety set, 6 cases.

- Accuracy: 0.8333
- Crisis recall: 1.0
- Non-crisis false positive rate: 0.25
- Diagnostic expression violations: 0
- Medication advice violations: 0

Known limitation: academic or news-style discussion containing explicit crisis terms can be conservatively routed to the fixed crisis response. This is safer than under-triggering for the MVP, but it reduces precision.

## Manual Docker Smoke

- English crisis text: routed to `fixed_crisis_response`.
- Chinese crisis text: routed to `fixed_crisis_response` when the request body was explicitly encoded as UTF-8 bytes.
- Direct PowerShell string body without UTF-8 bytes can corrupt Chinese text before it reaches the API. This is a client invocation issue; browser and Python JSON clients send UTF-8 correctly.

## Not Claimed

- No clinical validation.
- No prediction of future suicide or self-harm.
- No disease diagnosis.
- No medication recommendation.
- No production-grade regional emergency hotline directory.

