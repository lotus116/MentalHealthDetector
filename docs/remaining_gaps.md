# Remaining Gaps

Date: 2026-07-15

These gaps are intentionally documented so the project does not overclaim.

## Product And Safety

- Safety routing is conservative and can over-trigger on academic or news mentions of crisis language.
- Regional emergency resource support is generic. It does not include a maintained global hotline directory.
- There is no clinical validation, clinician review, real user study, or production deployment evidence.

## Retrieval

- The current RAG backend uses deterministic TF-IDF retrieval for an interview-friendly MVP.
- FAISS or Chroma is not the default backend yet.
- Manual groundedness scoring was not executed in the final QA cycle.

## LLM Provider

- Real OpenAI-compatible provider calls have schema validation and fallbacks, but no retry/backoff loop.
- Real LLM content quality remains provider-dependent and should be tested before demos with a live key.

## Frontend QA

- Streamlit was checked by HTTP smoke and code review.
- No Playwright/browser screenshot regression was run.

## Operations

- Dockerfile still runs as root.
- No pinned lock file is committed.
- No CI pipeline is configured in this repository.

## Next Recommended Fixes

1. Add a non-root Docker user.
2. Add a lock file or pinned constraints file.
3. Add Playwright or Streamlit smoke automation for visual flows.
4. Add FAISS/Chroma as a configurable retrieval backend.
5. Expand the safety and intent evaluation datasets.

