# Legacy Audit

Date: 2026-07-15

## Current Structure Before Refactor

- `chain.py`: CLI-style orchestration combining Qwen-compatible LLM, LangChain RAG, BERT classification and final report generation.
- `rag_preprocess.py`: LangChain FAISS preprocessing script with hardcoded document paths.
- `tran_bert.py`: BERT fine-tuning, evaluation and inference helpers.
- `data_process.ipynb`, `data_preprocess.ipynb`: notebook-based data processing.
- `data/`: CSV files; one file is large and appears to contain sensitive mental-health forum-style text.
- `README.md`: mojibake/encoding damage in observed terminal output and product positioning as an auxiliary diagnostic/evaluation system.

## Reusable Parts

- General idea of separating conversation, retrieval and classifier tasks.
- BERT training utilities can be preserved for optional offline comparison.
- RAG preprocessing concept can inform a cleaner ingestion pipeline.

## Technical Debt

- Core logic concentrated in `chain.py`.
- API keys, model names, local paths and device settings are hardcoded.
- Startup fails when `DASHSCOPE_API_KEY` is empty.
- Old LangChain abstractions and `allow_dangerous_deserialization=True` increase maintenance and security risk.
- No FastAPI service boundary, no UI product flow, no structured tests.
- No reliable mock mode for API-key-free demos.

## Safety Risks

- Old prompts and output modules use diagnosis/evaluation/risk language.
- BERT output is described as mental-health risk classification.
- LLM report generation may produce diagnostic conclusions and unsupported care advice.
- No deterministic crisis routing before normal conversation.
- No policy layer to block diagnostic claims or medication instructions.
- Logs/prints may expose full sensitive user text.

## Dependency Issues

- Required dependencies are implicit and not pinned in a project file.
- GPU/CUDA assumptions appear in comments and device selection.
- Local model and RAG folders are required but not included.

## README vs Code Differences

- README references files and directories not present in the repository, including model directories, `rag/`, `saved_rag/`, and `data_process.py`.
- README describes a diagnostic report pipeline, while the target v2 must be non-diagnostic.

## Hardcoded Secrets

- No real key was found, but empty `DASHSCOPE_API_KEY = ""` placeholders exist in source files.
- API configuration is embedded in code instead of environment variables.

## Broken or Non-Runnable Code

- `chain.py` raises on empty DashScope key.
- Local model and vector index paths are missing.
- Notebook-derived data paths do not match the described scripts.
- Some CSV files are two bytes long, suggesting placeholders or corrupted files.

## Test Gaps

- No automated unit, integration, safety, survey or RAG tests existed.
- No safety regression test for crisis handling, diagnostic claims, medication advice or logging leakage.

## Migration Decisions

- Preserve old files under `legacy/` instead of deleting them.
- Do not use legacy data as default demo data due to unclear source/license/privacy status.
- Rebuild v2 around FastAPI, Streamlit, deterministic SafetyRouter, config-driven SurveyEngine, cited retrieval and MockLLM.
- Keep BERT as disabled optional classifier only; do not use it for clinical risk prediction.

