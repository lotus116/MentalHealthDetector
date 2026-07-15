# Migration Notes

- Legacy implementation was moved to `legacy/` with original files preserved.
- Legacy data is not used by default because source, consent and license are unclear.
- Old diagnostic prompts were not reused.
- BERT logic is represented by `app/classifiers/optional_bert.py` as a disabled adapter. It is not a clinical risk model.
- RAG was rebuilt with project-authored summaries and source metadata.
- API keys and model names moved to environment-driven configuration.
- LangChain-dependent code is isolated in `legacy/` and not imported by v2.

