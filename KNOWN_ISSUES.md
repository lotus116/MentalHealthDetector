# Known Issues

- Docker build was not executed in the current environment because the `docker` command is not installed or not on `PATH`.
- The deterministic SafetyRouter currently over-triggers on some academic/news mentions of self-harm terms. This is measured in `evaluation/evaluate_safety.py` as a non-crisis false positive.
- The MVP retrieval backend is a lightweight local TF-IDF vector store. FAISS/Chroma can be added behind the existing retrieval interface in a later iteration.

