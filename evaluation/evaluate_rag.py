"""Evaluate small knowledge retrieval set."""

import json
import statistics
import time
from pathlib import Path

from app.core.config import get_settings
from app.services.rag_service import RagService


def main() -> dict:
    cases = json.loads(Path("evaluation/datasets/rag_questions.json").read_text(encoding="utf-8"))
    rag = RagService(get_settings())
    hits = 0
    citation_complete = 0
    latencies = []
    for case in cases:
        start = time.perf_counter()
        _, sources = rag.answer(case["question"])
        latencies.append((time.perf_counter() - start) * 1000)
        source_ids = {s.source_id.split("#")[0] for s in sources}
        hits += int(case["expected_source"] in source_ids)
        citation_complete += int(bool(sources))
    report = {
        "dataset": "30 synthetic RAG questions",
        "count": len(cases),
        "retrieval_hit_rate": hits / len(cases),
        "citation_completeness": citation_complete / len(cases),
        "source_coverage": "measured by expected source hit",
        "groundedness_manual_score": "not executed",
        "refusal_correctness": "covered by unit test for out-of-scope query",
        "latency_ms_avg": statistics.mean(latencies),
        "latency_ms_p95": sorted(latencies)[int(len(latencies) * 0.95) - 1],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()

