"""Knowledge manifest loader."""

import json
from pathlib import Path

from app.retrieval.schemas import KnowledgeDocument


def load_manifest(path: Path) -> list[KnowledgeDocument]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    base = path.parent
    docs: list[KnowledgeDocument] = []
    for item in raw.get("documents", []):
        text_path = base / item["path"]
        docs.append(
            KnowledgeDocument(
                title=item["title"],
                source_id=item["source_id"],
                text=text_path.read_text(encoding="utf-8"),
                license=item.get("license", "project-authored summary"),
            )
        )
    return docs
