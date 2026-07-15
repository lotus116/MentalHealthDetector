"""Cited knowledge-answer service."""

from app.core.config import Settings
from app.domain.models import Source
from app.retrieval.ingest import load_manifest
from app.retrieval.vector_store import LocalVectorStore


class RagService:
    """Retrieves project-authored summaries and returns cited answers."""

    def __init__(self, settings: Settings):
        docs = load_manifest(settings.knowledge_manifest_path)
        self.store = LocalVectorStore(docs)

    def answer(self, question: str) -> tuple[str, list[Source]]:
        chunks = self.store.search(question, k=3)
        if not chunks or chunks[0].score < 0.06:
            return (
                "我没有在当前示例知识库中找到足够依据，因此不能补充医学结论。"
                "你可以换一种问法，或咨询合格专业人员获取个体化建议。",
                [],
            )
        sources = [
            Source(title=c.title, source_id=c.chunk_id, snippet=c.text[:180])
            for c in chunks
        ]
        answer = (
            "根据当前知识库，相关信息提示："
            f"{chunks[0].text[:260]} "
            "这只能作为一般信息，不能用于诊断或替代专业评估。"
        )
        return answer, sources

