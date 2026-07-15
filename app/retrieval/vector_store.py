"""Small local retrieval backend for the demo knowledge base."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.retrieval.schemas import KnowledgeDocument, RetrievedChunk
from app.retrieval.splitter import split_text


class LocalVectorStore:
    """Character n-gram TF-IDF store; deterministic and lightweight for tests."""

    def __init__(self, documents: list[KnowledgeDocument]):
        self.chunks: list[RetrievedChunk] = []
        for doc in documents:
            for idx, chunk in enumerate(split_text(doc.text)):
                self.chunks.append(
                    RetrievedChunk(title=doc.title, source_id=doc.source_id, chunk_id=f"{doc.source_id}#{idx}", text=chunk, score=0.0)
                )
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.matrix = self.vectorizer.fit_transform([c.text for c in self.chunks]) if self.chunks else None

    def search(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        if not self.chunks or self.matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked = scores.argsort()[::-1][:k]
        results: list[RetrievedChunk] = []
        for idx in ranked:
            chunk = self.chunks[int(idx)].model_copy(update={"score": float(scores[idx])})
            if chunk.score > 0:
                results.append(chunk)
        return results

