"""Retrieval data structures."""

from pydantic import BaseModel


class KnowledgeDocument(BaseModel):
    title: str
    source_id: str
    text: str
    license: str = "project-authored summary"


class RetrievedChunk(BaseModel):
    title: str
    source_id: str
    chunk_id: str
    text: str
    score: float
