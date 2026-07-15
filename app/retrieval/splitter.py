"""Simple text splitter for small demo knowledge bases."""


def split_text(text: str, chunk_size: int = 420, overlap: int = 60) -> list[str]:
    chunks: list[str] = []
    start = 0
    clean = " ".join(text.split())
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(0, end - overlap)
    return chunks
