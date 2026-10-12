from __future__ import annotations
from typing import List
from sentence_transformers import SentenceTransformer


class textEmbedding:

    def __init__(self, model_name: str="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def wordChunker(self, text: str, chunk_size: int= 300, overlap: int= 50, ) -> List[str]:

        words = text.split()
        if not words:
            return []
        chunks = []
        step = chunk_size - overlap

        for i in range(0, len(words), step):
            chunk_words = words[i: i + chunk_size]
            chunks.append(" ".join(chunk_words))
        return chunks

    def markdownChunker(self, text: str, source: str, max_words: int= 300) -> List[dict]:

        title = ""
        sections = []
        for line in text.splitlines():
            if line.startswith("## "):
                sections.append((line[3:].strip(), []))
            elif line.startswith("# ") and not sections and not title:
                title = line[2:].strip()
            elif sections:
                sections[-1][1].append(line)

        chunks = []
        for heading, lines in sections:
            body = "\n".join(lines).strip()
            # the heading is embedded with the body so its words help retrieval
            prefix = f"{title} — {heading}" if title else heading
            pieces = [body] if len(body.split()) <= max_words else self.wordChunker(body)
            for piece in pieces:
                chunks.append({"text": f"{prefix}\n{piece}", "section": heading, "source": source})
        return chunks

    def generateEmbeddings(self, rawText: str):
        chunks = self.wordChunker(rawText)

        embeddings = self.model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
        return chunks, embeddings