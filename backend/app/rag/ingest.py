import logging
from functools import lru_cache
from pathlib import Path

from app import config
from app.rag.textEmbedding import textEmbedding
from app.rag.textReader import MarkdownReader, PDFReader
from app.rag.vectorStore import VectorStore

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedder() -> textEmbedding:
    return textEmbedding(config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    return VectorStore(collection_name=config.COLLECTION_NAME, db_path=str(config.CHROMA_DIR))


def load_chunks(policy_dir: Path) -> list[dict]:
    embedder = get_embedder()
    chunks = []

    for path in sorted(Path(policy_dir).iterdir()):
        suffix = path.suffix.lower()
        if suffix == ".md":
            file_chunks = embedder.markdownChunker(MarkdownReader(path).extract_text(), source=path.name)
            headings, part = [], 0
            for chunk in file_chunks:
                # a long section is split into several pieces that share one heading
                if not headings or chunk["section"] != headings[-1]:
                    headings.append(chunk["section"])
                    part = 1
                else:
                    part += 1
                chunk["id"] = f"{path.name}::section-{len(headings)}" + (f"-part-{part}" if part > 1 else "")
        elif suffix == ".pdf":
            pieces = embedder.wordChunker(PDFReader(path).extract_text())
            file_chunks = [
                {"text": piece, "section": f"chunk {i}", "source": path.name, "id": f"{path.name}::chunk-{i}"}
                for i, piece in enumerate(pieces, start=1)
            ]
        else:
            continue

        log.info("[ingest] %s -> %d chunks", path.name, len(file_chunks))
        chunks.extend(file_chunks)

    return chunks


def build_index() -> int:
    chunks = load_chunks(config.POLICY_DIR)
    store = get_store()
    store.reset_collection()
    if not chunks:
        log.warning("[ingest] no policy documents found in %s", config.POLICY_DIR)
        return 0

    embeddings = get_embedder().model.encode(
        [chunk["text"] for chunk in chunks], show_progress_bar=False, normalize_embeddings=True
    )
    store.add_chunks(
        [chunk["text"] for chunk in chunks],
        embeddings,
        metadatas=[{"section": chunk["section"], "source": chunk["source"]} for chunk in chunks],
        ids=[chunk["id"] for chunk in chunks],
    )
    return len(chunks)
