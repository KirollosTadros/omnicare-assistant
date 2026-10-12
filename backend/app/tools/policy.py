from app.rag.ingest import get_embedder, get_store


def search_policy(query: str, k: int = 2) -> list[dict]:
    return get_store().search(query, get_embedder().model, k=k)
