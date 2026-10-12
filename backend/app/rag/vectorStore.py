from __future__ import annotations
import chromadb
from typing import List, Optional
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__ (self, collection_name: str="pdf_collection", db_path: str="./doc_db"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"})

    def reset_collection(self):
            """Deletes the current collection and recreates an empty one."""
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"})

    def add_chunks(self, chunks: List[str], embeddings, metadatas: Optional[List[dict]]=None, ids: Optional[List[str]]=None):

        if ids is None:
            ids = [f"chunk_{i}" for i in range(len(chunks))]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas)
        print(f"Added {len(chunks)} chunks to DB")

    def search(self, query: str, embedding_model: SentenceTransformer, k: int=2) -> List[dict]:
        query = embedding_model.encode(query, normalize_embeddings=True).tolist()
        results = self.collection.query(query_embeddings=[query], n_results=k)

        metadatas = results["metadatas"][0]
        return [
            {
                "text": text,
                "section": (metadata or {}).get("section", ""),
                "source": (metadata or {}).get("source", ""),
                "distance": distance,
            }
            for text, metadata, distance in zip(results["documents"][0], metadatas, results["distances"][0])
        ]
