"""Semantic retriever over the complaint vector store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb

from src.config import COLLECTION_NAME, DEFAULT_TOP_K, EMBEDDING_MODEL, VECTOR_STORE_PATH


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    distance: float | None = None
    id: str | None = None


class ComplaintRetriever:
    """Embed queries and retrieve the most relevant complaint chunks from ChromaDB."""

    def __init__(
        self,
        persist_path: Path | str = VECTOR_STORE_PATH,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        self.persist_path = Path(persist_path)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self._embedding_model = None

        client = chromadb.PersistentClient(path=str(self.persist_path))
        self.collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def embed_query(self, query: str) -> list[float]:
        vector = self.embedding_model.encode(query, show_progress_bar=False)
        return vector.tolist()

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        product_category: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-k most similar complaint chunks for a natural-language query."""
        if self.chunk_count == 0:
            return []

        query_embedding = self.embed_query(query)

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if product_category:
            query_kwargs["where"] = {"product_category": {"$eq": product_category}}

        results = self.collection.query(**query_kwargs)

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        chunks: list[RetrievedChunk] = []
        for doc, meta, dist, chunk_id in zip(documents, metadatas, distances, ids):
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    metadata=meta or {},
                    distance=dist,
                    id=chunk_id,
                )
            )
        return chunks

    @property
    def chunk_count(self) -> int:
        return self.collection.count()
