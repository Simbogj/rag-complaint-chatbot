<<<<<<< HEAD
"""Semantic retriever over the complaint vector store."""
=======
"""Task 3: Semantic retriever over complaint vector stores."""
>>>>>>> task-2/chunking-embeddings

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
<<<<<<< HEAD
from typing import Any

import chromadb

from src.config import COLLECTION_NAME, DEFAULT_TOP_K, EMBEDDING_MODEL, VECTOR_STORE_PATH
=======

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    PREBUILT_COLLECTION,
    PREBUILT_VECTOR_STORE_PATH,
    TASK2_COLLECTION,
    TASK2_VECTOR_STORE_PATH,
)
from src.load_prebuilt_store import get_prebuilt_collection
>>>>>>> task-2/chunking-embeddings


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
<<<<<<< HEAD
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
=======
    score: float
    metadata: dict


class ComplaintRetriever:
    """Embed a question and retrieve the top-k complaint chunks."""

    def __init__(
        self,
        collection,
        model_name: str = EMBEDDING_MODEL,
    ) -> None:
        self.collection = collection
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @classmethod
    def from_task2_store(cls, persist_path: Path = TASK2_VECTOR_STORE_PATH) -> "ComplaintRetriever":
        client = chromadb.PersistentClient(path=str(persist_path))
        collection = client.get_collection(name=TASK2_COLLECTION)
        return cls(collection)

    @classmethod
    def from_prebuilt_store(
        cls,
        persist_path: Path = PREBUILT_VECTOR_STORE_PATH,
        max_chunks: int | None = 5_000,
    ) -> "ComplaintRetriever":
        collection = get_prebuilt_collection(persist_path=persist_path, max_chunks=max_chunks)
        return cls(collection)

    def retrieve(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        product_category: str | None = None,
    ) -> list[RetrievedChunk]:
        query_embedding = self.model.encode(question, show_progress_bar=False).tolist()
        where = {"product_category": product_category} if product_category else None

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks: list[RetrievedChunk] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            chunks.append(
                RetrievedChunk(
                    text=text,
                    score=1.0 - float(distance),
                    metadata=metadata or {},
                )
            )
        return chunks
>>>>>>> task-2/chunking-embeddings
