"""Task 3: Semantic retriever over complaint vector stores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
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
