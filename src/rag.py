"""Task 3: End-to-end RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from src.generator import ComplaintGenerator, GenerationResult
from src.prompts import build_rag_prompt
from src.retriever import ComplaintRetriever, RetrievedChunk


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list[RetrievedChunk]
    model_name: str


class ComplaintRAG:
    """Retrieve relevant complaints and generate a grounded answer."""

    def __init__(
        self,
        retriever: ComplaintRetriever,
        generator: ComplaintGenerator | None = None,
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.generator = generator or ComplaintGenerator()
        self.top_k = top_k

    def ask(
        self,
        question: str,
        product_category: str | None = None,
        top_k: int | None = None,
    ) -> RAGResponse:
        k = top_k or self.top_k
        sources = self.retriever.retrieve(
            question=question,
            top_k=k,
            product_category=product_category,
        )
        prompt = build_rag_prompt(question, [source.text for source in sources])
        generation: GenerationResult = self.generator.generate(prompt)
        return RAGResponse(
            question=question,
            answer=generation.answer,
            sources=sources,
            model_name=generation.model_name,
        )
