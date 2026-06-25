"""End-to-end RAG pipeline for complaint analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import DEFAULT_TOP_K, LLM_MODEL, VECTOR_STORE_PATH
from src.generator import ComplaintGenerator
from src.prompts import build_rag_prompt, format_context
from src.retriever import ComplaintRetriever, RetrievedChunk


@dataclass
class RAGResponse:
    question: str
    answer: str
    sources: list[RetrievedChunk]
    prompt: str
    product_category: str | None = None


class RAGPipeline:
    """Combine retrieval and generation to answer complaint questions."""

    def __init__(
        self,
        retriever: ComplaintRetriever | None = None,
        generator: ComplaintGenerator | None = None,
        vector_store_path: Path | str = VECTOR_STORE_PATH,
        llm_model: str = LLM_MODEL,
        use_fallback: bool = False,
    ) -> None:
        self.retriever = retriever or ComplaintRetriever(persist_path=vector_store_path)
        self.generator = generator or ComplaintGenerator(
            model_name=llm_model,
            use_fallback=use_fallback,
        )

    def ask(
        self,
        question: str,
        product_category: str | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> RAGResponse:
        """Retrieve relevant chunks and generate an evidence-backed answer."""
        sources = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            product_category=product_category,
        )

        chunk_dicts = [{"text": s.text, "metadata": s.metadata} for s in sources]
        context = format_context(chunk_dicts)
        prompt = build_rag_prompt(context=context, question=question)
        answer = self.generator.generate(
            prompt=prompt,
            context=context,
            question=question,
        )

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            prompt=prompt,
            product_category=product_category,
        )

    def ask_batch(
        self,
        questions: list[dict],
        top_k: int = DEFAULT_TOP_K,
    ) -> list[RAGResponse]:
        """Run the pipeline on a list of question dicts with optional product filters."""
        responses: list[RAGResponse] = []
        for item in questions:
            responses.append(
                self.ask(
                    question=item["question"],
                    product_category=item.get("product_category"),
                    top_k=top_k,
                )
            )
        return responses
