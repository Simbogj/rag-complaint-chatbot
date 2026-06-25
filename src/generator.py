"""Task 3: LLM generation for RAG answers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GenerationResult:
    answer: str
    model_name: str


class ComplaintGenerator:
    """Generate analyst answers from a RAG prompt."""

    def __init__(self, model_name: str = "google/flan-t5-base") -> None:
        self.model_name = model_name
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        from transformers import pipeline

        self._pipeline = pipeline(
            "text2text-generation",
            model=self.model_name,
            max_new_tokens=256,
        )
        return self._pipeline

    def generate(self, prompt: str) -> GenerationResult:
        try:
            generator = self._load_pipeline()
            output = generator(prompt[:3500])[0]["generated_text"]
            return GenerationResult(answer=output.strip(), model_name=self.model_name)
        except Exception:
            return GenerationResult(
                answer=_fallback_answer(prompt),
                model_name="fallback-extractive",
            )


def _fallback_answer(prompt: str) -> str:
    """Lightweight fallback when the LLM is unavailable."""
    marker = "Context:"
    question_marker = "Question:"
    answer_marker = "Answer:"

    context_part = prompt.split(question_marker)[0] if question_marker in prompt else prompt
    if marker in context_part:
        context_part = context_part.split(marker, 1)[1]

    excerpts = [
        block.replace("[Excerpt", "").strip()
        for block in context_part.split("[Excerpt")
        if block.strip() and "No complaint excerpts" not in block
    ]

    question = ""
    if question_marker in prompt:
        question = prompt.split(question_marker, 1)[1]
        if answer_marker in question:
            question = question.split(answer_marker, 1)[0]
        question = question.strip()

    if not excerpts:
        return (
            "I do not have enough retrieved complaint context to answer that question. "
            "Try rephrasing or narrowing the product category."
        )

    preview = " ".join(excerpts[:2])
    preview = preview[:700] + ("..." if len(preview) > 700 else "")
    return (
        f"Based on the retrieved complaints, here is a summary for '{question}': "
        f"{preview}"
    )
