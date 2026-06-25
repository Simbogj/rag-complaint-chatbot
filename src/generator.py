<<<<<<< HEAD
"""LLM generation for RAG answers."""

from __future__ import annotations

from src.config import LLM_MODEL, MAX_NEW_TOKENS


class ComplaintGenerator:
    """Generate analyst answers from RAG prompts using a Hugging Face seq2seq model."""

    def __init__(
        self,
        model_name: str = LLM_MODEL,
        max_new_tokens: int = MAX_NEW_TOKENS,
        use_fallback: bool = False,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.use_fallback = use_fallback
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._pipeline = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
            )
        return self._pipeline

    def generate(self, prompt: str, context: str = "", question: str = "") -> str:
        """Generate an answer from the RAG prompt, with optional extractive fallback."""
        if self.use_fallback:
            return self._fallback_answer(context, question)

        try:
            output = self.pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            return output[0]["generated_text"].strip()
        except Exception:
            return self._fallback_answer(context, question)

    @staticmethod
    def _fallback_answer(context: str, question: str) -> str:
        """
        Extractive fallback when the LLM backend is unavailable.

        Summarizes retrieved excerpts so the pipeline remains testable offline.
        """
        if not context.strip() or context.startswith("No relevant"):
            return (
                "I don't have enough information in the retrieved complaints to "
                f"answer: {question}"
            )

        excerpts = [part.strip() for part in context.split("\n\n") if part.strip()]
        preview = excerpts[:3]
        joined = " | ".join(preview)
        if len(joined) > 900:
            joined = joined[:897] + "..."

        return (
            "Based on the retrieved complaint excerpts, customers frequently report "
            f"issues related to your question. Representative excerpts: {joined}"
        )
=======
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
>>>>>>> task-2/chunking-embeddings
