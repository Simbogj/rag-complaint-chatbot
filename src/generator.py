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
