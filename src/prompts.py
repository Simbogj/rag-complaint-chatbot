"""Task 3: Prompt templates for grounded complaint analysis."""

from __future__ import annotations

ANALYST_SYSTEM_PROMPT = """You are a financial analyst assistant for CrediTrust Financial.
Your task is to answer questions about customer complaints using ONLY the retrieved
complaint excerpts below. Be concise, evidence-based, and professional.

Rules:
- Ground every claim in the provided context.
- Mention product categories or issue types when relevant.
- If the context does not contain enough information, say so clearly.
- Do not invent facts, statistics, or complaint details."""


def format_context(chunks: list[str]) -> str:
    if not chunks:
        return "No complaint excerpts were retrieved."

    blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        blocks.append(f"[Excerpt {idx}]\n{chunk.strip()}")
    return "\n\n".join(blocks)


def build_rag_prompt(question: str, context_chunks: list[str]) -> str:
    context = format_context(context_chunks)
    return (
        f"{ANALYST_SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question.strip()}\n\n"
        "Answer:"
    )
