"""Prompt templates for the complaint RAG pipeline."""

ANALYST_PROMPT = """You are a financial analyst assistant for CrediTrust Financial.
Your task is to answer questions about customer complaints using ONLY the retrieved
complaint excerpts below.

Rules:
- Base your answer strictly on the provided context.
- Synthesize patterns and themes when multiple complaints relate to the question.
- Mention product categories or issue types when relevant.
- If the context does not contain enough information, say so clearly.
- Be concise, professional, and actionable for product and compliance teams.

Context:
{context}

Question: {question}

Answer:"""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the LLM."""
    if not chunks:
        return "No relevant complaint excerpts were retrieved."

    parts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        header = (
            f"[Source {idx} | {meta.get('product_category', 'Unknown product')} "
            f"| Issue: {meta.get('issue', 'N/A')}]"
        )
        parts.append(f"{header}\n{chunk['text']}")

    return "\n\n".join(parts)


def build_rag_prompt(context: str, question: str) -> str:
    """Build the full analyst prompt from context and user question."""
    return ANALYST_PROMPT.format(context=context.strip(), question=question.strip())
