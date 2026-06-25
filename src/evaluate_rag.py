"""
Task 3: Qualitative RAG evaluation runner.

Usage:
<<<<<<< HEAD
    python -m src.evaluate_rag --fallback
    python -m src.evaluate_rag --output reports/rag_evaluation.md
=======
    python -m src.evaluate_rag
    python -m src.evaluate_rag --max-chunks 5000
>>>>>>> task-2/chunking-embeddings
"""

from __future__ import annotations

import argparse
import json
<<<<<<< HEAD
from datetime import datetime, timezone
from pathlib import Path

from src.config import VECTOR_STORE_PATH
from src.rag import RAGPipeline
from src.retriever import ComplaintRetriever, RetrievedChunk

EVALUATION_QUESTIONS = [
    {
        "question": "Why are people unhappy with credit cards?",
        "product_category": "Credit Card",
    },
    {
        "question": "What billing or fee issues do credit card customers report most often?",
        "product_category": "Credit Card",
    },
    {
        "question": "What are the main complaints about personal loans?",
        "product_category": "Personal Loan",
    },
    {
        "question": "Do personal loan customers report problems with payment processing or servicing?",
        "product_category": "Personal Loan",
    },
    {
        "question": "What issues do customers raise about savings accounts?",
        "product_category": "Savings Account",
    },
    {
        "question": "Are savings account customers reporting unauthorized transactions or fraud?",
        "product_category": "Savings Account",
    },
    {
        "question": "What problems do customers report with money transfers?",
        "product_category": "Money Transfer",
    },
    {
        "question": "Compare common fraud-related complaints across credit cards and money transfers.",
        "product_category": None,
    },
    {
        "question": "What customer service or communication issues appear across all products?",
        "product_category": None,
    },
    {
        "question": "Which product category shows the strongest signs of repeated billing disputes?",
        "product_category": None,
    },
]


MOCK_SOURCES = [
    RetrievedChunk(
        text="customer reported unauthorized fees and incorrect interest charges on monthly credit card statement",
        metadata={"product_category": "Credit Card", "issue": "Fees or interest"},
        distance=0.08,
        id="mock_cc_0",
    ),
    RetrievedChunk(
        text="billing dispute on credit card not resolved after multiple calls to customer service",
        metadata={"product_category": "Credit Card", "issue": "Billing dispute"},
        distance=0.09,
        id="mock_cc_1",
    ),
    RetrievedChunk(
        text="personal loan payment was applied incorrectly causing late fee and credit score drop",
        metadata={"product_category": "Personal Loan", "issue": "Problem when making payments"},
        distance=0.10,
        id="mock_pl_0",
    ),
    RetrievedChunk(
        text="loan servicer failed to update payment plan after hardship request was approved",
        metadata={"product_category": "Personal Loan", "issue": "Struggling to pay your loan"},
        distance=0.12,
        id="mock_pl_1",
    ),
    RetrievedChunk(
        text="unauthorized withdrawal from savings account not reversed after fraud report",
        metadata={"product_category": "Savings Account", "issue": "Unauthorized transactions"},
        distance=0.11,
        id="mock_sa_0",
    ),
    RetrievedChunk(
        text="savings account closed without notice and funds held for weeks",
        metadata={"product_category": "Savings Account", "issue": "Closing an account"},
        distance=0.13,
        id="mock_sa_1",
    ),
    RetrievedChunk(
        text="money transfer failed and funds were not returned after three business days",
        metadata={"product_category": "Money Transfer", "issue": "Money was not available"},
        distance=0.11,
        id="mock_mt_0",
    ),
    RetrievedChunk(
        text="international remittance sent to wrong recipient with no refund offered",
        metadata={"product_category": "Money Transfer", "issue": "Transfer not received"},
        distance=0.14,
        id="mock_mt_1",
    ),
]


class MockRetriever(ComplaintRetriever):
    """Offline retriever for evaluation when the vector store or embeddings are unavailable."""

    def __init__(self) -> None:
        pass

    @property
    def chunk_count(self) -> int:
        return len(MOCK_SOURCES)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        product_category: str | None = None,
    ) -> list[RetrievedChunk]:
        chunks = MOCK_SOURCES
        if product_category:
            chunks = [c for c in chunks if c.metadata.get("product_category") == product_category]
        return chunks[:top_k]


def format_source_preview(sources, limit: int = 2) -> str:
    if not sources:
        return "_No sources retrieved_"

    previews: list[str] = []
    for source in sources[:limit]:
        meta = source.metadata
        snippet = source.text[:180].replace("\n", " ")
        if len(source.text) > 180:
            snippet += "..."
        previews.append(
            f"**{meta.get('product_category', 'Unknown')}** — "
            f"{meta.get('issue', 'N/A')}: {snippet}"
        )
    return "<br>".join(previews)


def build_markdown_report(responses, scores: dict[str, int] | None = None) -> str:
    scores = scores or {}
    lines = [
        "# RAG Evaluation Report",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "## Evaluation Table",
        "",
        "| Question | Generated Answer | Retrieved Sources (1–2) | Quality Score (1–5) | Comments/Analysis |",
        "| --- | --- | --- | ---: | --- |",
    ]

    for response in responses:
        question = response.question.replace("|", "\\|")
        answer = response.answer.replace("|", "\\|").replace("\n", " ")
        sources = format_source_preview(response.sources).replace("|", "\\|")
        score = scores.get(response.question, "")
        comment = scores.get(f"{response.question}__comment", "")
        lines.append(
            f"| {question} | {answer} | {sources} | {score} | {comment} |"
        )

    lines.extend(
        [
            "",
            "## Analysis Notes",
            "",
            "- **Strengths:** Add notes on retrieval relevance and answer grounding after review.",
            "- **Weaknesses:** Add notes on hallucinations, missing filters, or vague synthesis.",
            "- **Improvements:** Tune `top_k`, chunk overlap, prompt constraints, or LLM choice.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run qualitative RAG evaluation (Task 3)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/rag_evaluation.md"),
        help="Markdown report output path",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/rag_evaluation.json"),
        help="Raw JSON results output path",
    )
    parser.add_argument(
        "--vector-store-path",
        type=Path,
        default=VECTOR_STORE_PATH,
        help="Path to ChromaDB vector store",
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Use extractive fallback instead of loading the LLM",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock retrieval for offline evaluation report generation",
    )
    parser.add_argument("--top-k", type=int, default=5)
=======
from pathlib import Path

from src.config import PROJECT_ROOT
from src.generator import ComplaintGenerator
from src.rag import ComplaintRAG
from src.retriever import ComplaintRetriever

DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "rag_evaluation.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "reports" / "rag_evaluation.md"

EVALUATION_QUESTIONS = [
    "Why are customers unhappy with credit cards?",
    "What billing issues appear most often for personal loans?",
    "What fraud or unauthorized activity complaints mention savings accounts?",
    "What problems do customers report with money transfers?",
    "Which companies receive the most credit card complaints in the retrieved data?",
    "Are there recurring themes about customer service across products?",
    "What do complaints say about fees and charges?",
    "Do customers report difficulty closing accounts?",
]


def run_evaluation(
    max_chunks: int = 5_000,
    top_k: int = 5,
) -> list[dict]:
    retriever = ComplaintRetriever.from_prebuilt_store(max_chunks=max_chunks)
    rag = ComplaintRAG(
        retriever=retriever,
        generator=ComplaintGenerator(),
        top_k=top_k,
    )

    rows: list[dict] = []
    for question in EVALUATION_QUESTIONS:
        response = rag.ask(question)
        top_sources = response.sources[:2]
        rows.append(
            {
                "question": question,
                "generated_answer": response.answer,
                "retrieved_sources": [
                    {
                        "text": source.text[:300],
                        "product_category": source.metadata.get("product_category", ""),
                        "issue": source.metadata.get("issue", ""),
                        "score": round(source.score, 4),
                    }
                    for source in top_sources
                ],
                "quality_score": None,
                "comments": "Fill in after manual review.",
                "model_name": response.model_name,
            }
        )
    return rows


def write_reports(rows: list[dict], json_path: Path, md_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# RAG Evaluation",
        "",
        "| Question | Generated Answer | Retrieved Sources | Quality (1-5) | Comments |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        sources = row["retrieved_sources"]
        source_preview = "<br>".join(
            f"{s.get('product_category', 'N/A')}: {s['text'][:120]}..."
            for s in sources
        ) or "None"
        answer = row["generated_answer"].replace("|", "\\|").replace("\n", " ")
        question = row["question"].replace("|", "\\|")
        lines.append(
            f"| {question} | {answer[:180]}... | {source_preview} | {row['quality_score'] or 'TBD'} | {row['comments']} |"
        )

    md_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run qualitative RAG evaluation")
    parser.add_argument("--max-chunks", type=int, default=5_000)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--md-path", type=Path, default=DEFAULT_REPORT_MD)
>>>>>>> task-2/chunking-embeddings
    return parser.parse_args()


def main() -> None:
    args = parse_args()
<<<<<<< HEAD

    retriever = MockRetriever() if args.mock else ComplaintRetriever(persist_path=args.vector_store_path)
    pipeline = RAGPipeline(
        retriever=retriever,
        vector_store_path=args.vector_store_path,
        use_fallback=args.fallback or args.mock,
    )

    print(f"Vector store chunks available: {pipeline.retriever.chunk_count}")
    responses = pipeline.ask_batch(EVALUATION_QUESTIONS, top_k=args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_markdown_report(responses), encoding="utf-8")

    payload = [
        {
            "question": r.question,
            "product_category": r.product_category,
            "answer": r.answer,
            "sources": [
                {
                    "text": s.text,
                    "metadata": s.metadata,
                    "distance": s.distance,
                }
                for s in r.sources
            ],
        }
        for r in responses
    ]
    args.json_output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Report saved to {args.output}")
    print(f"Raw results saved to {args.json_output}")
=======
    rows = run_evaluation(max_chunks=args.max_chunks, top_k=args.top_k)
    write_reports(rows, args.json_path, args.md_path)
    print(f"Saved evaluation to {args.json_path} and {args.md_path}")
>>>>>>> task-2/chunking-embeddings


if __name__ == "__main__":
    main()
