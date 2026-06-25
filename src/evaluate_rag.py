"""
Task 3: Qualitative RAG evaluation runner.

Usage:
    python -m src.evaluate_rag
    python -m src.evaluate_rag --max-chunks 5000
"""

from __future__ import annotations

import argparse
import json
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_evaluation(max_chunks=args.max_chunks, top_k=args.top_k)
    write_reports(rows, args.json_path, args.md_path)
    print(f"Saved evaluation to {args.json_path} and {args.md_path}")


if __name__ == "__main__":
    main()
