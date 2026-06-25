import json
from pathlib import Path
from typing import List, Dict

from src.rag_core import RAGPipeline

# ---------------------------------------------------------------------------
# Evaluation configuration
# ---------------------------------------------------------------------------
EVAL_QUESTIONS: List[str] = [
    "Why are customers unhappy with credit cards?",
    "What are the most common issues with personal loans?",
    "What complaints do users have about savings accounts?",
    "What problems are reported for money transfers?",
    "Which product has the highest number of fraud-related complaints?",
]

DEFAULT_TOP_K = 5
DEFAULT_CATEGORY = None  # No filter – retrieve across all products

OUTPUT_DIR = Path(__file__).parent.parent / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_evaluation(pipeline: RAGPipeline) -> List[Dict[str, any]]:
    """Run the RAG pipeline on each evaluation question.

    Returns a list of dictionaries containing the question, answer, and a concise
    representation of the retrieved sources.
    """
    results = []
    for q in EVAL_QUESTIONS:
        response = pipeline.ask(q, top_k=DEFAULT_TOP_K, product_category=DEFAULT_CATEGORY)
        answer = response["answer"]
        sources = response["sources"]
        # Keep only a short snippet of each source for the markdown table
        source_snippets = [
            {
                "metadata": src["metadata"],
                "snippet": src["text"][:200].replace("\n", " ") + "...",
                "distance": round(src["distance"], 3),
            }
            for src in sources
        ]
        results.append({"question": q, "answer": answer, "sources": source_snippets})
    return results

def write_markdown(results: List[Dict[str, any]]) -> None:
    md_path = OUTPUT_DIR / "rag_evaluation.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# RAG Evaluation Report\n\n")
        f.write("| Question | Answer | Sources (snippet) | Quality (1‑5) | Comments |\n")
        f.write("|---|---|---|---|---|\n")
        for item in results:
            q = item["question"].replace("|", "\\|")
            a = item["answer"].replace("|", "\\|")
            srcs = " ".join([f"* {s['snippet']}" for s in item["sources"]])
            srcs = srcs.replace("|", "\\|")
            # Placeholder columns for manual scoring
            f.write(f"| {q} | {a} | {srcs} |   |   |\n")
    print(f"Markdown evaluation written to {md_path}")

def write_json(results: List[Dict[str, any]]) -> None:
    json_path = OUTPUT_DIR / "rag_evaluation.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"JSON evaluation written to {json_path}")

if __name__ == "__main__":
    rag = RAGPipeline()
    eval_results = run_evaluation(rag)
    write_markdown(eval_results)
    write_json(eval_results)
