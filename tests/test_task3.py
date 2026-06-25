"""Tests for Task 3 RAG pipeline components."""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from src.generator import ComplaintGenerator, _fallback_answer
from src.prompts import build_rag_prompt
from src.rag import ComplaintRAG
from src.retriever import ComplaintRetriever, RetrievedChunk


def test_build_rag_prompt_includes_question_and_context() -> None:
    prompt = build_rag_prompt(
        "Why are credit card customers unhappy?",
        ["billing dispute on monthly statement"],
    )
    assert "Why are credit card customers unhappy?" in prompt
    assert "billing dispute on monthly statement" in prompt
    assert "Answer:" in prompt


def test_fallback_answer_uses_context_excerpts() -> None:
    prompt = build_rag_prompt(
        "What billing issues appear?",
        ["customer charged twice for the same transaction"],
    )
    answer = _fallback_answer(prompt)
    assert "billing issues" in answer.lower() or "billing" in answer.lower()
    assert "charged twice" in answer


@patch("src.generator.ComplaintGenerator._load_pipeline", side_effect=RuntimeError("no model"))
def test_generator_falls_back_when_model_unavailable(_mock_pipeline) -> None:
    generator = ComplaintGenerator()
    prompt = build_rag_prompt("Test question?", ["sample complaint text"])
    result = generator.generate(prompt)
    assert result.model_name == "fallback-extractive"
    assert result.answer


def test_rag_pipeline_combines_retriever_and_generator() -> None:
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        RetrievedChunk(
            text="unauthorized charge on credit card",
            score=0.91,
            metadata={"product_category": "Credit Card", "issue": "Billing dispute"},
        )
    ]

    mock_generator = MagicMock()
    mock_generator.generate.return_value = MagicMock(
        answer="Customers report unauthorized charges.",
        model_name="test-model",
    )

    rag = ComplaintRAG(retriever=mock_retriever, generator=mock_generator, top_k=3)
    response = rag.ask("Why are people unhappy with credit cards?")

    assert response.answer == "Customers report unauthorized charges."
    assert len(response.sources) == 1
    mock_retriever.retrieve.assert_called_once()
    mock_generator.generate.assert_called_once()


def test_retriever_returns_ranked_chunks() -> None:
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["first chunk", "second chunk"]],
        "metadatas": [[{"product_category": "Credit Card"}, {"product_category": "Credit Card"}]],
        "distances": [[0.1, 0.3]],
    }

    retriever = ComplaintRetriever(collection=mock_collection)
    mock_embedding = MagicMock()
    mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
    with patch.object(ComplaintRetriever, "model", new_callable=PropertyMock) as mock_model:
        mock_model.return_value.encode.return_value = mock_embedding
        results = retriever.retrieve("credit card fees", top_k=2)

    assert len(results) == 2
    assert results[0].score > results[1].score
    assert results[0].text == "first chunk"
