"""Tests for Task 3 RAG pipeline components."""

from unittest.mock import MagicMock, patch

import pytest

from src.generator import ComplaintGenerator
from src.prompts import build_rag_prompt, format_context
from src.rag import RAGPipeline


@pytest.fixture
def sample_chunks() -> list[dict]:
    return [
        {
            "text": "customer charged unauthorized fees on credit card statement",
            "metadata": {
                "product_category": "Credit Card",
                "issue": "Fees or interest",
            },
        },
        {
            "text": "billing dispute not resolved after multiple calls to support",
            "metadata": {
                "product_category": "Credit Card",
                "issue": "Billing dispute",
            },
        },
    ]


def test_format_context_includes_metadata(sample_chunks: list[dict]) -> None:
    context = format_context(sample_chunks)
    assert "Credit Card" in context
    assert "Billing dispute" in context
    assert "Source 1" in context
    assert "Source 2" in context


def test_build_rag_prompt_includes_question_and_context(sample_chunks: list[dict]) -> None:
    context = format_context(sample_chunks)
    prompt = build_rag_prompt(context, "Why are credit card customers unhappy?")
    assert "CrediTrust" in prompt
    assert "Why are credit card customers unhappy?" in prompt
    assert "billing dispute" in prompt.lower()


def test_generator_fallback_when_no_context() -> None:
    generator = ComplaintGenerator(use_fallback=True)
    answer = generator.generate(prompt="", context="", question="Why are fees high?")
    assert "don't have enough information" in answer


def test_generator_fallback_summarizes_context(sample_chunks: list[dict]) -> None:
    generator = ComplaintGenerator(use_fallback=True)
    context = format_context(sample_chunks)
    answer = generator.generate(
        prompt="ignored",
        context=context,
        question="What billing issues appear?",
    )
    assert "retrieved complaint excerpts" in answer.lower()
    assert "billing" in answer.lower()


@patch("src.rag.ComplaintRetriever")
def test_rag_pipeline_ask_returns_sources(mock_retriever_cls: MagicMock) -> None:
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [
        MagicMock(
            text="late fee charged incorrectly",
            metadata={"product_category": "Credit Card", "issue": "Fees"},
            distance=0.12,
            id="1_0",
        )
    ]
    mock_retriever_cls.return_value = mock_retriever

    pipeline = RAGPipeline(use_fallback=True)
    response = pipeline.ask(
        "Why are credit card customers unhappy?",
        product_category="Credit Card",
        top_k=3,
    )

    mock_retriever.retrieve.assert_called_once_with(
        query="Why are credit card customers unhappy?",
        top_k=3,
        product_category="Credit Card",
    )
    assert response.answer
    assert len(response.sources) == 1
    assert "Credit Card" in response.prompt
