"""Smoke tests for the Streamlit app module."""

from unittest.mock import MagicMock, patch

import pytest


def test_app_imports() -> None:
    import app

    assert callable(app.main)


@patch("app.load_pipeline")
def test_source_to_dict(mock_load_pipeline: MagicMock) -> None:
    from app import source_to_dict
    from src.retriever import RetrievedChunk

    chunk = RetrievedChunk(
        text="billing dispute on credit card",
        metadata={"product_category": "Credit Card", "issue": "Billing dispute"},
        distance=0.1,
        id="1_0",
    )
    result = source_to_dict(chunk)
    assert result["text"] == chunk.text
    assert result["metadata"]["product_category"] == "Credit Card"
