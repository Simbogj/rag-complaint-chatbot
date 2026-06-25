"""Tests for Task 2 sampling and chunking utilities."""

import pandas as pd
import pytest

from src.chunking import chunk_text
from src.sampling import stratified_sample


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_category": ["Credit Card"] * 40
            + ["Personal Loan"] * 10
            + ["Savings Account"] * 30
            + ["Money Transfer"] * 20,
            "clean_narrative": ["sample complaint text"] * 100,
        }
    )


def test_stratified_sample_preserves_categories(sample_df: pd.DataFrame) -> None:
    result = stratified_sample(sample_df, n=50, random_state=42)
    assert len(result) == 50
    assert set(result["product_category"]) == set(sample_df["product_category"])


def test_stratified_sample_proportions(sample_df: pd.DataFrame) -> None:
    result = stratified_sample(sample_df, n=50, random_state=42)
    orig_pct = sample_df["product_category"].value_counts(normalize=True)
    sample_pct = result["product_category"].value_counts(normalize=True)
    for category in orig_pct.index:
        assert abs(orig_pct[category] - sample_pct[category]) < 0.05


def test_chunk_text_returns_non_empty_chunks() -> None:
    text = "word " * 200
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert chunks
    assert all(chunk.text for chunk in chunks)
    assert chunks[0].chunk_index == 0
    assert chunks[-1].total_chunks == len(chunks)


def test_chunk_text_respects_size_limit() -> None:
    text = "a" * 1200
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
    assert all(len(chunk.text) <= 500 for chunk in chunks)


def test_chunk_text_applies_overlap() -> None:
    text = "word " * 400
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) >= 2
    overlap_region = chunks[0].text[-40:].strip()
    assert overlap_region and overlap_region in chunks[1].text
