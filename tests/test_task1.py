<<<<<<< HEAD
"""Tests for Task 1 preprocessing and EDA helpers."""

import pandas as pd

from src.eda import word_count_summary
from src.preprocess import clean_narrative, filter_products, map_product, preprocess


def test_map_product_categories() -> None:
    assert map_product("Credit card or other retail") == "Credit Card"
    assert map_product("Payday loan") == "Personal Loan"
    assert map_product("Checking or savings account") == "Savings Account"
    assert map_product("Money transfer, virtual currency") == "Money Transfer"
    assert map_product("Debt collection") is None


def test_clean_narrative_removes_boilerplate_and_special_chars() -> None:
    text = "I am writing to file a complaint about BILLING!!! XXXX issue."
    cleaned = clean_narrative(text)
    assert "complaint about" not in cleaned
    assert "billing" in cleaned
    assert "!" not in cleaned


def test_filter_products_keeps_target_categories_only() -> None:
    df = pd.DataFrame(
        {
            "product": ["Credit card", "Debt collection", "Personal loan"],
            "consumer_complaint_narrative": [
                "billing issue on my card",
                "invalid debt claim",
                "loan payment problem",
            ],
        }
    )
    filtered = filter_products(df)
    assert len(filtered) == 2


def test_preprocess_drops_short_narratives() -> None:
    df = pd.DataFrame(
        {
            "product": ["Credit card", "Credit card"],
            "consumer_complaint_narrative": [
                "long enough complaint about unauthorized fees on my account",
                "too short",
            ],
            "product_category": ["Credit Card", "Credit Card"],
        }
    )
    result = preprocess(df)
    assert len(result) == 1
    assert "clean_narrative" in result.columns


def test_word_count_summary() -> None:
    df = pd.DataFrame({"word_count": [10, 20, 30, 40, 50]})
    summary = word_count_summary(df)
    assert summary["median"] == 30
=======
"""Tests for Task 1 preprocessing helpers."""

import pandas as pd

from src.preprocess import clean_narrative, filter_products, map_product


def test_map_product_credit_card() -> None:
    assert map_product("Credit card or other revolving credit") == "Credit Card"


def test_clean_narrative_lowercases_and_strips_noise() -> None:
    text = "I am writing to file a complaint about BILLING!!!"
    cleaned = clean_narrative(text)
    assert cleaned == cleaned.lower()
    assert "billing" in cleaned


def test_filter_products_keeps_target_categories() -> None:
    df = pd.DataFrame(
        {
            "product": ["Credit card", "Debt collection", "Money transfer"],
            "consumer_complaint_narrative": ["issue one", "issue two", "issue three"],
        }
    )
    filtered = filter_products(df)
    assert set(filtered["product_category"]) == {"Credit Card", "Money Transfer"}
>>>>>>> task-2/chunking-embeddings
