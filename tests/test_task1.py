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
