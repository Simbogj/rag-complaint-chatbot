"""
Task 2: Stratified sampling for embedding pipeline.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    category_col: str = "product_category",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Draw a stratified sample that preserves product-category proportions.

    Uses sklearn's stratified split so each category keeps roughly the same
    share as in the full dataset.
    """
    if n >= len(df):
        return df.reset_index(drop=True)

    if category_col not in df.columns:
        raise ValueError(f"Column '{category_col}' not found in dataframe")

    categories = df[category_col]
    if categories.nunique() < 2:
        return df.sample(n=n, random_state=random_state).reset_index(drop=True)

    sample, _ = train_test_split(
        df,
        train_size=n,
        stratify=categories,
        random_state=random_state,
    )
    return sample.reset_index(drop=True)


def sampling_summary(df: pd.DataFrame, category_col: str = "product_category") -> pd.Series:
    """Return counts and percentages per product category."""
    counts = df[category_col].value_counts().sort_index()
    pct = (counts / len(df) * 100).round(2)
    return pd.DataFrame({"count": counts, "pct": pct})
