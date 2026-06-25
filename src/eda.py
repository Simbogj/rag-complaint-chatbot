"""
Task 1: Chunked EDA helpers for the full CFPB dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.preprocess import normalize_columns


def narrative_availability(path: str | Path, chunksize: int = 100_000) -> dict:
    """Count complaints with and without consumer narratives."""
    path = Path(path)
    total = 0
    with_narrative = 0

    for chunk in pd.read_csv(path, chunksize=chunksize, dtype=str, low_memory=True):
        chunk.columns = normalize_columns(chunk.columns)
        if "consumer_complaint_narrative" not in chunk.columns:
            continue

        text = chunk["consumer_complaint_narrative"].fillna("").astype(str)
        has_text = text.str.strip().ne("")
        total += len(chunk)
        with_narrative += int(has_text.sum())

    without_narrative = total - with_narrative
    return {
        "total": total,
        "with_narrative": with_narrative,
        "without_narrative": without_narrative,
        "with_narrative_pct": with_narrative / total if total else 0.0,
    }


def raw_product_distribution(path: str | Path, chunksize: int = 100_000) -> pd.Series:
    """Count complaints by raw product field before filtering."""
    path = Path(path)
    counts: dict[str, int] = {}

    for chunk in pd.read_csv(path, chunksize=chunksize, dtype=str, low_memory=True):
        chunk.columns = normalize_columns(chunk.columns)
        if "product" not in chunk.columns:
            continue

        for product, count in chunk["product"].value_counts().items():
            counts[product] = counts.get(product, 0) + int(count)

    return pd.Series(counts).sort_values(ascending=False)


def word_count_summary(df: pd.DataFrame) -> dict:
    """Summarize cleaned narrative word counts."""
    stats = df["word_count"].describe()
    return {
        "mean": float(stats["mean"]),
        "median": float(df["word_count"].median()),
        "min": int(stats["min"]),
        "max": int(stats["max"]),
        "q25": float(stats["25%"]),
        "q75": float(stats["75%"]),
    }
