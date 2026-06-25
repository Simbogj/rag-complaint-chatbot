"""
Task 1: EDA helpers and complaint preprocessing.

Memory-safe chunked loading for the full CFPB dataset (~5.6 GB).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

TARGET_PRODUCTS = {
    "Credit Card",
    "Personal Loan",
    "Savings Account",
    "Money Transfer",
}

LOAD_COLUMNS = [
    "date_received",
    "product",
    "issue",
    "sub_issue",
    "consumer_complaint_narrative",
    "company",
    "state",
    "complaint_id",
]

OUTPUT_COLUMNS = [
    "complaint_id",
    "date_received",
    "product",
    "product_category",
    "issue",
    "sub_issue",
    "company",
    "state",
    "narrative",
    "clean_narrative",
    "word_count",
]

_BOILERPLATE = re.compile(
    r"(i am writing to (file|submit|report).*?complaint\s|"
    r"this is a complaint (about|regarding).*?\s|"
    r"i would like to (file|report|submit).*?\s)",
    re.IGNORECASE,
)
_SPECIAL_CHARS = re.compile(r"[^a-z0-9\s]")
_WHITESPACE = re.compile(r"\s+")


def normalize_columns(columns: pd.Index) -> pd.Index:
    return columns.str.strip().str.lower().str.replace(" ", "_")


def map_product(product: str | float | None) -> str | None:
    if pd.isna(product):
        return None

    p = str(product).lower()

    if "credit card" in p:
        return "Credit Card"
    if "personal loan" in p or "payday loan" in p or "title loan" in p:
        return "Personal Loan"
    if "savings" in p or "checking" in p:
        return "Savings Account"
    if "money transfer" in p or "virtual currency" in p or "money service" in p:
        return "Money Transfer"
    return None


def clean_narrative(text: object) -> str:
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = _BOILERPLATE.sub(" ", text)
    text = re.sub(r"\bx{2,}\b", " ", text)
    text = _SPECIAL_CHARS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def _prepare_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk.columns = normalize_columns(chunk.columns)

    available = [col for col in LOAD_COLUMNS if col in chunk.columns]
    if "product" not in available or "consumer_complaint_narrative" not in available:
        return pd.DataFrame()

    chunk = chunk[available].copy()
    narrative = chunk["consumer_complaint_narrative"]
    chunk = chunk[narrative.notna() & narrative.astype(str).str.strip().ne("")]
    if chunk.empty:
        return chunk

    chunk["product_category"] = chunk["product"].apply(map_product)
    chunk = chunk[chunk["product_category"].isin(TARGET_PRODUCTS)]
    return chunk


def load_dataset(path: str | Path, chunksize: int = 50_000) -> pd.DataFrame:
    """Load and filter the dataset in chunks to limit memory usage."""
    path = Path(path)
    chunks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(path, chunksize=chunksize, dtype=str, low_memory=True):
        prepared = _prepare_chunk(chunk)
        if not prepared.empty:
            chunks.append(prepared)

    if not chunks:
        return pd.DataFrame(columns=LOAD_COLUMNS + ["product_category"])

    return pd.concat(chunks, ignore_index=True)


def filter_products(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure only target products with non-empty narratives remain."""
    df = df.copy()
    narrative_col = "consumer_complaint_narrative"

    df = df[df[narrative_col].notna()]
    df = df[df[narrative_col].astype(str).str.strip().ne("")]
    df["product_category"] = df["product"].apply(map_product)
    df = df[df["product_category"].isin(TARGET_PRODUCTS)]
    return df.reset_index(drop=True)


def preprocess(df: pd.DataFrame, min_word_count: int = 3) -> pd.DataFrame:
    """Clean narratives and drop very short entries."""
    df = df.copy()
    df["narrative"] = df["consumer_complaint_narrative"].astype(str)
    df["clean_narrative"] = df["narrative"].apply(clean_narrative)
    df["word_count"] = df["clean_narrative"].str.split().str.len()
    df = df[df["word_count"] > min_word_count]

    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def run_pipeline(
    raw_path: str | Path = "data/raw/complaints.csv",
    filtered_path: str | Path = "data/filtered_complaints.csv",
    processed_path: str | Path = "data/processed/complaints_clean.csv",
    chunksize: int = 50_000,
) -> pd.DataFrame:
    """Load, filter, clean, and save the complaint dataset."""
    raw_path = Path(raw_path)
    filtered_path = Path(filtered_path)
    processed_path = Path(processed_path)

    df = load_dataset(raw_path, chunksize=chunksize)
    df = preprocess(df)

    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(filtered_path, index=False)
    df.to_csv(processed_path, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Task 1 EDA preprocessing pipeline")
    parser.add_argument("--raw-path", default="data/raw/complaints.csv")
    parser.add_argument("--filtered-path", default="data/filtered_complaints.csv")
    parser.add_argument("--processed-path", default="data/processed/complaints_clean.csv")
    parser.add_argument("--chunksize", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = run_pipeline(
        raw_path=args.raw_path,
        filtered_path=args.filtered_path,
        processed_path=args.processed_path,
        chunksize=args.chunksize,
    )
    print(f"Saved {len(df):,} complaints")
    print(df["product_category"].value_counts())
    print(f"Output: {args.filtered_path}")
    print(f"Output: {args.processed_path}")


if __name__ == "__main__":
    main()
