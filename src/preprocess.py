"""
Task 1: EDA helpers and complaint text preprocessing (memory-safe version)
"""

import re
import pandas as pd


# -----------------------------
# Target product mapping
# -----------------------------
TARGET_PRODUCTS = {
        "credit card": "Credit Card",
    "credit card or prepaid card": "Credit Card",
    "personal loans": "Personal Loan",
    "payday loan, title loan, or personal loan": "Personal Loan",
    "checking or savings account": "Savings Account",
    "money transfers": "Money Transfer",
    "money transfer, virtual currency, or money service": "Money Transfer",
}


# -----------------------------
# Regex patterns
# -----------------------------
_BOILERPLATE = re.compile(
    r"(i am writing to (file|submit|report) a complaint.*?[.!]"
    r"|this is a complaint (about|regarding).*?[.!]"
    r"|i would like to (file|report|submit).*?[.!])",
    re.IGNORECASE,
)

_SPECIAL_CHARS = re.compile(r"[^a-z0-9\s.,!?;:()\-']")
_WHITESPACE = re.compile(r"\s+")


# -----------------------------
# Load dataset (memory-safe)
# -----------------------------
def load_dataset(path: str, chunksize: int = 10000) -> pd.DataFrame:
    """
    Load dataset in chunks safely without crashing memory.
    Only keeps required columns early.
    """

    required_cols = {"product", "consumer_complaint_narrative"}

    processed_chunks = []

    for chunk in pd.read_csv(
        path,
        chunksize=chunksize,
        dtype=str,
        low_memory=True,
    ):
        # normalize column names
        chunk.columns = (
            chunk.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        # keep only needed columns if present
        available_cols = [c for c in required_cols if c in chunk.columns]

        if len(available_cols) < 2:
            continue

        chunk = chunk[available_cols]

        processed_chunks.append(chunk)

    return pd.concat(processed_chunks, ignore_index=True)


# -----------------------------
# Find column helper
# -----------------------------
def _find_col(df: pd.DataFrame, candidates: list) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Missing columns: {candidates}")


# -----------------------------
# Filter products
# -----------------------------
def filter_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only target product categories and valid narratives.
    """

    df = df.copy()

    product_col = _find_col(df, ["product"])
    narrative_col = _find_col(df, ["consumer_complaint_narrative", "narrative"])

    # normalize product mapping
    df["product_category"] = (
        df[product_col]
        .str.lower()
        .str.strip()
        .map(TARGET_PRODUCTS)
    )

    # filter valid categories
    df = df[df["product_category"].notna()]

    # remove empty narratives
    df = df[df[narrative_col].notna()]
    df = df[df[narrative_col].str.strip().ne("")]


    df = df.rename(columns={narrative_col: "narrative"})

    return df.reset_index(drop=True)


# -----------------------------
# Clean text
# -----------------------------
def clean_narrative(text) -> str:
    """
    Basic NLP cleaning for complaint text.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # remove boilerplate phrases
    text = _BOILERPLATE.sub("", text)

    # remove repeated x placeholders
    text = re.sub(r"\bx{2,}\b", "", text, flags=re.IGNORECASE)

    # remove special characters
    text = _SPECIAL_CHARS.sub(" ", text)

    # normalize whitespace
    text = _WHITESPACE.sub(" ", text)

    return text.strip()


# -----------------------------
# Preprocess dataset
# -----------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean narratives and generate NLP features.
    """

    df = df.copy()

    # clean text
    df["clean_narrative"] = df["narrative"].apply(clean_narrative)

    # remove very short texts
    df = df[df["clean_narrative"].str.len() > 20]

    # word count feature
    df["word_count"] = df["clean_narrative"].str.split().str.len()

    return df.reset_index(drop=True)