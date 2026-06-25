"""
Task 1: EDA helpers and complaint preprocessing (clean + memory-safe version)
"""

import re
import pandas as pd


# =========================================================
# 1. Target product mapping
# =========================================================
def map_product(product: str):
    if pd.isna(product):
        return None

    p = product.lower()

    if "credit card" in p:
        return "Credit Card"

    elif "personal loan" in p or "payday loan" in p or "title loan" in p:
        return "Personal Loan"

    elif "savings" in p or "checking" in p:
        return "Savings Account"

    elif "money transfer" in p or "virtual currency" in p or "money service" in p:
        return "Money Transfer"

    return None


TARGET_PRODUCTS = {
    "Credit Card",
    "Personal Loan",
    "Savings Account",
    "Money Transfer"
}


# =========================================================
# 2. Cleaning patterns
# =========================================================
_BOILERPLATE = re.compile(
    r"(i am writing to (file|submit|report).*?complaint\s|"
    r"this is a complaint (about|regarding).*?\s|"
    r"i would like to (file|report|submit).*?\s)",
    re.IGNORECASE,
)

_SPECIAL_CHARS = re.compile(r"[^a-z0-9\s]")
_WHITESPACE = re.compile(r"\s+")


# =========================================================
# 3. Memory-safe loader
# =========================================================
def load_dataset(path: str, chunksize: int = 10000) -> pd.DataFrame:

    required_cols = ["product", "consumer_complaint_narrative"]
    chunks = []

    for chunk in pd.read_csv(path, chunksize=chunksize, dtype=str, low_memory=True):

        chunk.columns = (
            chunk.columns.str.strip().str.lower().str.replace(" ", "_")
        )

        if not all(col in chunk.columns for col in required_cols):
            continue

        chunks.append(chunk[required_cols])

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


# =========================================================
# 4. Filter products (FIXED ORDER + PRESERVE RAW)
# =========================================================
def filter_products(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    raw_col = "consumer_complaint_narrative"
    product_col = "product"

    # ✔ keep raw text column (IMPORTANT for EDA correctness)
    df = df[df[raw_col].notna()]
    df = df[df[raw_col].astype(str).str.strip().ne("")]

#  map product
    df["product_category"] = df[product_col].apply(map_product)

    # keep only targets
    df = df[df["product_category"].isin(TARGET_PRODUCTS)]

    return df.reset_index(drop=True)


# =========================================================
# 5. Clean text
# =========================================================
def clean_narrative(text) -> str:

    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = _BOILERPLATE.sub(" ", text)
    text = re.sub(r"\bx{2,}\b", " ", text)

    text = _SPECIAL_CHARS.sub(" ", text)
    text = _WHITESPACE.sub(" ", text)

    return text.strip()


# =========================================================
# 6. Preprocessing pipeline (CLEAN + CONSISTENT)
# =========================================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # IMPORTANT: keep raw column
    df["clean_narrative"] = df["consumer_complaint_narrative"].apply(clean_narrative)

    # feature engineering
    df["word_count"] = df["clean_narrative"].str.split().str.len()

    # remove very short/noisy text
    df = df[df["word_count"] > 3]

    return df.reset_index(drop=True)