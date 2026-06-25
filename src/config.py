"""Shared configuration for the RAG complaint chatbot."""

from pathlib import Path

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"
COLLECTION_NAME = "complaint_chunks"

VECTOR_STORE_PATH = Path("vector_store/chromadb")
PREBUILT_PARQUET_PATH = Path("data/complaint_embeddings.parquet")

DEFAULT_TOP_K = 5
MAX_NEW_TOKENS = 512

TARGET_PRODUCTS = (
    "Credit Card",
    "Personal Loan",
    "Savings Account",
    "Money Transfer",
)
