"""Shared configuration for the RAG complaint chatbot."""

from pathlib import Path

<<<<<<< HEAD
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
=======
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Task 2 — sample vector store
TASK2_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "complaints_clean.csv"
TASK2_SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "sampled_complaints.csv"
TASK2_VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store" / "chromadb"
TASK2_COLLECTION = "complaint_chunks"

# Task 3–4 — pre-built full-scale store
PREBUILT_PARQUET_PATH = PROJECT_ROOT / "data" / "raw" / "complaint_embeddings.parquet"
PREBUILT_VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store" / "prebuilt_chromadb"
PREBUILT_COLLECTION = "complaint_embeddings"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_SAMPLE_SIZE = 12_000
>>>>>>> task-2/chunking-embeddings
