"""
Load the pre-built complaint embedding store (Task 3–4).

The full parquet (~1.37M chunks) is indexed into a persistent ChromaDB
collection on first use. Pass max_chunks to limit indexing for local dev/tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
import pandas as pd
from tqdm import tqdm

from src.config import (
    EMBEDDING_MODEL,
    PREBUILT_COLLECTION,
    PREBUILT_PARQUET_PATH,
    PREBUILT_VECTOR_STORE_PATH,
)


def _manifest_path(persist_path: Path) -> Path:
    return persist_path / "index_manifest.json"


def is_index_ready(persist_path: Path = PREBUILT_VECTOR_STORE_PATH) -> bool:
    manifest = _manifest_path(persist_path)
    return manifest.exists() and persist_path.exists()


def build_prebuilt_index(
    parquet_path: Path = PREBUILT_PARQUET_PATH,
    persist_path: Path = PREBUILT_VECTOR_STORE_PATH,
    batch_size: int = 512,
    max_chunks: int | None = None,
) -> dict:
    """Index pre-built parquet embeddings into ChromaDB."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Pre-built parquet not found: {parquet_path}")

    persist_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_or_create_collection(
        name=PREBUILT_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    parquet_file = pd.read_parquet(parquet_path)
    if max_chunks is not None:
        parquet_file = parquet_file.head(max_chunks)

    total = len(parquet_file)
    indexed = 0

    for start in tqdm(range(0, total, batch_size), desc="Indexing pre-built store"):
        batch = parquet_file.iloc[start : start + batch_size]
        ids = batch["id"].astype(str).tolist()
        documents = batch["document"].astype(str).tolist()
        embeddings = batch["embedding"].tolist()
        metadatas = [_normalize_metadata(row["metadata"]) for _, row in batch.iterrows()]

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        indexed += len(ids)

    stats = {
        "model_name": EMBEDDING_MODEL,
        "collection_name": PREBUILT_COLLECTION,
        "total_chunks": indexed,
        "parquet_path": str(parquet_path),
        "vector_store_path": str(persist_path),
        "max_chunks": max_chunks,
    }
    _manifest_path(persist_path).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def get_prebuilt_collection(
    persist_path: Path = PREBUILT_VECTOR_STORE_PATH,
    max_chunks: int | None = None,
    rebuild: bool = False,
):
    """Return a ChromaDB collection backed by the pre-built parquet index."""
    if rebuild or not is_index_ready(persist_path):
        build_prebuilt_index(
            persist_path=persist_path,
            max_chunks=max_chunks,
        )

    client = chromadb.PersistentClient(path=str(persist_path))
    return client.get_collection(name=PREBUILT_COLLECTION)


def _normalize_metadata(metadata: object) -> dict[str, str | int | float | bool]:
    if not isinstance(metadata, dict):
        return {}

    normalized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            normalized[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized
