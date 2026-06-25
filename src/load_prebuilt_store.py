<<<<<<< HEAD
"""Load the pre-built full-scale complaint embeddings into ChromaDB."""
=======
"""
Load the pre-built complaint embedding store (Task 3–4).

The full parquet (~1.37M chunks) is indexed into a persistent ChromaDB
collection on first use. Pass max_chunks to limit indexing for local dev/tests.
"""
>>>>>>> task-2/chunking-embeddings

from __future__ import annotations

import json
from pathlib import Path

import chromadb
import pandas as pd
from tqdm import tqdm

<<<<<<< HEAD
from src.build_vector_store import COLLECTION_NAME
from src.config import PREBUILT_PARQUET_PATH, VECTOR_STORE_PATH


def load_prebuilt_store(
    parquet_path: Path | str = PREBUILT_PARQUET_PATH,
    persist_path: Path | str = VECTOR_STORE_PATH,
    batch_size: int = 1000,
) -> dict:
    """
    Load the challenge pre-built parquet embeddings into a ChromaDB collection.

    Expected parquet columns (minimum):
    - text or chunk_text: chunk content
    - embedding: vector (list/array)
    - complaint_id, product_category, chunk_index, total_chunks
    """
    parquet_path = Path(parquet_path)
    persist_path = Path(persist_path)

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Pre-built parquet not found at {parquet_path}. "
            "Download complaint_embeddings.parquet from the challenge resources."
        )

    df = pd.read_parquet(parquet_path)
    text_col = "text" if "text" in df.columns else "chunk_text"
    if text_col not in df.columns:
        raise ValueError(f"Parquet must contain a text column; found: {list(df.columns)}")

    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0:
        return {
            "status": "skipped",
            "reason": "collection already populated",
            "chunk_count": collection.count(),
            "parquet_path": str(parquet_path),
            "vector_store_path": str(persist_path),
        }

    records = len(df)
    for start in tqdm(range(0, records, batch_size), desc="Loading pre-built index"):
        batch = df.iloc[start : start + batch_size]
        ids = []
        documents = batch[text_col].astype(str).tolist()
        embeddings = batch["embedding"].tolist()
        metadatas = []
        for offset, (_, row) in enumerate(batch.iterrows()):
            ids.append(
                f"{row.get('complaint_id', start + offset)}_{row.get('chunk_index', 0)}"
            )
            metadatas.append(
                {
                    "complaint_id": str(row.get("complaint_id", "")),
                    "product_category": str(row.get("product_category", "")),
                    "product": str(row.get("product", "")),
                    "issue": str(row.get("issue", "")),
                    "sub_issue": str(row.get("sub_issue", "")),
                    "company": str(row.get("company", "")),
                    "state": str(row.get("state", "")),
                    "date_received": str(row.get("date_received", "")),
                    "chunk_index": int(row.get("chunk_index", 0)),
                    "total_chunks": int(row.get("total_chunks", 1)),
                }
            )

        collection.add(
=======
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
>>>>>>> task-2/chunking-embeddings
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
<<<<<<< HEAD

    stats = {
        "status": "loaded",
        "chunk_count": collection.count(),
        "parquet_path": str(parquet_path),
        "vector_store_path": str(persist_path),
    }
    stats_path = persist_path / "prebuilt_load_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    stats = load_prebuilt_store()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
=======
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
>>>>>>> task-2/chunking-embeddings
