"""Load the pre-built full-scale complaint embeddings into ChromaDB."""

from __future__ import annotations

import json
from pathlib import Path

import chromadb
import pandas as pd
from tqdm import tqdm

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
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

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
