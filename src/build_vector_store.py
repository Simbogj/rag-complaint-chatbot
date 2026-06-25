"""
Task 2: Build a ChromaDB vector store from cleaned complaint narratives.

Usage:
    python -m src.build_vector_store
    python -m src.build_vector_store --sample-size 12000 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import chromadb
import pandas as pd
from tqdm import tqdm

from src.chunking import chunk_text
from src.sampling import sampling_summary, stratified_sample

DEFAULT_DATA_PATH = Path("data/processed/complaints_clean.csv")
DEFAULT_VECTOR_STORE_PATH = Path("vector_store/chromadb")
DEFAULT_SAMPLE_PATH = Path("data/processed/sampled_complaints.csv")
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SAMPLE_SIZE = 12_000
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_BATCH_SIZE = 64
COLLECTION_NAME = "complaint_chunks"


def load_complaints(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"clean_narrative", "product_category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def build_chunk_records(df: pd.DataFrame, chunk_size: int, chunk_overlap: int) -> list[dict]:
    records: list[dict] = []

    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        narrative = str(row["clean_narrative"])
        chunks = chunk_text(narrative, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            continue

        complaint_id = str(row.get("complaint_id", row_idx))
        for chunk in chunks:
            records.append(
                {
                    "id": f"{complaint_id}_{chunk.chunk_index}",
                    "text": chunk.text,
                    "metadata": {
                        "complaint_id": complaint_id,
                        "product_category": str(row["product_category"]),
                        "product": str(row.get("product", "")),
                        "issue": str(row.get("issue", "")),
                        "chunk_index": chunk.chunk_index,
                        "total_chunks": chunk.total_chunks,
                    },
                }
            )

    return records


def embed_and_index(
    records: list[dict],
    model_name: str,
    persist_path: Path,
    batch_size: int,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    if persist_path.exists():
        shutil.rmtree(persist_path)
    persist_path.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [r["text"] for r in records]
    ids = [r["id"] for r in records]
    metadatas = [r["metadata"] for r in records]

    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        end = start + batch_size
        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        batch_metas = metadatas[start:end]
        embeddings = model.encode(batch_texts, show_progress_bar=False).tolist()
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_metas,
        )

    stats = {
        "model_name": model_name,
        "collection_name": COLLECTION_NAME,
        "total_chunks": len(records),
        "embedding_dimensions": len(embeddings[0]) if embeddings else 0,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build complaint vector store (Task 2)")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--vector-store-path", type=Path, default=DEFAULT_VECTOR_STORE_PATH)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sample and chunk only; skip embedding/indexing (useful for validation)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading data from {args.data_path}...")
    df = load_complaints(args.data_path)
    print(f"Total complaints: {len(df):,}")

    print(f"Drawing stratified sample of {args.sample_size:,}...")
    sample_df = stratified_sample(
        df,
        n=args.sample_size,
        random_state=args.random_state,
    )
    args.sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(args.sample_path, index=False)
    print(f"Sample saved to {args.sample_path}")
    print(sampling_summary(sample_df))

    print("Building text chunks...")
    records = build_chunk_records(sample_df, args.chunk_size, args.chunk_overlap)
    print(f"Total chunks: {len(records):,}")

    if args.dry_run:
        manifest_path = args.vector_store_path / "chunk_manifest.json"
        args.vector_store_path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "sample_size": len(sample_df),
            "total_chunks": len(records),
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "model_name": args.model_name,
            "sample_path": str(args.sample_path),
            "dry_run": True,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Dry run complete. Manifest saved to {manifest_path}")
        return

    print(f"Embedding with {args.model_name} and indexing into ChromaDB...")
    stats = embed_and_index(
        records=records,
        model_name=args.model_name,
        persist_path=args.vector_store_path,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    stats.update(
        {
            "sample_size": len(sample_df),
            "sample_path": str(args.sample_path),
            "vector_store_path": str(args.vector_store_path),
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
        }
    )

    stats_path = args.vector_store_path / "build_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Vector store saved to {args.vector_store_path}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
