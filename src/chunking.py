"""
Task 2: Text chunking utilities for complaint narratives.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass(frozen=True)
class TextChunk:
    text: str
    chunk_index: int
    total_chunks: int


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[TextChunk]:
    """
    Split text into overlapping character chunks.

    Uses LangChain's RecursiveCharacterTextSplitter so chunk boundaries and
    overlap match the pre-built full-scale index spec (500 chars / 50 overlap).
    """
    if not text or not text.strip():
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=DEFAULT_SEPARATORS,
        is_separator_regex=False,
    )
    parts = splitter.split_text(text.strip())
    total = len(parts)
    return [
        TextChunk(text=part, chunk_index=i, total_chunks=total)
        for i, part in enumerate(parts)
    ]
