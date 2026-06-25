"""
Task 2: Text chunking utilities for complaint narratives.
"""

from __future__ import annotations

from dataclasses import dataclass


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

    Mirrors LangChain's RecursiveCharacterTextSplitter priority:
    paragraphs, lines, sentences, then words.
    """
    if not text or not text.strip():
        return []

    separators = ["\n\n", "\n", ". ", " ", ""]
    raw_chunks = _split_recursive(text.strip(), separators, chunk_size, chunk_overlap)
    raw_chunks = [c.strip() for c in raw_chunks if c.strip()]

    total = len(raw_chunks)
    return [
        TextChunk(text=chunk, chunk_index=i, total_chunks=total)
        for i, chunk in enumerate(raw_chunks)
    ]


def _split_recursive(
    text: str,
    separators: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    final_chunks: list[str] = []
    separator = separators[-1]
    new_separators: list[str] = []

    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator = sep
            new_separators = separators[i + 1 :]
            break

    if separator:
        splits = text.split(separator) if separator else [text]
    else:
        splits = [text]

    current = ""
    for split in splits:
        piece = split if not separator else split + separator
        if len(piece) > chunk_size:
            if current:
                final_chunks.append(current)
                current = ""

            if new_separators:
                final_chunks.extend(
                    _split_recursive(piece, new_separators, chunk_size, chunk_overlap)
                )
            else:
                final_chunks.extend(_fixed_size_chunks(piece, chunk_size, chunk_overlap))
            continue

        candidate = current + piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                final_chunks.append(current)
            current = piece

    if current:
        final_chunks.append(current)

    return _merge_with_overlap(final_chunks, chunk_size, chunk_overlap)


def _fixed_size_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    step = max(chunk_size - chunk_overlap, 1)
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]


def _merge_with_overlap(chunks: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks

    merged = [chunks[0]]
    for chunk in chunks[1:]:
        prev = merged[-1]
        if len(prev) + len(chunk) - chunk_overlap <= chunk_size:
            overlap = prev[-chunk_overlap:] if len(prev) >= chunk_overlap else prev
            merged[-1] = prev + chunk[len(overlap) :]
        else:
            merged.append(chunk)
    return merged
