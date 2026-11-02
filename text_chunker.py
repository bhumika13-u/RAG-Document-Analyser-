"""
text_chunker.py
───────────────
Splits page-level text dicts (produced by document_loader) into smaller,
overlapping chunks suitable for embedding.

Each output chunk is a dict:

    {
        "text":        str,   # chunk text
        "source":      str,   # original filename
        "page":        int,   # source page number
        "total_pages": int,   # total pages in the source document
        "chunk_index": int,   # 0-based index within the document
        "chunk_id":    str,   # "<source>::p<page>::c<chunk_index>"
    }

Uses LangChain's RecursiveCharacterTextSplitter which splits on paragraph
boundaries first, then sentences, then words — keeping chunks semantically
coherent.
"""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CHUNK_SIZE = 500       # characters
DEFAULT_CHUNK_OVERLAP = 100    # characters


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def chunk_pages(
    pages: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Split a list of page dicts into overlapping text chunks.

    Parameters
    ----------
    pages : list[dict]
        Output of ``document_loader.load_document`` / ``load_documents``.
    chunk_size : int
        Maximum number of characters per chunk.
    chunk_overlap : int
        Number of characters shared between adjacent chunks to preserve
        context across boundaries.

    Returns
    -------
    list[dict]
        Flat list of chunk dicts (see module docstring).
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Split on paragraphs → sentences → words → characters
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[dict[str, Any]] = []
    global_index = 0

    for page in pages:
        raw_text = page["text"]
        if not raw_text.strip():
            continue

        pieces = splitter.split_text(raw_text)
        for piece in pieces:
            if not piece.strip():
                continue
            chunk_id = f"{page['source']}::p{page['page']}::c{global_index}"
            all_chunks.append(
                {
                    "text": piece,
                    "source": page["source"],
                    "page": page["page"],
                    "total_pages": page.get("total_pages", 1),
                    "chunk_index": global_index,
                    "chunk_id": chunk_id,
                }
            )
            global_index += 1

    return all_chunks


def get_chunk_texts(chunks: list[dict[str, Any]]) -> list[str]:
    """Extract the plain text list from a list of chunk dicts."""
    return [c["text"] for c in chunks]
