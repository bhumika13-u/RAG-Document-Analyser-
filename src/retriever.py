"""
retriever.py
────────────
High-level retrieval interface:
  embed a user query → search the VectorStore → return top-K chunks.

Keeps embeddings.py and vector_store.py as independent components;
this module is the thin glue between them.

Retrieval Quality Tuning Guide
───────────────────────────────
The quality of retrieved passages depends on four interacting parameters:

1. **chunk_size** (default 500 chars)
   - Smaller (200-300): higher precision, less context per chunk.
     Good for fact lookups ("What year did X happen?").
   - Larger (600-1000): more context per chunk, better for complex questions.
     Good for conceptual questions ("Explain the process of …").
   - Recommended range for student laptops: 300–600.

2. **chunk_overlap** (default 100 chars, ~20% of chunk_size)
   - Prevents facts being split across two chunks.
   - A good rule of thumb: overlap ≈ 15–25% of chunk_size.
   - Too high overlap wastes index space; too low causes missed facts.

3. **top_k** (default 5)
   - How many chunks to retrieve per query.
   - Increase to 7–10 if answers are incomplete or questions are multi-part.
   - Decrease to 3 for faster, more focused answers.
   - More chunks = more tokens sent to the LLM (higher cost/latency).

4. **min_score** (default 0.0 in UI, 0.25 in pipeline threshold)
   - Cosine similarity threshold below which chunks are flagged as
     "low evidence".  Range: 0.0 (off) to 1.0 (only perfect matches).
   - For general questions: 0.25–0.35 is a good starting point.
   - For strict factual Q&A: 0.4–0.5 to avoid hallucination risk.
   - Setting it too high filters out everything; 0.0 surfaces everything.

Embedding model: ``sentence-transformers/all-MiniLM-L6-v2``
   - 384 dimensions, ~22 MB, fast on CPU (~5 ms/sentence).
   - Alternative for higher quality (slower): ``all-mpnet-base-v2`` (768-dim).
   - Set EMBEDDING_MODEL env var to switch models.  Re-index after changing.
"""

from __future__ import annotations

from typing import Any, Optional

from src.embeddings import embed_query, DEFAULT_EMBEDDING_MODEL
from src.vector_store import VectorStore


def retrieve(
    query: str,
    store: VectorStore,
    top_k: int = 5,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Retrieve the most relevant chunks for *query* from *store*.

    Parameters
    ----------
    query : str
        User's natural-language question.
    store : VectorStore
        A populated VectorStore instance.
    top_k : int
        Maximum number of chunks to return.
    model_name : str
        Embedding model used to encode the query (must match the model
        used during indexing).
    min_score : float
        Minimum cosine similarity threshold (0–1).  Chunks below this
        score are dropped.  Default 0.0 keeps everything.

    Returns
    -------
    list[dict]
        Chunk dicts sorted by descending similarity score, each containing:
        ``text``, ``source``, ``page``, ``chunk_index``, ``chunk_id``,
        ``score``.
    """
    if not query.strip():
        raise ValueError("Query must not be empty.")
    if store.total_chunks == 0:
        raise ValueError(
            "The vector store is empty. Please upload and index at least one document first."
        )

    q_emb = embed_query(query, model_name=model_name)
    results = store.search(q_emb, top_k=top_k)

    if min_score > 0.0:
        results = [r for r in results if r["score"] >= min_score]

    return results


def format_retrieved_chunks(
    chunks: list[dict[str, Any]],
    max_chars_per_chunk: Optional[int] = None,
) -> str:
    """
    Format retrieved chunks into a single context string for the LLM prompt.

    Each chunk is prefixed with its source and page number so the LLM can
    attribute claims.

    Parameters
    ----------
    chunks : list[dict]
        Output of :func:`retrieve`.
    max_chars_per_chunk : int | None
        Truncate each chunk's text to this many characters.  ``None`` keeps
        the full text.

    Returns
    -------
    str
        Multi-section context block.
    """
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk["text"]
        if max_chars_per_chunk:
            text = text[:max_chars_per_chunk]
        header = f"[{i}] Source: {chunk['source']} | Page: {chunk['page']} | Score: {chunk['score']:.3f}"
        parts.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(parts)
