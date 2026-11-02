"""
summarizer.py
─────────────
Document summarisation and key-point extraction.

Strategy
────────
Rather than running a second LLM call (which may not be available), we
use a purely extractive approach that works with zero dependencies beyond
what's already required by the project:

1. **Summarisation** — TF-IDF sentence scoring.
   Every sentence in the sampled chunks is scored by the sum of TF-IDF
   weights of its words.  The top N sentences (by score, in document order)
   form the extractive summary.

2. **Key-point extraction** — term-frequency heuristic.
   Short, informative noun-phrase-like fragments are extracted from
   high-scoring sentences to form a bullet-point list.

If an LLM is configured (OPENAI_API_KEY or GGUF_MODEL_PATH), the
``summarize_with_llm`` helper can be used instead for higher quality.
"""

from __future__ import annotations

import re
import math
from collections import Counter
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "as", "is", "it", "its", "that",
    "this", "be", "are", "was", "were", "been", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "i", "we", "you", "he", "she", "they", "them", "their", "our", "your",
    "not", "no", "so", "if", "then", "than", "which", "who", "what", "when",
    "where", "how", "all", "also", "into", "about", "after", "before", "can",
    "each", "more", "such", "there", "these", "those", "through", "used",
}

def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[a-zA-Z]{3,}", text) if w.lower() not in _STOP_WORDS]

def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]

def _tfidf_scores(all_sentences: list[str]) -> dict[str, float]:
    """Return a mapping of sentence → TF-IDF score."""
    n = len(all_sentences)
    if n == 0:
        return {}

    # Term frequency per sentence
    tf: list[Counter] = [Counter(_tokenize(s)) for s in all_sentences]
    # Document frequency
    df: Counter = Counter()
    for counter in tf:
        df.update(counter.keys())

    scores: dict[str, float] = {}
    for i, sentence in enumerate(all_sentences):
        score = 0.0
        tokens = _tokenize(sentence)
        if not tokens:
            scores[sentence] = 0.0
            continue
        for token in tokens:
            tfidf = (tf[i][token] / len(tokens)) * math.log((n + 1) / (df[token] + 1))
            score += tfidf
        scores[sentence] = score / len(tokens)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def summarize_chunks(
    chunks: list[dict[str, Any]],
    num_sentences: int = 8,
) -> str:
    """
    Produce an extractive summary of the given chunks.

    Parameters
    ----------
    chunks : list[dict]
        Chunk dicts (must have ``"text"`` key).
    num_sentences : int
        Approximate number of sentences in the summary.

    Returns
    -------
    str
        The extractive summary as a paragraph.
    """
    combined_text = " ".join(c["text"] for c in chunks)
    sentences = _split_sentences(combined_text)

    if not sentences:
        return "No summary could be generated — the document appears to contain no readable sentences."

    scores = _tfidf_scores(sentences)
    ranked = sorted(sentences, key=lambda s: scores.get(s, 0.0), reverse=True)
    top = ranked[:num_sentences]

    # Restore original document order
    order_map = {s: i for i, s in enumerate(sentences)}
    top_sorted = sorted(top, key=lambda s: order_map.get(s, 0))

    return " ".join(top_sorted)


def extract_key_points(
    chunks: list[dict[str, Any]],
    num_points: int = 7,
) -> list[str]:
    """
    Extract key points (short informative phrases) from the given chunks.

    Parameters
    ----------
    chunks : list[dict]
        Chunk dicts (must have ``"text"`` key).
    num_points : int
        Number of key points to return.

    Returns
    -------
    list[str]
        A list of short key-point strings.
    """
    combined_text = " ".join(c["text"] for c in chunks)
    sentences = _split_sentences(combined_text)

    if not sentences:
        return ["No key points could be extracted."]

    scores = _tfidf_scores(sentences)
    ranked = sorted(sentences, key=lambda s: scores.get(s, 0.0), reverse=True)

    # Trim long sentences into shorter point-like fragments
    key_points: list[str] = []
    for sentence in ranked:
        if len(key_points) >= num_points:
            break
        # Strip to first 120 characters and close with ellipsis if truncated
        point = sentence.strip()
        if len(point) > 120:
            # Try to cut at a word boundary
            cut = point[:120].rsplit(" ", 1)[0]
            point = cut + "…"
        if point and point not in key_points:
            key_points.append(point)

    return key_points


def summarize_with_llm(
    chunks: list[dict[str, Any]],
    question_hint: str = "Summarise the main content and key findings of this document.",
) -> str:
    """
    Higher-quality summarisation using the configured LLM.
    Falls back to extractive summarisation if no LLM is available.

    Parameters
    ----------
    chunks : list[dict]
        Chunk dicts to summarise.
    question_hint : str
        Instruction sent to the LLM.

    Returns
    -------
    str
        LLM-generated or extractive summary.
    """
    from src.llm_service import generate_answer, detect_provider  # noqa: PLC0415

    provider = detect_provider()
    if provider == "extractive":
        return summarize_chunks(chunks)

    context = "\n\n".join(c["text"] for c in chunks[:20])
    result = generate_answer(
        question=question_hint,
        context=context,
        chunks=chunks,
        provider=provider,
    )
    return result["answer"]


# ─────────────────────────────────────────────────────────────────────────────
# Map-reduce summarization for long documents
# ─────────────────────────────────────────────────────────────────────────────

def map_reduce_summarize(
    chunks: list[dict[str, Any]],
    map_batch_size: int = 10,
    num_final_sentences: int = 10,
) -> str:
    """
    Map-reduce extractive summarization for long documents.

    Strategy
    ────────
    1. **Map** — split chunks into batches of ``map_batch_size``.
       Summarize each batch independently into 3 sentences.
    2. **Reduce** — combine all batch summaries and run one final
       TF-IDF pass to select the top ``num_final_sentences``.

    This respects context limits by never feeding the entire document
    to a single summarization pass.

    Parameters
    ----------
    chunks : list[dict]
        Chunk dicts from the vector store.
    map_batch_size : int
        Number of chunks per map-phase batch.
    num_final_sentences : int
        Target length of the final summary in sentences.

    Returns
    -------
    str
        Final combined summary.
    """
    if not chunks:
        return "No content available for summarization."

    # Map phase: summarize each batch into 3 sentences
    batch_summaries: list[dict[str, Any]] = []
    for i in range(0, len(chunks), map_batch_size):
        batch = chunks[i : i + map_batch_size]
        summary_text = summarize_chunks(batch, num_sentences=3)
        # Wrap as a pseudo-chunk so reduce phase can process uniformly
        batch_summaries.append({
            "text": summary_text,
            "source": batch[0].get("source", "unknown"),
            "page": batch[0].get("page", 1),
            "total_pages": batch[0].get("total_pages", 1),
            "chunk_index": i,
            "chunk_id": f"map_batch_{i}",
        })

    # Reduce phase: one final summarization pass over all batch summaries
    return summarize_chunks(batch_summaries, num_sentences=num_final_sentences)
