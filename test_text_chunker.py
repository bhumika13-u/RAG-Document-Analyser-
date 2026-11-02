"""
tests/test_text_chunker.py
───────────────────────────
Unit tests for src/text_chunker.py.
"""

import pytest
from src.text_chunker import chunk_pages, get_chunk_texts, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_pages(texts: list[str], source: str = "doc.txt") -> list[dict]:
    return [
        {"text": t, "source": source, "page": i + 1, "total_pages": len(texts)}
        for i, t in enumerate(texts)
    ]


SHORT_TEXT = "Hello world. This is a short sentence."

LONG_TEXT = (
    "Artificial intelligence (AI) is intelligence demonstrated by machines, "
    "as opposed to the natural intelligence displayed by animals including humans. "
    "AI research has been defined as the field of study of intelligent agents, "
    "which refers to any system that perceives its environment and takes actions "
    "that maximize its chance of achieving its goals. "
    "The term 'artificial intelligence' had previously been used to describe "
    "machines that mimic and display human cognitive skills associated with the "
    "human mind, such as learning and problem-solving. "
    "This definition has since been rejected by major AI researchers who now "
    "describe AI in terms of rationality and acting rationally. "
    "Machine learning is a method of data analysis that automates analytical model building. "
    "It is based on the idea that systems can learn from data, identify patterns, "
    "and make decisions with minimal human intervention. "
) * 5   # repeat to ensure multiple chunks


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkPages:
    def test_returns_list(self):
        pages = _make_pages([SHORT_TEXT])
        result = chunk_pages(pages)
        assert isinstance(result, list)

    def test_each_chunk_has_required_keys(self):
        pages = _make_pages([LONG_TEXT])
        chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=50)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "text" in chunk
            assert "source" in chunk
            assert "page" in chunk
            assert "chunk_index" in chunk
            assert "chunk_id" in chunk

    def test_chunk_size_respected(self):
        pages = _make_pages([LONG_TEXT])
        chunk_size = 300
        chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=50)
        for chunk in chunks:
            # Allow slight overshoot at sentence boundaries (≤ 20% overshoot)
            assert len(chunk["text"]) <= chunk_size * 1.2, (
                f"Chunk too long: {len(chunk['text'])} chars"
            )

    def test_multiple_pages_produce_chunks_from_all(self):
        pages = _make_pages([LONG_TEXT, LONG_TEXT], source="multi.pdf")
        chunks = chunk_pages(pages)
        page_numbers = {c["page"] for c in chunks}
        assert 1 in page_numbers
        assert 2 in page_numbers

    def test_source_propagated(self):
        pages = _make_pages([LONG_TEXT], source="myfile.pdf")
        chunks = chunk_pages(pages)
        assert all(c["source"] == "myfile.pdf" for c in chunks)

    def test_chunk_ids_are_unique(self):
        pages = _make_pages([LONG_TEXT, LONG_TEXT])
        chunks = chunk_pages(pages)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_overlap_too_large_raises(self):
        pages = _make_pages([LONG_TEXT])
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_pages(pages, chunk_size=100, chunk_overlap=150)

    def test_empty_pages_returns_empty(self):
        pages = _make_pages(["   ", ""])
        result = chunk_pages(pages)
        assert result == []


class TestGetChunkTexts:
    def test_extracts_texts(self):
        pages = _make_pages([LONG_TEXT])
        chunks = chunk_pages(pages)
        texts = get_chunk_texts(chunks)
        assert isinstance(texts, list)
        assert all(isinstance(t, str) for t in texts)
        assert len(texts) == len(chunks)
