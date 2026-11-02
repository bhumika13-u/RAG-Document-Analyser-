"""
tests/test_document_loader.py
──────────────────────────────
Unit tests for src/document_loader.py.

These tests run without any uploaded files by creating minimal in-memory
objects so that no real documents are needed on disk.
"""

import io
import pytest
from src.document_loader import load_document, load_documents, SUPPORTED_EXTENSIONS


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_txt_file(content: str) -> io.BytesIO:
    buf = io.BytesIO(content.encode("utf-8"))
    buf.name = "sample.txt"
    return buf


def _make_docx_file(content: str) -> io.BytesIO:
    """Create a minimal in-memory .docx with a single paragraph."""
    import docx as _docx

    doc = _docx.Document()
    doc.add_paragraph(content)
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    buf.name = "sample.docx"
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSupportedExtensions:
    def test_pdf_in_extensions(self):
        assert ".pdf" in SUPPORTED_EXTENSIONS

    def test_txt_in_extensions(self):
        assert ".txt" in SUPPORTED_EXTENSIONS

    def test_docx_in_extensions(self):
        assert ".docx" in SUPPORTED_EXTENSIONS


class TestTxtLoader:
    def test_loads_simple_text(self):
        buf = _make_txt_file("Hello world. This is a test document.")
        pages = load_document(buf, filename="sample.txt")
        assert len(pages) == 1
        assert "Hello world" in pages[0]["text"]

    def test_page_metadata(self):
        buf = _make_txt_file("Test content.")
        pages = load_document(buf, filename="my_file.txt")
        assert pages[0]["source"] == "my_file.txt"
        assert pages[0]["page"] == 1
        assert pages[0]["total_pages"] == 1

    def test_latin1_fallback(self):
        # Bytes that are not valid UTF-8 but valid latin-1
        raw = io.BytesIO(b"caf\xe9 au lait")
        raw.name = "latin.txt"
        pages = load_document(raw, filename="latin.txt")
        assert "café" in pages[0]["text"] or "caf" in pages[0]["text"]

    def test_empty_txt_raises(self):
        buf = _make_txt_file("   \n  ")
        with pytest.raises(ValueError, match="empty"):
            load_document(buf, filename="empty.txt")


class TestUnsupportedFormat:
    def test_csv_raises_value_error(self):
        buf = io.BytesIO(b"col1,col2\n1,2")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_document(buf, filename="data.csv")

    def test_png_raises_value_error(self):
        buf = io.BytesIO(b"\x89PNG\r\n")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_document(buf, filename="image.png")


class TestDocxLoader:
    def test_loads_docx(self):
        buf = _make_docx_file(
            "Artificial intelligence is transforming every industry. "
            "Machine learning models are at the core of this revolution."
        )
        pages = load_document(buf, filename="sample.docx")
        assert len(pages) >= 1
        text = pages[0]["text"]
        assert "Artificial intelligence" in text

    def test_docx_source_metadata(self):
        buf = _make_docx_file("Some content here.")
        pages = load_document(buf, filename="report.docx")
        assert pages[0]["source"] == "report.docx"


class TestLoadDocuments:
    def test_multiple_txt_files(self):
        f1 = _make_txt_file("First document content about AI.")
        f2 = _make_txt_file("Second document content about ML.")
        pages = load_documents([f1, f2], filenames=["doc1.txt", "doc2.txt"])
        sources = {p["source"] for p in pages}
        assert "doc1.txt" in sources
        assert "doc2.txt" in sources
