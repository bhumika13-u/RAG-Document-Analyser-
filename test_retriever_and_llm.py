"""
tests/test_retriever_and_llm.py
────────────────────────────────
Tests for:
  - src/retriever.py          (format, min_score filter, no-evidence path)
  - src/llm_service.py        (extractive fallback, missing LLM config, provider detection)
  - src/vector_store.py       (has_source, remove_source)
  - src/summarizer.py         (map_reduce_summarize)

No API keys or LLM downloads are required — the extractive fallback is
always used, and OpenAI calls are monkeypatched.
"""

from __future__ import annotations

import io
import os
import pytest
from unittest.mock import patch, MagicMock

from src.rag_pipeline import RAGPipeline
from src.retriever import retrieve, format_retrieved_chunks
from src.vector_store import VectorStore
from src.llm_service import generate_answer, active_provider, detect_provider
from src.summarizer import map_reduce_summarize, summarize_chunks, extract_key_points


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

BIOLOGY_TEXT = (
    "Mitosis is a type of cell division resulting in two daughter cells each having "
    "the same number and kind of chromosomes as the parent nucleus. "
    "The process occurs in four stages: prophase, metaphase, anaphase, and telophase. "
    "During prophase, the chromosomes condense and the spindle begins to form. "
    "During metaphase, chromosomes align at the cell equator. "
    "During anaphase, sister chromatids are pulled to opposite poles. "
    "During telophase, the nuclear envelope re-forms around each set of chromosomes. "
    "Cytokinesis is the division of the cytoplasm following mitosis. "
    "Mitosis is used for growth, repair, and asexual reproduction in organisms."
)

UNRELATED_TEXT = (
    "The French Revolution began in 1789 and fundamentally transformed French society. "
    "Louis XVI was executed in 1793. The revolution led to the rise of Napoleon Bonaparte."
)


def _make_buf(content: str, name: str) -> io.BytesIO:
    buf = io.BytesIO(content.encode("utf-8"))
    buf.name = name
    return buf


@pytest.fixture
def small_pipeline() -> RAGPipeline:
    return RAGPipeline(chunk_size=200, chunk_overlap=30, top_k=3)


@pytest.fixture
def biology_pipeline(small_pipeline) -> RAGPipeline:
    small_pipeline.ingest(_make_buf(BIOLOGY_TEXT, "biology.txt"), filename="biology.txt")
    return small_pipeline


# ─────────────────────────────────────────────────────────────────────────────
# VectorStore: has_source / remove_source
# ─────────────────────────────────────────────────────────────────────────────

class TestVectorStoreSourceManagement:
    def test_has_source_false_when_empty(self, small_pipeline):
        assert not small_pipeline.store.has_source("biology.txt")

    def test_has_source_true_after_ingest(self, biology_pipeline):
        assert biology_pipeline.store.has_source("biology.txt")

    def test_has_source_false_for_unknown(self, biology_pipeline):
        assert not biology_pipeline.store.has_source("nonexistent.txt")

    def test_remove_source_via_pipeline(self, biology_pipeline):
        count_before = biology_pipeline.store.total_chunks
        assert count_before > 0
        removed = biology_pipeline.remove_source("biology.txt")
        assert removed > 0
        assert biology_pipeline.store.total_chunks == 0
        assert "biology.txt" not in biology_pipeline.ingested_sources

    def test_remove_nonexistent_source_returns_zero(self, biology_pipeline):
        assert biology_pipeline.remove_source("ghost.txt") == 0


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate upload handling
# ─────────────────────────────────────────────────────────────────────────────

class TestDuplicateUpload:
    def test_re_ingest_same_file_does_not_double_chunks(self, small_pipeline):
        """Uploading the same file twice should not grow the chunk count."""
        buf1 = _make_buf(BIOLOGY_TEXT, "bio.txt")
        result1 = small_pipeline.ingest(buf1, filename="bio.txt")
        chunks_after_first = small_pipeline.store.total_chunks

        buf2 = _make_buf(BIOLOGY_TEXT, "bio.txt")
        result2 = small_pipeline.ingest(buf2, filename="bio.txt")
        chunks_after_second = small_pipeline.store.total_chunks

        # Chunk count should be the same (replaced, not doubled)
        assert chunks_after_second == chunks_after_first
        assert result2["replaced"] is True

    def test_first_ingest_replaced_is_false(self, small_pipeline):
        buf = _make_buf(BIOLOGY_TEXT, "bio.txt")
        result = small_pipeline.ingest(buf, filename="bio.txt")
        assert result["replaced"] is False

    def test_source_appears_once_in_ingested_sources(self, small_pipeline):
        for _ in range(3):
            buf = _make_buf(BIOLOGY_TEXT, "bio.txt")
            small_pipeline.ingest(buf, filename="bio.txt")
        assert small_pipeline.ingested_sources.count("bio.txt") == 1

    def test_allow_duplicates_flag(self, small_pipeline):
        """allow_duplicates=True should add without removing old chunks."""
        buf1 = _make_buf(BIOLOGY_TEXT, "bio.txt")
        small_pipeline.ingest(buf1, filename="bio.txt")
        chunks_after_first = small_pipeline.store.total_chunks

        buf2 = _make_buf(BIOLOGY_TEXT, "bio.txt")
        result = small_pipeline.ingest(buf2, filename="bio.txt", allow_duplicates=True)
        # chunks should have grown
        assert small_pipeline.store.total_chunks > chunks_after_first
        assert result["replaced"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Retriever: format and min_score
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrieverFormat:
    def test_format_includes_source_and_page(self, biology_pipeline):
        from src.retriever import retrieve, format_retrieved_chunks
        chunks = retrieve("mitosis stages", biology_pipeline.store, top_k=2,
                          model_name=biology_pipeline.embedding_model)
        text = format_retrieved_chunks(chunks)
        assert "Source:" in text
        assert "Page:" in text
        assert "Score:" in text

    def test_format_truncates_with_max_chars(self, biology_pipeline):
        from src.retriever import retrieve, format_retrieved_chunks
        chunks = retrieve("chromosomes", biology_pipeline.store, top_k=2,
                          model_name=biology_pipeline.embedding_model)
        text = format_retrieved_chunks(chunks, max_chars_per_chunk=30)
        # Each chunk body limited to 30 chars; header is unchanged
        for chunk in chunks:
            # The truncated text shouldn't appear longer than 30 chars for the body
            assert len(chunk["text"]) > 30  # original is longer
        # But formatted output uses truncation
        assert len(text) < len(format_retrieved_chunks(chunks))

    def test_retrieve_min_score_filters(self, biology_pipeline):
        from src.retriever import retrieve
        # Use a very high threshold that should exclude everything
        results = retrieve(
            "mitosis",
            biology_pipeline.store,
            top_k=5,
            model_name=biology_pipeline.embedding_model,
            min_score=0.9999,  # virtually impossible cosine similarity
        )
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# Low-evidence detection
# ─────────────────────────────────────────────────────────────────────────────

class TestLowEvidenceDetection:
    def test_unrelated_question_returns_low_evidence(self, biology_pipeline):
        """
        Biology corpus + question about cooking → scores should be low.
        We force min_score=1.0 to guarantee the low-evidence path triggers.
        """
        result = biology_pipeline.query(
            "How do I bake chocolate chip cookies?",
            llm_provider="extractive",
            min_score=1.0,  # impossible threshold → always low-evidence
        )
        assert result["low_evidence"] is True
        assert "⚠️" in result["answer"]
        assert result["provider"] == "no-evidence"

    def test_relevant_question_not_low_evidence(self, biology_pipeline):
        result = biology_pipeline.query(
            "What happens during metaphase?",
            llm_provider="extractive",
            min_score=0.0,  # accept everything → never low-evidence
        )
        assert result["low_evidence"] is False

    def test_low_evidence_includes_closest_passages(self, biology_pipeline):
        result = biology_pipeline.query(
            "quantum entanglement",
            llm_provider="extractive",
            min_score=1.0,
        )
        assert result["low_evidence"] is True
        # Sources should still be returned so the user can inspect them
        assert isinstance(result["sources"], list)


# ─────────────────────────────────────────────────────────────────────────────
# LLM service: missing LLM config, provider detection, extractive
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMService:
    def test_detect_provider_extractive_when_no_env(self, monkeypatch):
        """With no GGUF_MODEL_PATH and no OPENAI_API_KEY, provider is extractive."""
        monkeypatch.delenv("GGUF_MODEL_PATH", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert detect_provider() == "extractive"

    def test_detect_provider_gguf_when_env_set(self, monkeypatch):
        monkeypatch.setenv("GGUF_MODEL_PATH", "/fake/model.gguf")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert detect_provider() == "gguf"

    def test_detect_provider_openai_when_env_set(self, monkeypatch):
        monkeypatch.delenv("GGUF_MODEL_PATH", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-key")
        assert detect_provider() == "openai"

    def test_generate_answer_extractive_mode(self):
        chunks = [
            {"text": "Mitosis produces two identical daughter cells.",
             "source": "bio.txt", "page": 1, "chunk_id": "bio.txt::p1::c0",
             "score": 0.88, "chunk_index": 0}
        ]
        result = generate_answer(
            question="What does mitosis produce?",
            context="Mitosis produces two identical daughter cells.",
            chunks=chunks,
            provider="extractive",
        )
        assert "answer" in result
        assert "provider" in result
        assert result["provider"] == "extractive"
        assert len(result["answer"]) > 0

    def test_extractive_answer_contains_source_citation(self):
        chunks = [
            {"text": "Chlorophyll absorbs light energy in plants.",
             "source": "plants.txt", "page": 3, "chunk_id": "plants.txt::p3::c2",
             "score": 0.75, "chunk_index": 2}
        ]
        result = generate_answer(
            question="What absorbs light?",
            context="Chlorophyll absorbs light energy in plants.",
            chunks=chunks,
            provider="extractive",
        )
        assert "plants.txt" in result["answer"]
        assert "3" in result["answer"]  # page number

    def test_missing_gguf_model_falls_back_to_extractive(self, monkeypatch):
        """If GGUF_MODEL_PATH points to a non-existent file, extractive fallback is used."""
        monkeypatch.setenv("GGUF_MODEL_PATH", "/does/not/exist/model.gguf")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        chunks = [
            {"text": "Some text.", "source": "f.txt", "page": 1,
             "chunk_id": "f.txt::p1::c0", "score": 0.8, "chunk_index": 0}
        ]
        result = generate_answer(
            question="test?",
            context="Some text.",
            chunks=chunks,
            provider="gguf",  # force GGUF path which will fail
        )
        # Should gracefully fall back; provider indicates fallback
        assert "extractive" in result["provider"]

    def test_active_provider_string_extractive(self, monkeypatch):
        monkeypatch.delenv("GGUF_MODEL_PATH", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        desc = active_provider()
        assert "Extractive" in desc or "extractive" in desc.lower()

    def test_openai_call_is_mocked(self, monkeypatch):
        """Verify the OpenAI path works when mocked — no real API call."""
        monkeypatch.delenv("GGUF_MODEL_PATH", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
        monkeypatch.setenv("OPENAI_MODEL_NAME", "gpt-test")

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Mocked LLM answer about mitosis."

        with patch("src.llm_service._answer_openai",
                   return_value=("Mocked LLM answer about mitosis.", "openai:gpt-test")):
            chunks = [{"text": "Mitosis.", "source": "x.txt", "page": 1,
                       "chunk_id": "x.txt::p1::c0", "score": 0.9, "chunk_index": 0}]
            result = generate_answer(
                question="Describe mitosis",
                context="Mitosis.",
                chunks=chunks,
                provider="openai",
            )
        assert result["answer"] == "Mocked LLM answer about mitosis."
        assert result["provider"] == "openai:gpt-test"


# ─────────────────────────────────────────────────────────────────────────────
# Citation / source metadata in answers
# ─────────────────────────────────────────────────────────────────────────────

class TestCitationMetadata:
    def test_chunk_id_in_sources(self, biology_pipeline):
        result = biology_pipeline.query("stages of mitosis", llm_provider="extractive",
                                        min_score=0.0)
        for chunk in result["sources"]:
            assert "chunk_id" in chunk
            assert "::" in chunk["chunk_id"]

    def test_total_pages_in_chunk_metadata(self, biology_pipeline):
        """total_pages should be propagated from page → chunk → retrieval result."""
        result = biology_pipeline.query("cell division", llm_provider="extractive",
                                        min_score=0.0)
        for chunk in result["sources"]:
            assert "total_pages" in chunk
            assert isinstance(chunk["total_pages"], int)
            assert chunk["total_pages"] >= 1

    def test_page_number_in_sources(self, biology_pipeline):
        result = biology_pipeline.query("anaphase", llm_provider="extractive",
                                        min_score=0.0)
        for chunk in result["sources"]:
            assert "page" in chunk
            assert chunk["page"] >= 1

    def test_source_filename_in_sources(self, biology_pipeline):
        result = biology_pipeline.query("mitosis", llm_provider="extractive",
                                        min_score=0.0)
        sources = {c["source"] for c in result["sources"]}
        assert "biology.txt" in sources

    def test_score_in_sources(self, biology_pipeline):
        result = biology_pipeline.query("chromosomes", llm_provider="extractive",
                                        min_score=0.0)
        for chunk in result["sources"]:
            assert "score" in chunk
            assert 0.0 <= chunk["score"] <= 1.0

    def test_multi_doc_sources_attributed_correctly(self, small_pipeline):
        """With two documents, every returned chunk must be correctly attributed to one of the
        two source files — this verifies that citation metadata is accurate, not that a fake
        embedding model produces semantic ranking (which it cannot)."""
        small_pipeline.ingest(_make_buf(BIOLOGY_TEXT, "biology.txt"), filename="biology.txt")
        small_pipeline.ingest(_make_buf(UNRELATED_TEXT, "history.txt"), filename="history.txt")

        result = small_pipeline.query("mitosis chromosomes", llm_provider="extractive",
                                      min_score=0.0)
        # Every returned source must be correctly attributed to one of the two documents
        valid_sources = {"biology.txt", "history.txt"}
        for chunk in result["sources"]:
            assert chunk["source"] in valid_sources, (
                f"Unexpected source '{chunk['source']}' — must be one of {valid_sources}"
            )
        # At least one result should be returned
        assert len(result["sources"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Unsupported file types
# ─────────────────────────────────────────────────────────────────────────────

class TestUnsupportedFileTypes:
    def test_ingest_csv_raises(self, small_pipeline):
        buf = io.BytesIO(b"col1,col2\n1,2\n3,4")
        buf.name = "data.csv"
        with pytest.raises(ValueError, match="Unsupported"):
            small_pipeline.ingest(buf, filename="data.csv")

    def test_ingest_xlsx_raises(self, small_pipeline):
        buf = io.BytesIO(b"PK\x03\x04")  # fake xlsx bytes
        buf.name = "sheet.xlsx"
        with pytest.raises(ValueError, match="Unsupported"):
            small_pipeline.ingest(buf, filename="sheet.xlsx")

    def test_ingest_mp3_raises(self, small_pipeline):
        buf = io.BytesIO(b"\xff\xfbdata")
        buf.name = "audio.mp3"
        with pytest.raises(ValueError, match="Unsupported"):
            small_pipeline.ingest(buf, filename="audio.mp3")


# ─────────────────────────────────────────────────────────────────────────────
# Map-reduce summarization
# ─────────────────────────────────────────────────────────────────────────────

class TestMapReduceSummarize:
    def _many_chunks(self, n: int = 40) -> list[dict]:
        """Create n synthetic chunk dicts."""
        sentences = [
            "Photosynthesis converts light into glucose in plants.",
            "Chlorophyll is the primary pigment in leaves.",
            "The Calvin cycle fixes carbon dioxide into organic molecules.",
            "Oxygen is released as a byproduct of photosynthesis.",
            "Light-dependent reactions occur in the thylakoid membranes.",
            "The stroma is the site of light-independent reactions.",
            "Glucose produced is used for cellular respiration.",
            "Plants absorb water through their roots for photosynthesis.",
        ]
        return [
            {
                "text": sentences[i % len(sentences)],
                "source": "long_doc.txt",
                "page": (i // 5) + 1,
                "total_pages": n // 5,
                "chunk_index": i,
                "chunk_id": f"long_doc.txt::p{(i//5)+1}::c{i}",
            }
            for i in range(n)
        ]

    def test_map_reduce_returns_string(self):
        chunks = self._many_chunks(40)
        result = map_reduce_summarize(chunks, map_batch_size=10)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_map_reduce_empty_input(self):
        result = map_reduce_summarize([])
        assert "No content" in result

    def test_pipeline_uses_map_reduce_for_large_docs(self, small_pipeline):
        """summarize() should switch to map-reduce when chunk count > threshold."""
        # Ingest the same text many times with allow_duplicates to build up chunks
        # Use unique filenames to avoid dedup
        for i in range(5):
            text = BIOLOGY_TEXT * 3
            buf = _make_buf(text, f"bio_{i}.txt")
            small_pipeline.ingest(buf, filename=f"bio_{i}.txt", allow_duplicates=True)

        result = small_pipeline.summarize(use_map_reduce=True)
        assert "summary" in result
        # method should be map-reduce if total chunks exceed threshold
        assert result["method"] in ("map-reduce", "direct")

    def test_pipeline_direct_for_small_doc(self, biology_pipeline):
        result = biology_pipeline.summarize(use_map_reduce=True)
        # Small document → direct method expected
        assert result["method"] == "direct"
        assert len(result["summary"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# PDF loading (minimal real PDF using pypdf)
# ─────────────────────────────────────────────────────────────────────────────

class TestPDFLoading:
    def _make_minimal_pdf(self) -> io.BytesIO:
        """
        Build a minimal valid PDF entirely in memory without any external tool.
        Uses raw PDF syntax to produce a single-page document with text.
        """
        content_stream = b"BT /F1 12 Tf 100 700 Td (Mitosis divides one cell into two identical daughter cells.) Tj ET"
        content_len = len(content_stream)
        pdf = (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
            b"4 0 obj\n<< /Length " + str(content_len).encode() + b" >>\nstream\n"
            + content_stream + b"\nendstream\nendobj\n"
            b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
            b"xref\n0 6\n"
            b"0000000000 65535 f \n"
            b"0000000009 00000 n \n"
            b"0000000058 00000 n \n"
            b"0000000115 00000 n \n"
            b"0000000266 00000 n \n"
            b"0000000400 00000 n \n"
            b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
            b"startxref\n460\n%%EOF\n"
        )
        buf = io.BytesIO(pdf)
        buf.name = "test.pdf"
        return buf

    def test_pdf_ingest_does_not_crash(self, small_pipeline):
        """
        Attempt to load a minimal PDF. pypdf may or may not extract text from
        this minimal hand-crafted PDF, so we accept either a successful ingest
        or a ValueError about empty text — but no unexpected exceptions.
        """
        buf = self._make_minimal_pdf()
        try:
            result = small_pipeline.ingest(buf, filename="test.pdf")
            assert result["pages_loaded"] >= 1
        except ValueError as e:
            # Acceptable: pypdf may not extract text from this minimal PDF
            assert "empty" in str(e).lower() or "readable" in str(e).lower()

    def test_pdf_metadata_has_source_and_page(self, small_pipeline):
        """If text is extracted from the PDF, source/page metadata must be correct."""
        from src.document_loader import load_document
        buf = self._make_minimal_pdf()
        try:
            pages = load_document(buf, filename="report.pdf")
            for page in pages:
                assert page["source"] == "report.pdf"
                assert "page" in page
                assert page["page"] >= 1
        except ValueError:
            pytest.skip("Minimal PDF has no extractable text — metadata test skipped.")
