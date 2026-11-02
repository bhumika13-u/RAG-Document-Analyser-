"""
tests/test_rag_pipeline.py
───────────────────────────
Integration-style tests for src/rag_pipeline.py.

These tests run without any external LLM by relying on the built-in
extractive fallback mode.  No API keys or model downloads beyond the
embedding model are required.
"""

import io
import pytest
from src.rag_pipeline import RAGPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_TEXT = (
    "Photosynthesis is the process by which green plants, algae, and some bacteria "
    "convert light energy into chemical energy stored as glucose. "
    "This process occurs mainly in the chloroplasts of plant cells. "
    "The overall equation is: 6CO2 + 6H2O + light → C6H12O6 + 6O2. "
    "Chlorophyll is the primary pigment responsible for absorbing light energy. "
    "The light-dependent reactions occur in the thylakoid membranes, while "
    "the light-independent reactions (Calvin cycle) occur in the stroma. "
    "Photosynthesis is fundamental to life on Earth as it produces oxygen and "
    "organic compounds used by almost all living organisms. "
    "Factors affecting the rate of photosynthesis include light intensity, "
    "carbon dioxide concentration, temperature, and water availability."
)


def _make_txt_buf(content: str, name: str = "test.txt") -> io.BytesIO:
    buf = io.BytesIO(content.encode("utf-8"))
    buf.name = name
    return buf


@pytest.fixture
def pipeline() -> RAGPipeline:
    """A fresh pipeline with small chunks for fast tests."""
    return RAGPipeline(chunk_size=200, chunk_overlap=40, top_k=3)


@pytest.fixture
def loaded_pipeline(pipeline: RAGPipeline) -> RAGPipeline:
    """Pipeline with SAMPLE_TEXT already ingested."""
    buf = _make_txt_buf(SAMPLE_TEXT, name="photosynthesis.txt")
    pipeline.ingest(buf, filename="photosynthesis.txt")
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Ingest tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIngest:
    def test_pipeline_empty_initially(self, pipeline):
        assert not pipeline.is_ready
        assert pipeline.store.total_chunks == 0

    def test_ingest_txt(self, pipeline):
        buf = _make_txt_buf(SAMPLE_TEXT)
        result = pipeline.ingest(buf, filename="sample.txt")
        assert result["pages_loaded"] >= 1
        assert result["chunks_added"] > 0
        assert pipeline.is_ready

    def test_ingest_records_source(self, pipeline):
        buf = _make_txt_buf(SAMPLE_TEXT, name="bio.txt")
        pipeline.ingest(buf, filename="bio.txt")
        assert "bio.txt" in pipeline.ingested_sources

    def test_ingest_empty_raises(self, pipeline):
        buf = _make_txt_buf("   ", name="empty.txt")
        with pytest.raises(ValueError):
            pipeline.ingest(buf, filename="empty.txt")

    def test_ingest_many(self, pipeline):
        files = [
            _make_txt_buf("Document one about machine learning models.", "doc1.txt"),
            _make_txt_buf("Document two about natural language processing.", "doc2.txt"),
        ]
        results = pipeline.ingest_many(files, filenames=["doc1.txt", "doc2.txt"])
        assert len(results) == 2
        assert pipeline.store.total_chunks > 0


# ─────────────────────────────────────────────────────────────────────────────
# Query tests
# ─────────────────────────────────────────────────────────────────────────────

class TestQuery:
    def test_query_returns_answer(self, loaded_pipeline):
        result = loaded_pipeline.query("What is photosynthesis?", llm_provider="extractive",
                                       min_score=0.0)
        assert "answer" in result
        assert len(result["answer"]) > 10

    def test_query_has_sources(self, loaded_pipeline):
        result = loaded_pipeline.query("Where does photosynthesis occur?", llm_provider="extractive",
                                       min_score=0.0)
        assert "sources" in result
        assert len(result["sources"]) > 0

    def test_query_source_metadata(self, loaded_pipeline):
        result = loaded_pipeline.query("What is chlorophyll?", llm_provider="extractive",
                                       min_score=0.0)
        for chunk in result["sources"]:
            assert "source" in chunk
            assert "page" in chunk
            assert "score" in chunk
            assert 0.0 <= chunk["score"] <= 1.0

    def test_query_empty_question_raises(self, loaded_pipeline):
        with pytest.raises(ValueError, match="Query must not be empty"):
            loaded_pipeline.query("")

    def test_query_empty_store_raises(self, pipeline):
        with pytest.raises(ValueError, match="empty"):
            pipeline.query("What is AI?")

    def test_relevant_chunk_retrieved(self, loaded_pipeline):
        """The retrieved chunks should contain terms related to the query."""
        result = loaded_pipeline.query("Calvin cycle", llm_provider="extractive",
                                       min_score=0.0)
        # At minimum the relevant source document should have been hit
        assert any(c["source"] == "photosynthesis.txt" for c in result["sources"])

    def test_query_has_low_evidence_key(self, loaded_pipeline):
        result = loaded_pipeline.query("What is photosynthesis?", llm_provider="extractive",
                                       min_score=0.0)
        assert "low_evidence" in result
        assert result["low_evidence"] is False


# ─────────────────────────────────────────────────────────────────────────────
# Summarization tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSummarize:
    def test_summarize_returns_dict(self, loaded_pipeline):
        result = loaded_pipeline.summarize()
        assert "summary" in result
        assert "key_points" in result

    def test_summary_is_non_empty(self, loaded_pipeline):
        result = loaded_pipeline.summarize()
        assert len(result["summary"]) > 10

    def test_key_points_is_list(self, loaded_pipeline):
        result = loaded_pipeline.summarize()
        assert isinstance(result["key_points"], list)
        assert len(result["key_points"]) > 0

    def test_source_filter(self, loaded_pipeline):
        result = loaded_pipeline.summarize(source_filter="photosynthesis.txt")
        assert result["source_filter"] == "photosynthesis.txt"

    def test_empty_pipeline_summarize_raises(self, pipeline):
        with pytest.raises(ValueError):
            pipeline.summarize()

    def test_summarize_has_method_key(self, loaded_pipeline):
        result = loaded_pipeline.summarize()
        assert "method" in result
        assert result["method"] in ("direct", "map-reduce")

    def test_summarize_has_chunks_used_key(self, loaded_pipeline):
        result = loaded_pipeline.summarize()
        assert "chunks_used" in result
        assert result["chunks_used"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Reset test
# ─────────────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_store(self, loaded_pipeline):
        assert loaded_pipeline.is_ready
        loaded_pipeline.reset()
        assert not loaded_pipeline.is_ready
        assert loaded_pipeline.ingested_sources == []
