"""
rag_pipeline.py
───────────────
Orchestrates the full RAG workflow end-to-end.

Ingestion flow:
    documents → load_document → chunk_pages → embed_texts → VectorStore

Query flow:
    question → embed_query → VectorStore.search → (evidence check) → generate_answer

This module is the single entry-point that app.py uses; it keeps UI code
completely separated from ML logic.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from src.document_loader import load_document
from src.text_chunker import chunk_pages, get_chunk_texts
from src.embeddings import embed_texts, embedding_dimension, DEFAULT_EMBEDDING_MODEL
from src.vector_store import VectorStore
from src.retriever import retrieve, format_retrieved_chunks
from src.llm_service import generate_answer, active_provider
from src.summarizer import summarize_chunks, extract_key_points, map_reduce_summarize

# Threshold below which all retrieved chunks are considered "not relevant enough"
_LOW_EVIDENCE_THRESHOLD = 0.25
# Threshold used to decide between simple vs. map-reduce summarization
_MAP_REDUCE_CHUNK_THRESHOLD = 30


# ─────────────────────────────────────────────────────────────────────────────
# RAGPipeline
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Stateful RAG pipeline.

    Usage
    -----
    >>> pipeline = RAGPipeline()
    >>> pipeline.ingest(file_obj, filename="report.pdf")
    >>> result = pipeline.query("What are the main findings?")
    >>> print(result["answer"])
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k: int = 5,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        faiss_dir: Optional[str] = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.faiss_dir = faiss_dir or os.getenv("FAISS_INDEX_DIR")

        dim = embedding_dimension(embedding_model)
        self.store = VectorStore(embedding_dim=dim)
        self._ingested_sources: list[str] = []   # filenames processed so far

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest(
        self,
        file_source: Any,
        filename: str = "",
        show_progress: bool = False,
        allow_duplicates: bool = False,
    ) -> dict[str, Any]:
        """
        Load, chunk, embed, and index a single document.

        Duplicate handling
        ──────────────────
        If a document with the same filename was already indexed and
        ``allow_duplicates=False`` (the default), the existing chunks are
        removed and the document is re-indexed fresh.  This prevents stale
        or doubled vectors when a user re-uploads the same file.

        Parameters
        ----------
        file_source : path or file-like
            A filesystem path or Streamlit UploadedFile.
        filename : str
            Original filename (required for file-like objects).
        show_progress : bool
            Show tqdm progress bar during embedding.
        allow_duplicates : bool
            If True, skip the duplicate check and always add.
            Defaults to False (safe, idempotent behaviour).

        Returns
        -------
        dict with:
            - ``source``        : str   — filename
            - ``pages_loaded``  : int   — number of non-empty pages
            - ``chunks_added``  : int   — number of chunks indexed
            - ``replaced``      : bool  — True if existing chunks were removed first
        """
        pages = load_document(file_source, filename)
        chunks = chunk_pages(pages, self.chunk_size, self.chunk_overlap)

        if not chunks:
            raise ValueError(
                f"No text could be extracted from '{filename or file_source}'. "
                "The file may be empty or image-only."
            )

        source_name = chunks[0]["source"]
        replaced = False

        # Duplicate detection: remove stale index for this source, then re-embed
        if not allow_duplicates and self.store.has_source(source_name):
            self._rebuild_without(source_name)
            if source_name in self._ingested_sources:
                self._ingested_sources.remove(source_name)
            replaced = True

        texts = get_chunk_texts(chunks)
        embeddings = embed_texts(
            texts,
            model_name=self.embedding_model,
            show_progress=show_progress,
        )
        self.store.add_chunks(chunks, embeddings)

        if source_name not in self._ingested_sources:
            self._ingested_sources.append(source_name)

        # Optionally persist to disk
        if self.faiss_dir:
            self.store.save(self.faiss_dir)

        return {
            "source": source_name,
            "pages_loaded": len(pages),
            "chunks_added": len(chunks),
            "replaced": replaced,
        }

    def ingest_many(
        self,
        file_sources: list[Any],
        filenames: list[str] | None = None,
        show_progress: bool = False,
    ) -> list[dict[str, Any]]:
        """Ingest multiple documents.  Returns a list of ingest result dicts."""
        filenames = filenames or [""] * len(file_sources)
        results = []
        for src, name in zip(file_sources, filenames):
            results.append(self.ingest(src, name, show_progress))
        return results

    def remove_source(self, source_name: str) -> int:
        """
        Remove all chunks for *source_name* and rebuild the FAISS index
        from the remaining chunks (re-embedding them).

        Returns the number of chunks removed.
        """
        if source_name not in self._ingested_sources:
            return 0
        removed = self._rebuild_without(source_name)
        self._ingested_sources = [s for s in self._ingested_sources if s != source_name]
        if self.faiss_dir:
            self.store.save(self.faiss_dir)
        return removed

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        llm_provider: Optional[str] = None,
        min_score: float = _LOW_EVIDENCE_THRESHOLD,
    ) -> dict[str, Any]:
        """
        Answer a question using the indexed documents.

        Low-evidence detection
        ──────────────────────
        After retrieval, if every chunk's cosine similarity is below
        ``min_score`` (default 0.25) the system honestly reports that the
        documents don't contain relevant information rather than fabricating
        an answer.

        Parameters
        ----------
        question : str
            User's question.
        top_k : int | None
            Override the default ``top_k`` for this query.
        llm_provider : str | None
            Override provider: ``'gguf'``, ``'openai'``, or ``'extractive'``.
        min_score : float
            Minimum acceptable cosine similarity.  All-below → no-evidence reply.

        Returns
        -------
        dict with:
            - ``answer``          : str
            - ``provider``        : str
            - ``sources``         : list[dict]  — retrieved chunks with scores
            - ``question``        : str
            - ``low_evidence``    : bool — True when no chunk passed the threshold
        """
        k = top_k or self.top_k
        chunks = retrieve(
            question,
            self.store,
            top_k=k,
            model_name=self.embedding_model,
        )

        # Low-evidence check: all chunks below threshold?
        low_evidence = bool(chunks) and all(c["score"] < min_score for c in chunks)
        if low_evidence or not chunks:
            no_evidence_msg = (
                "⚠️ The uploaded documents do not contain enough relevant information "
                "to answer this question confidently.\n\n"
                "The most similar passages found had low relevance scores. "
                "Consider uploading additional documents or rephrasing your question."
            )
            if chunks:
                no_evidence_msg += "\n\n**Closest passages (low relevance):**\n"
                for i, c in enumerate(chunks[:3], 1):
                    no_evidence_msg += (
                        f"\n[{i}] `{c['source']}` p.{c['page']} "
                        f"(score: {c['score']:.3f})\n> {c['text'][:200]}…\n"
                    )
            return {
                "answer": no_evidence_msg,
                "provider": "no-evidence",
                "sources": chunks,
                "question": question,
                "low_evidence": True,
            }

        context = format_retrieved_chunks(chunks)
        result = generate_answer(
            question=question,
            context=context,
            chunks=chunks,
            provider=llm_provider,
        )
        result["question"] = question
        result["low_evidence"] = False
        return result

    # ── Summarization ─────────────────────────────────────────────────────────

    def summarize(
        self,
        source_filter: Optional[str] = None,
        max_chunks: int = 30,
        use_map_reduce: bool = True,
    ) -> dict[str, Any]:
        """
        Summarize the indexed documents (or a specific source).

        For documents with more than ``max_chunks`` chunks,
        map-reduce summarization is used automatically when
        ``use_map_reduce=True`` (default) so that no single pass
        exceeds the context budget.

        Parameters
        ----------
        source_filter : str | None
            If given, only chunks from this source are summarized.
        max_chunks : int
            Simple-pass cap; beyond this, map-reduce is used.
        use_map_reduce : bool
            Enable map-reduce for long documents.

        Returns
        -------
        dict with ``summary``, ``key_points``, ``source_filter``,
        ``chunks_used``, ``method``.
        """
        chunks = list(self.store.chunks)
        if source_filter:
            chunks = [c for c in chunks if c["source"] == source_filter]

        if not chunks:
            raise ValueError(
                "No chunks available for summarization. "
                "Please upload and index documents first."
            )

        method: str
        if use_map_reduce and len(chunks) > _MAP_REDUCE_CHUNK_THRESHOLD:
            summary = map_reduce_summarize(chunks, map_batch_size=10, num_final_sentences=10)
            method = "map-reduce"
            sampled = chunks  # key_points uses all chunks for map-reduce path
        else:
            step = max(1, len(chunks) // max_chunks)
            sampled = chunks[::step][:max_chunks]
            summary = summarize_chunks(sampled)
            method = "direct"

        key_points = extract_key_points(sampled if method == "direct" else chunks)

        return {
            "summary": summary,
            "key_points": key_points,
            "source_filter": source_filter or "all documents",
            "chunks_used": len(chunks),
            "method": method,
        }

    # ── State helpers ─────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True if at least one document has been indexed."""
        return self.store.total_chunks > 0

    @property
    def ingested_sources(self) -> list[str]:
        return list(self._ingested_sources)

    def active_llm_provider(self) -> str:
        return active_provider()

    def reset(self) -> None:
        """Clear the index and start fresh."""
        dim = self.store.embedding_dim
        self.store = VectorStore(embedding_dim=dim)
        self._ingested_sources = []

    def load_from_disk(self) -> bool:
        """
        Attempt to load a persisted FAISS index from ``faiss_dir``.
        Returns True on success, False if no saved index exists.
        """
        if not self.faiss_dir:
            return False
        try:
            self.store = VectorStore.load(self.faiss_dir)
            self._ingested_sources = self.store.source_names()
            return True
        except FileNotFoundError:
            return False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _rebuild_without(self, source_name: str) -> int:
        """
        Rebuild the FAISS index excluding all chunks from *source_name*.

        Re-embeds the surviving chunks so the index stays valid.
        Returns the number of removed chunks.
        """
        surviving = [c for c in self.store.chunks if c.get("source") != source_name]
        removed = len(self.store.chunks) - len(surviving)
        if removed == 0:
            return 0

        # Rebuild store from scratch with only surviving chunks
        dim = self.store.embedding_dim
        self.store = VectorStore(embedding_dim=dim)

        if surviving:
            texts = [c["text"] for c in surviving]
            embeddings = embed_texts(texts, model_name=self.embedding_model)
            self.store.add_chunks(surviving, embeddings)

        return removed
