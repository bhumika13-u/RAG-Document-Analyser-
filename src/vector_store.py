"""
vector_store.py
───────────────
Builds and queries a FAISS flat (exact) index over chunk embeddings.

The index is kept in memory and can optionally be persisted to disk so
that re-loading the same documents does not require re-embedding.

Why FAISS?
──────────
FAISS (Facebook AI Similarity Search) is a C++ library with a Python
wrapper that performs highly optimised nearest-neighbour search in
embedding space.  ``IndexFlatIP`` does exact inner-product (cosine, since
embeddings are L2-normalised) search — perfectly accurate for document
collections up to ~100 k chunks.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# VectorStore class
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """
    In-memory FAISS vector store with optional disk persistence.

    Attributes
    ----------
    index : faiss.Index
        The underlying FAISS index.
    chunks : list[dict]
        Metadata for each indexed chunk (parallel to the index vectors).
    """

    def __init__(self, embedding_dim: int) -> None:
        self.embedding_dim = embedding_dim
        # Inner-product index (cosine similarity because embeddings are
        # unit-normalised before being stored)
        self.index: faiss.Index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: list[dict[str, Any]] = []

    # ── Build ─────────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """
        Add chunk metadata and their corresponding embeddings to the store.

        Parameters
        ----------
        chunks : list[dict]
            Chunk dicts from text_chunker (must have "text", "source", etc.).
        embeddings : np.ndarray
            Float32 array of shape ``(len(chunks), embedding_dim)``.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks length ({len(chunks)}) does not match "
                f"embeddings rows ({embeddings.shape[0]})."
            )
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Return the ``top_k`` most similar chunks to the query embedding.

        Parameters
        ----------
        query_embedding : np.ndarray
            Shape ``(1, embedding_dim)`` or ``(embedding_dim,)``, float32.
        top_k : int
            Number of results to return.

        Returns
        -------
        list[dict]
            Each dict is the chunk metadata with an added ``"score"`` key
            (cosine similarity in [0, 1]).
        """
        if self.index.ntotal == 0:
            return []

        vec = np.asarray(query_embedding, dtype="float32")
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)

        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(vec, actual_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])   # shallow copy
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str | Path) -> None:
        """
        Persist the FAISS index and chunk metadata to ``directory``.

        Creates two files:
        - ``faiss.index`` — the serialised FAISS index
        - ``chunks.pkl``  — pickled list of chunk dicts
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / "faiss.index"))
        with open(directory / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        # Also save dim so we can reconstruct correctly
        with open(directory / "meta.json", "w") as f:
            json.dump({"embedding_dim": self.embedding_dim}, f)

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        """
        Load a previously saved VectorStore from ``directory``.

        Raises
        ------
        FileNotFoundError
            If the directory or expected files are missing.
        """
        directory = Path(directory)
        meta_path = directory / "meta.json"
        index_path = directory / "faiss.index"
        chunks_path = directory / "chunks.pkl"

        for p in (meta_path, index_path, chunks_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"VectorStore file not found: {p}. "
                    "Please re-index your documents."
                )

        with open(meta_path) as f:
            meta = json.load(f)

        store = cls(embedding_dim=meta["embedding_dim"])
        store.index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as f:
            store.chunks = pickle.load(f)

        return store

    # ── Info ──────────────────────────────────────────────────────────────────

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal

    def source_names(self) -> list[str]:
        """Return deduplicated list of source document names."""
        seen: set[str] = set()
        result: list[str] = []
        for c in self.chunks:
            s = c.get("source", "unknown")
            if s not in seen:
                seen.add(s)
                result.append(s)
        return result

    def has_source(self, name: str) -> bool:
        """Return True if any indexed chunk comes from *name*."""
        return any(c.get("source") == name for c in self.chunks)

    def remove_source(self, name: str) -> int:
        """
        Remove all chunks (and their vectors) belonging to *name*.

        Because FAISS IndexFlatIP does not support in-place deletion,
        we reset the index to empty and update the chunk list.  The
        surviving chunks' embeddings are NOT re-added here — the caller
        (``RAGPipeline._rebuild_without``) is responsible for re-embedding
        and re-adding the surviving chunks so the index stays consistent.

        After calling this method the store is in a *transitional* state
        (chunks list is cleared, index is empty).  Always use
        ``RAGPipeline.remove_source()`` from application code; call this
        method directly only if you immediately re-add the surviving vectors.

        Returns
        -------
        int
            Number of chunks removed.
        """
        surviving = [c for c in self.chunks if c.get("source") != name]
        removed = len(self.chunks) - len(surviving)
        if removed == 0:
            return 0

        # Reset both index and chunk list atomically so they are always
        # in sync (both empty).  Caller must re-add surviving embeddings.
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.chunks = []
        # Return removed count; caller re-adds surviving chunks via add_chunks()
        return removed
