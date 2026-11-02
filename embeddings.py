"""
embeddings.py
─────────────
Wraps sentence-transformers to produce dense vector embeddings.

⚠️  Embeddings vs. the answer-generating LLM — what's the difference?
──────────────────────────────────────────────────────────────────────
• **Embeddings** convert a piece of text into a fixed-length numeric vector
  that captures its *meaning*.  They are used purely for similarity search —
  finding which stored chunks are most relevant to a query.  They do NOT
  generate new text.

• **LLM (Large Language Model)** (see llm_service.py) takes the retrieved
  chunks plus the user's question and *generates* a coherent answer.

Both are neural networks, but they serve completely different roles.  The
embedding model runs 100% locally and for free; the LLM is optional and
configurable.

Default model: ``sentence-transformers/all-MiniLM-L6-v2``
  - 384-dimensional vectors
  - ~22 MB download on first use
  - Fast on CPU (~5 ms per sentence on a modern laptop)
  - Downloaded automatically to ~/.cache/huggingface/
"""

from __future__ import annotations

import os
import numpy as np
from typing import Optional

from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton so the model is loaded only once per process
# ─────────────────────────────────────────────────────────────────────────────

_model: Optional[SentenceTransformer] = None
_loaded_model_name: str = ""


def _get_model(model_name: str) -> SentenceTransformer:
    global _model, _loaded_model_name
    if _model is None or _loaded_model_name != model_name:
        _model = SentenceTransformer(model_name)
        _loaded_model_name = model_name
    return _model


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Convert a list of strings into a 2-D float32 numpy array.

    Parameters
    ----------
    texts : list[str]
        Texts to embed.
    model_name : str
        Hugging Face model identifier or local path.
    batch_size : int
        Number of sentences encoded per batch.
    show_progress : bool
        Print a tqdm progress bar when True.

    Returns
    -------
    np.ndarray
        Shape ``(len(texts), embedding_dim)``, dtype ``float32``.
    """
    if not texts:
        raise ValueError("embed_texts received an empty list.")

    model = _get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # unit-length → cosine ≡ dot product
    )
    return embeddings.astype("float32")


def embed_query(
    query: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> np.ndarray:
    """
    Embed a single query string.

    Returns
    -------
    np.ndarray
        Shape ``(1, embedding_dim)``, dtype ``float32``.
    """
    return embed_texts([query], model_name=model_name)


def embedding_dimension(model_name: str = DEFAULT_EMBEDDING_MODEL) -> int:
    """Return the output vector dimension of the given model."""
    model = _get_model(model_name)
    return model.get_embedding_dimension()
