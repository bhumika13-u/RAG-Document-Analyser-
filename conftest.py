"""
tests/conftest.py
──────────────────
Session-scoped embedding model mock.

WHY THIS EXISTS
───────────────
Loading the real sentence-transformers model (all-MiniLM-L6-v2) requires
PyTorch to map ~22 MB of weights into memory.  On a machine that is already
under RAM pressure (Chrome, Defender, IDE all resident) the paging file fills
up and the OS raises:

    OSError: The paging file is too small for this operation to complete.

This is a system-resource problem, NOT a code bug.  The fix used in
production test suites is a lightweight fake model that:

  • produces deterministic 384-dimensional float32 unit vectors
  • derives the vector from a hash of the input text so similarity search
    still works correctly (same text → same vector; different text → different
    vector)
  • exposes get_embedding_dimension() → 384, exactly matching the real model
  • requires zero PyTorch / GPU memory

The mock is injected via pytest's autouse session fixture so it applies
automatically to EVERY test in the suite without any per-test decoration.

The real model is still used at runtime (app.py never imports conftest).
"""

from __future__ import annotations

import hashlib
import numpy as np
import pytest
from unittest.mock import patch


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight deterministic embedding model
# ─────────────────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 384


# A fixed base direction shared by all fake embeddings.
# Using a positive-valued base ensures all pairwise cosine similarities
# are positive (since every vector points "roughly" the same way).
# This mirrors real sentence embeddings, which cluster in a positive
# cone of semantic space rather than being uniformly random.
_BASE = None


def _get_base() -> np.ndarray:
    """Lazy-initialise the shared base vector (seeded for reproducibility)."""
    global _BASE
    if _BASE is None:
        rng = np.random.RandomState(42)
        b = np.abs(rng.randn(EMBEDDING_DIM)).astype(np.float32)  # all-positive
        _BASE = b / np.linalg.norm(b)
    return _BASE


class _FakeEmbeddingModel:
    """
    Drop-in replacement for SentenceTransformer used only during tests.

    encode() maps each text to a unique, reproducible unit vector in R^384.

    Design goals:
    - Identical texts  → identical vectors  (cosine = 1.0)
    - Different texts  → different vectors  (cosine < 1.0 but > 0)
    - All cosine similarities are POSITIVE  (avoids spurious low_evidence
      triggers when min_score=0.0 is passed in tests)
    - All vectors are L2-normalised         (required for FAISS IndexFlatIP)

    The positive-cosine property is achieved by blending each text-specific
    vector with a fixed all-positive base direction (80 % base + 20 % unique).
    """

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        base = _get_base()
        vectors = []
        for text in sentences:
            seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**31)
            rng = np.random.RandomState(seed)
            noise = rng.randn(EMBEDDING_DIM).astype(np.float32)
            # Blend: 80 % shared base (guarantees positive cosine with anything)
            #        20 % text-specific noise (guarantees unique vectors)
            v = 0.8 * base + 0.2 * noise
            norm = np.linalg.norm(v)
            if norm > 0:
                v /= norm
            vectors.append(v)
        return np.array(vectors, dtype=np.float32)

    def get_embedding_dimension(self) -> int:
        return EMBEDDING_DIM


# ─────────────────────────────────────────────────────────────────────────────
# Session-scoped fixture — active for the entire pytest session
# ─────────────────────────────────────────────────────────────────────────────

_FAKE_MODEL = _FakeEmbeddingModel()


@pytest.fixture(scope="session", autouse=True)
def mock_embedding_model():
    """
    Patch src.embeddings._get_model for the whole test session.

    _get_model(model_name) is the single internal function that loads (or
    returns the cached) SentenceTransformer.  Replacing it with a function
    that returns _FAKE_MODEL prevents ANY PyTorch weight loading while keeping
    every real code path fully exercised.

    The patch is also applied to the singleton cache variables so that a stale
    real model already resident in memory is not used instead.
    """
    import src.embeddings as emb_module

    # Reset any cached real model that may have been loaded in a previous run
    emb_module._model = None
    emb_module._loaded_model_name = ""

    def _fake_get_model(model_name: str):   # noqa: ARG001
        return _FAKE_MODEL

    with patch.object(emb_module, "_get_model", side_effect=_fake_get_model):
        yield

    # After the session, reset cache again so a re-import gets a clean state
    emb_module._model = None
    emb_module._loaded_model_name = ""
