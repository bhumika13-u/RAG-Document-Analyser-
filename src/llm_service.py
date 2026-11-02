"""
llm_service.py
──────────────
Provider-independent LLM interface for answer generation.

Architecture note — Embeddings vs. Answer-generating LLM
─────────────────────────────────────────────────────────
• ``embeddings.py`` uses **sentence-transformers** to turn text into numbers
  (vectors).  It does NOT produce new words — just numbers.
• THIS file handles a completely different model whose job is to *read* the
  retrieved chunks and *write* a coherent answer.  These are two distinct
  neural networks with different roles.

Provider selection (checked in order)
───────────────────────────────────────
1. **Local GGUF** — set ``GGUF_MODEL_PATH`` in .env.
   Requires ``pip install llama-cpp-python``.
   Runs 100 % offline, free, but needs ≥4 GB RAM for a Q4 7-B model.

2. **OpenAI-compatible API** — set ``OPENAI_API_KEY`` (and optionally
   ``OPENAI_API_BASE`` and ``OPENAI_MODEL_NAME``).
   Works with OpenAI, Together AI, Groq, Ollama's ``/v1`` endpoint, etc.

3. **Extractive fallback** (always available) — no LLM required.
   Simply returns the top retrieved passages formatted as an answer.
   Honest, transparent, and always functional.

API keys are never hardcoded; they are read exclusively from environment
variables or a .env file loaded at startup.
"""

from __future__ import annotations

import os
import textwrap
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Provider detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_provider() -> str:
    """Return 'gguf', 'openai', or 'extractive'."""
    if os.getenv("GGUF_MODEL_PATH"):
        return "gguf"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "extractive"


# Keep private alias for backward compatibility
_detect_provider = detect_provider


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a helpful AI assistant that answers questions strictly based on "
    "the provided document context. "
    "If the answer is not contained in the context, say so honestly. "
    "Do not make up information."
)

def _build_prompt(question: str, context: str) -> str:
    return textwrap.dedent(f"""
        Context extracted from the user's documents:
        ──────────────────────────────────────────────
        {context}
        ──────────────────────────────────────────────

        Question: {question}

        Instructions:
        - Answer using ONLY the information in the context above.
        - Cite which source / page the information came from when possible.
        - If the context does not contain enough information, say:
          "The uploaded documents do not contain enough information to answer this question."
        - Be concise but complete.

        Answer:
    """).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Provider implementations
# ─────────────────────────────────────────────────────────────────────────────

def _answer_gguf(question: str, context: str) -> tuple[str, str]:
    """Generate an answer using a local GGUF model via llama-cpp-python."""
    try:
        from llama_cpp import Llama  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is not installed. "
            "Run: pip install llama-cpp-python"
        ) from exc

    model_path = os.environ["GGUF_MODEL_PATH"]
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"GGUF model not found at: {model_path}\n"
            "Please download a GGUF model and update GGUF_MODEL_PATH in your .env file."
        )

    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=os.cpu_count() or 4,
        verbose=False,
    )
    prompt = _build_prompt(question, context)
    response = llm(
        prompt,
        max_tokens=512,
        temperature=0.2,
        stop=["Question:", "──────"],
    )
    answer = response["choices"][0]["text"].strip()
    return answer, "local-gguf"


def _answer_openai(question: str, context: str) -> tuple[str, str]:
    """Generate an answer using an OpenAI-compatible API."""
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "openai package is not installed. Run: pip install openai"
        ) from exc

    api_key = os.environ["OPENAI_API_KEY"]
    api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

    client = OpenAI(api_key=api_key, base_url=api_base)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_prompt(question, context)},
        ],
        temperature=0.2,
        max_tokens=512,
    )
    answer = response.choices[0].message.content.strip()
    return answer, f"openai:{model_name}"


def _answer_extractive(question: str, context: str, chunks: list[dict[str, Any]]) -> tuple[str, str]:
    """
    Extractive fallback — returns the top retrieved passages as the answer.
    No LLM needed; fully transparent about what it's doing.
    """
    intro = (
        "⚠️  No LLM is configured.  The following passages were retrieved from "
        "your documents and are the most relevant to your question.  "
        "Set GGUF_MODEL_PATH or OPENAI_API_KEY in your .env file to enable "
        "AI-generated answers.\n\n"
    )
    passages: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        passages.append(
            f"**Passage {i}** (Source: `{chunk['source']}`, Page {chunk['page']}, "
            f"Similarity: {chunk['score']:.2f})\n\n{chunk['text']}"
        )
    return intro + "\n\n---\n\n".join(passages), "extractive"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_answer(
    question: str,
    context: str,
    chunks: list[dict[str, Any]],
    provider: str | None = None,
) -> dict[str, Any]:
    """
    Generate an answer for *question* given the retrieved *context*.

    Parameters
    ----------
    question : str
        User's question.
    context : str
        Pre-formatted context string (output of ``retriever.format_retrieved_chunks``).
    chunks : list[dict]
        Raw retrieved chunks (used by the extractive fallback).
    provider : str | None
        Override provider detection: ``'gguf'``, ``'openai'``, or
        ``'extractive'``.  If ``None``, auto-detected.

    Returns
    -------
    dict with keys:
        - ``answer``  : str  — the generated (or extracted) answer
        - ``provider``: str  — which provider was used
        - ``sources`` : list[dict] — the input chunks (for UI attribution)
    """
    if provider is None:
        provider = detect_provider()

    try:
        if provider == "gguf":
            answer, used = _answer_gguf(question, context)
        elif provider == "openai":
            answer, used = _answer_openai(question, context)
        else:
            answer, used = _answer_extractive(question, context, chunks)
    except Exception as exc:
        # Graceful degradation: fall back to extractive if LLM errors out
        fallback_note = f"⚠️  LLM provider '{provider}' failed: {exc}\nFalling back to extractive mode.\n\n"
        extractive_answer, _ = _answer_extractive(question, context, chunks)
        answer = fallback_note + extractive_answer
        used = "extractive (fallback)"

    return {
        "answer": answer,
        "provider": used,
        "sources": chunks,
    }


def active_provider() -> str:
    """Return a human-readable description of the currently active provider."""
    p = detect_provider()
    if p == "gguf":
        return f"Local GGUF model: {os.getenv('GGUF_MODEL_PATH')}"
    if p == "openai":
        model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
        base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        return f"OpenAI-compatible API ({model} @ {base})"
    return "Extractive mode (no LLM configured)"
