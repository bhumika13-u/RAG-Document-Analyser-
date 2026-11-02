"""
document_loader.py
──────────────────
Reads PDF, TXT, and DOCX files and returns a list of page-level dicts:

    {
        "text":        str,   # readable text of this page/section
        "source":      str,   # original filename
        "page":        int,   # 1-based page number (best-effort)
        "total_pages": int,   # total pages in the document
    }

Supported formats
-----------------
- .pdf   via pypdf
- .txt   read as UTF-8 (fallback latin-1)
- .docx  via python-docx (paragraph-level grouping)
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

# ── third-party ──────────────────────────────────────────────────────────────
try:
    from pypdf import PdfReader
except ImportError as exc:  # pragma: no cover
    raise ImportError("pypdf is required: pip install pypdf") from exc

try:
    import docx  # python-docx
except ImportError as exc:  # pragma: no cover
    raise ImportError("python-docx is required: pip install python-docx") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


def load_document(file_source: str | Path | io.IOBase, filename: str = "") -> list[dict[str, Any]]:
    """
    Load a single document and return a list of page dicts.

    Parameters
    ----------
    file_source : str | Path | IO
        Either a filesystem path or an in-memory file-like object
        (e.g. from Streamlit's ``st.file_uploader``).
    filename : str
        Original filename — required when ``file_source`` is a file-like
        object so the loader can determine the format.

    Returns
    -------
    list[dict]
        List of page dicts (see module docstring).

    Raises
    ------
    ValueError
        On unsupported format or empty document.
    """
    if isinstance(file_source, (str, Path)):
        path = Path(file_source)
        ext = path.suffix.lower()
        name = filename or path.name
    else:
        # file-like object from Streamlit
        name = filename or getattr(file_source, "name", "unknown")
        ext = Path(name).suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{ext}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if ext == ".pdf":
        pages = _load_pdf(file_source, name)
    elif ext == ".txt":
        pages = _load_txt(file_source, name)
    elif ext == ".docx":
        pages = _load_docx(file_source, name)
    else:
        raise ValueError(f"Unhandled extension: {ext}")  # should not reach here

    if not pages or all(not p["text"].strip() for p in pages):
        raise ValueError(
            f"Document '{name}' appears to be empty or contains no readable text. "
            "Please check that it is not a scanned image-only PDF."
        )

    return pages


def load_documents(
    file_sources: list[str | Path | io.IOBase],
    filenames: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Convenience wrapper to load multiple documents.
    Returns a flat list of all page dicts.
    """
    filenames = filenames or [""] * len(file_sources)
    pages: list[dict[str, Any]] = []
    for src, name in zip(file_sources, filenames):
        pages.extend(load_document(src, name))
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# Format-specific loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_pdf(src: Any, name: str) -> list[dict[str, Any]]:
    reader = PdfReader(src)
    total = len(reader.pages)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text, "source": name, "page": i, "total_pages": total})
    return pages


def _load_txt(src: Any, name: str) -> list[dict[str, Any]]:
    if isinstance(src, (str, Path)):
        raw = Path(src).read_bytes()
    else:
        raw = src.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    return [{"text": text, "source": name, "page": 1, "total_pages": 1}]


def _load_docx(src: Any, name: str) -> list[dict[str, Any]]:
    if isinstance(src, (str, Path)):
        doc = docx.Document(str(src))
    else:
        doc = docx.Document(src)

    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    if not paragraphs:
        return []

    # Group every 30 paragraphs into a "page" so we preserve some structure
    PARA_PER_PAGE = 30
    groups: list[list[str]] = []
    for i in range(0, len(paragraphs), PARA_PER_PAGE):
        groups.append(paragraphs[i : i + PARA_PER_PAGE])

    total = len(groups)
    return [
        {
            "text": "\n".join(group),
            "source": name,
            "page": idx + 1,
            "total_pages": total,
        }
        for idx, group in enumerate(groups)
    ]
