"""
app.py
──────
RAG-Based AI Document Analyser — Streamlit web interface.

Run:
    streamlit run app.py

All business logic lives in src/.  This file contains only UI code that
calls the pipeline and formats results for display.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the FIRST Streamlit call in the file)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Document Analyser",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_MAX_FILE_MB    = 50
_MAX_FILE_BYTES = _MAX_FILE_MB * 1024 * 1024
_ALLOWED_EXT    = {".pdf", ".txt", ".docx"}

_PROVIDER_LABELS = {
    "gguf":       ("🟢", "Local GGUF model (offline)"),
    "openai":     ("🔵", "OpenAI-compatible API"),
    "extractive": ("🟡", "Extractive mode — no LLM configured"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Minimal CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; }
    .stChatMessage blockquote {
        border-left: 3px solid #3b82d4;
        padding-left: 0.8rem;
        color: #374151;
    }
    .rag-status {
        padding: 8px 14px; border-radius: 8px;
        font-size: 0.88rem; font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .rag-ready   { background: #d1fae5; color: #065f46; }
    .rag-pending { background: #fef9c3; color: #713f12; }
    .rag-empty   { background: #f3f4f6; color: #6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def _init_session():
    """Set every session-state key to its default if missing."""
    defaults = {
        "chat_history":     [],    # list[dict]
        "indexed_docs":     {},    # {filename: {pages, chunks, size_kb, replaced}}
        "last_sources":     [],    # list[dict] — retrieved chunks from last query
        "last_summary":     {},    # {} or {summary, key_points, method, ...}
        "settings_changed": False,
        "index_built":      False,
        "_last_settings":   None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline factory  — cached PER settings tuple
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Loading embedding model (first run only)…")
def _get_pipeline(chunk_size: int, chunk_overlap: int, top_k: int):
    from src.rag_pipeline import RAGPipeline
    return RAGPipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap, top_k=top_k)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 ** 2:
        return f"{n / 1024:.1f} KB"
    return f"{n / 1024**2:.1f} MB"


def _validate_file(uf) -> str | None:
    ext = Path(uf.name).suffix.lower()
    if ext not in _ALLOWED_EXT:
        return f"Unsupported format '{ext}'. Allowed: PDF, TXT, DOCX."
    if uf.size > _MAX_FILE_BYTES:
        return f"Too large ({_fmt_size(uf.size)}). Max {_MAX_FILE_MB} MB."
    return None


def _index_ready() -> bool:
    return bool(st.session_state.index_built) and not bool(st.session_state.settings_changed)


def _provider_info() -> tuple[str, str]:
    from src.llm_service import detect_provider
    p = detect_provider()
    return _PROVIDER_LABELS.get(p, ("⚪", p))


def _reset_session():
    """Full reset — clear state AND pipeline cache."""
    st.session_state.chat_history     = []
    st.session_state.indexed_docs     = {}
    st.session_state.last_sources     = []
    st.session_state.last_summary     = {}
    st.session_state.settings_changed = False
    st.session_state.index_built      = False
    st.session_state._last_settings   = None
    st.cache_resource.clear()

# ─────────────────────────────────────────────────────────────────────────────
# ── SIDEBAR ──────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # ── Chunking ──────────────────────────────────────────────────────────────
    with st.expander("📐 Chunking parameters", expanded=True):
        st.caption(
            "Documents are split into overlapping *chunks* before embedding. "
            "Smaller chunks → more precise retrieval. "
            "Larger chunks → more context per result."
        )
        chunk_size = st.slider(
            "Chunk size (characters)",
            min_value=200, max_value=2000, value=500, step=50,
            help="Maximum characters per chunk.",
        )
        max_overlap = max(0, chunk_size - 50)
        chunk_overlap = st.slider(
            "Chunk overlap (characters)",
            min_value=0, max_value=min(400, max_overlap), value=100, step=25,
            help="Characters shared between adjacent chunks.",
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────
    with st.expander("🔍 Retrieval parameters", expanded=True):
        st.caption(
            "Top-K: how many document chunks are retrieved per question. "
            "More chunks = more coverage but slower."
        )
        top_k = st.slider(
            "Top-K chunks",
            min_value=1, max_value=10, value=5,
            help="Chunks retrieved per query.",
        )

    # Settings-change detection
    _cur = (chunk_size, chunk_overlap, top_k)
    _prev = st.session_state.get("_last_settings")
    if _prev is not None and _prev != _cur and st.session_state.index_built:
        st.session_state.settings_changed = True
    st.session_state["_last_settings"] = _cur

    st.divider()

    # ── LLM provider ──────────────────────────────────────────────────────────
    st.markdown("### 🤖 LLM Provider")
    p_icon, p_label = _provider_info()
    st.markdown(f"{p_icon} **{p_label}**")

    from src.llm_service import detect_provider as _dp
    if _dp() == "extractive":
        with st.expander("How to enable an LLM", expanded=False):
            st.markdown(
                """
**Option A — Local GGUF (free, offline)**
1. Download [Mistral-7B Q4](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)
2. `pip install llama-cpp-python`
3. Add to `.env`:
```
GGUF_MODEL_PATH=C:/models/mistral.gguf
```

**Option B — OpenAI / Groq / Ollama API**
```
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-4o-mini
```
> API keys are loaded from `.env` only — never entered in the UI.
                """
            )
    else:
        st.caption("API key loaded from `.env` — not shown here.")

    st.divider()

    # ── Session controls ──────────────────────────────────────────────────────
    st.markdown("### 🗂️ Session")
    if st.session_state.index_built:
        n_docs = len(st.session_state.indexed_docs)
        n_q    = len(st.session_state.chat_history) // 2
        st.info(f"📄 **{n_docs}** doc(s) indexed · 💬 **{n_q}** question(s)")

    if st.button("🗑️ Clear session & start over",
                 use_container_width=True,
                 help="Removes all documents, the index, and chat history."):
        _reset_session()
        st.rerun()

    st.divider()

    # ── RAG explainer ─────────────────────────────────────────────────────────
    with st.expander("ℹ️ How does RAG work?", expanded=False):
        st.markdown(
            """
**RAG** grounds AI answers in real documents instead of training-data memory.

**Pipeline:**
1. 📄 Load PDF / TXT / DOCX
2. ✂️ Split into overlapping chunks
3. 🔢 Embed each chunk with *all-MiniLM-L6-v2* (local, free)
4. 🗄️ Store vectors in FAISS (in-memory)
5. ❓ Embed the question the same way
6. 🔍 Retrieve the K nearest chunks by cosine similarity
7. 💬 LLM (or extractive fallback) writes an answer from those chunks only

> **Embedding model ≠ LLM.**
> Embeddings convert text → numbers for *search*.
> The LLM converts retrieved text → a coherent written answer.
            """
        )

# ─────────────────────────────────────────────────────────────────────────────
# Get (or create) the pipeline for the current settings
# ─────────────────────────────────────────────────────────────────────────────

pipeline = _get_pipeline(chunk_size, chunk_overlap, top_k)

# ─────────────────────────────────────────────────────────────────────────────
# ── HEADER ───────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <h1 style="margin-bottom:0.2rem">🔍 RAG-Based AI Document Analyser</h1>
    <p style="color:#6b7280; margin-top:0; font-size:1rem;">
    Upload documents → process → ask questions.
    Every answer is grounded in your content with numbered source citations.
    </p>
    """,
    unsafe_allow_html=True,
)

# Status bar
if not st.session_state.index_built:
    st.markdown(
        '<div class="rag-status rag-empty">'
        "⬜  No documents indexed yet — upload files and click "
        "<b>⚡ Process documents</b> to begin."
        "</div>",
        unsafe_allow_html=True,
    )
elif st.session_state.settings_changed:
    st.markdown(
        '<div class="rag-status rag-pending">'
        "🟡  Settings changed — click <b>⚡ Process documents</b> to rebuild the index."
        "</div>",
        unsafe_allow_html=True,
    )
else:
    n = len(st.session_state.indexed_docs)
    c = pipeline.store.total_chunks
    st.markdown(
        f'<div class="rag-status rag-ready">'
        f"🟢  Index ready — <b>{n}</b> document(s), <b>{c}</b> chunks."
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# ── TWO-COLUMN LAYOUT ────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

left_col, right_col = st.columns([3, 2], gap="large")

# ═════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — upload, process, chat
# ═════════════════════════════════════════════════════════════════════════════

with left_col:

    # ── STEP 1: Upload ────────────────────────────────────────────────────────
    st.markdown("### 📤 Step 1 — Upload documents")

    uploaded_files = st.file_uploader(
        "Choose PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help=f"Max {_MAX_FILE_MB} MB per file.",
    )

    if uploaded_files:
        valid_files, bad_files = [], []
        for uf in uploaded_files:
            err = _validate_file(uf)
            if err:
                bad_files.append((uf.name, err))
            else:
                valid_files.append(uf)

        for fname, emsg in bad_files:
            st.error(f"❌ **{fname}**: {emsg}")

        if valid_files:
            # Show a simple table of selected files
            col_h1, col_h2, col_h3 = st.columns([4, 1, 1])
            col_h1.caption("**File**")
            col_h2.caption("**Size**")
            col_h3.caption("**Status**")
            for uf in valid_files:
                already = uf.name in st.session_state.indexed_docs
                c1, c2, c3 = st.columns([4, 1, 1])
                c1.text(uf.name)
                c2.text(_fmt_size(uf.size))
                c3.success("indexed") if already else c3.warning("pending")

    # ── STEP 2: Process ───────────────────────────────────────────────────────
    st.markdown("### ⚙️ Step 2 — Process & build index")

    if st.session_state.settings_changed:
        st.warning(
            "⚠️ Settings changed after indexing. "
            "Click **Process documents** to rebuild with the new settings."
        )

    btn_disabled = not bool(uploaded_files)
    process_clicked = st.button(
        "⚡ Process documents",
        use_container_width=True,
        type="primary",
        disabled=btn_disabled,
        help="Embed all uploaded files and build the FAISS vector index.",
    )

    if process_clicked and uploaded_files:
        # Rebuild index from scratch if settings changed
        if st.session_state.settings_changed:
            pipeline.reset()
            st.session_state.indexed_docs     = {}
            st.session_state.last_sources     = []
            st.session_state.last_summary     = {}
            st.session_state.settings_changed = False

        to_process = [uf for uf in uploaded_files if _validate_file(uf) is None]
        if to_process:
            prog = st.progress(0, text="Starting…")
            msg  = st.empty()
            errs: list[str] = []

            for idx, uf in enumerate(to_process):
                pct = int(idx / len(to_process) * 100)
                prog.progress(pct, text=f"Embedding {uf.name}…")
                msg.info(f"⏳ Processing **{uf.name}**…")
                try:
                    raw = io.BytesIO(uf.read())
                    raw.name = uf.name
                    res = pipeline.ingest(raw, filename=uf.name)
                    st.session_state.indexed_docs[uf.name] = {
                        "pages":    res["pages_loaded"],
                        "chunks":   res["chunks_added"],
                        "size_kb":  round(uf.size / 1024, 1),
                        "replaced": res.get("replaced", False),
                    }
                    st.session_state.index_built      = True
                    st.session_state.settings_changed = False
                except ValueError as exc:
                    errs.append(f"**{uf.name}**: {exc}")
                except Exception as exc:
                    errs.append(f"**{uf.name}**: Unexpected error — {exc}")

            prog.progress(100, text="Done!")
            msg.empty()

            for e in errs:
                st.error(f"❌ {e}")

            if st.session_state.index_built:
                ok_count = len(to_process) - len(errs)
                st.success(
                    f"✅ {ok_count} file(s) processed — "
                    f"{pipeline.store.total_chunks} chunks in index."
                )
                st.rerun()

    # ── Indexed documents list ────────────────────────────────────────────────
    if st.session_state.indexed_docs:
        st.markdown("#### 📂 Indexed documents")
        for doc_name, meta in list(st.session_state.indexed_docs.items()):
            row_left, row_right = st.columns([6, 1])
            with row_left:
                tag = "♻️" if meta.get("replaced") else "✅"
                st.markdown(
                    f"{tag} **{doc_name}** — "
                    f"{meta['pages']} page(s), "
                    f"{meta['chunks']} chunks, "
                    f"{meta['size_kb']} KB"
                )
            with row_right:
                if st.button("✕", key=f"rm_{doc_name}",
                             help=f"Remove {doc_name} from index"):
                    pipeline.remove_source(doc_name)
                    del st.session_state.indexed_docs[doc_name]
                    st.session_state.last_sources = []
                    st.session_state.last_summary = {}
                    if not st.session_state.indexed_docs:
                        st.session_state.index_built = False
                    st.rerun()

    st.markdown("---")

    # ── STEP 3: Ask questions ─────────────────────────────────────────────────
    st.markdown("### 💬 Step 3 — Ask questions")

    if not _index_ready():
        if st.session_state.settings_changed:
            st.info("🟡 Settings changed. Click **⚡ Process documents** to rebuild the index.")
        else:
            st.info("📌 Upload and process at least one document first.")
    else:
        # Render chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    meta = msg.get("meta", {})
                    if meta.get("low_evidence"):
                        st.caption("⚠️ Low relevance — documents may not cover this topic.")
                    elif meta.get("provider"):
                        st.caption(
                            f"Provider: `{meta['provider']}` · "
                            f"{meta.get('n_sources', 0)} passage(s) retrieved"
                        )

        # Chat input — NOTE: do NOT give it a static key; Streamlit manages it
        user_question = st.chat_input("Ask anything about your documents…")

        if user_question:
            user_question = user_question.strip()
            if not user_question:
                st.stop()

            st.session_state.chat_history.append(
                {"role": "user", "content": user_question, "meta": {}}
            )
            with st.chat_message("user"):
                st.markdown(user_question)

            answer_text = ""
            result_meta: dict = {}
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching and generating answer…"):
                    try:
                        res = pipeline.query(
                            user_question,
                            top_k=top_k,
                            min_score=0.0,
                        )
                        answer_text = res["answer"]
                        st.session_state.last_sources = res.get("sources", [])
                        result_meta = {
                            "provider":     res.get("provider", ""),
                            "low_evidence": res.get("low_evidence", False),
                            "n_sources":    len(res.get("sources", [])),
                        }
                        st.markdown(answer_text)
                        if result_meta["low_evidence"]:
                            st.caption("⚠️ Low relevance — documents may not cover this topic.")
                        else:
                            st.caption(
                                f"Provider: `{result_meta['provider']}` · "
                                f"{result_meta['n_sources']} passage(s) retrieved"
                            )
                    except ValueError as exc:
                        answer_text = f"⚠️ {exc}"
                        st.warning(answer_text)
                    except Exception as exc:
                        answer_text = f"❌ Unexpected error: {exc}"
                        st.error(answer_text)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer_text, "meta": result_meta}
            )
            st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — tabs: Sources | Summary | Key Points | About
# ═════════════════════════════════════════════════════════════════════════════

with right_col:
    tab_src, tab_sum, tab_kp, tab_about = st.tabs(
        ["🔎 Sources", "📋 Summary", "💡 Key Points", "ℹ️ About"]
    )

    # ── Tab: Retrieved Sources ────────────────────────────────────────────────
    with tab_src:
        st.markdown("#### Retrieved source passages")
        st.caption(
            "The passages below were the most semantically similar to your last question. "
            "Similarity is cosine distance between embeddings."
        )
        if not st.session_state.last_sources:
            st.info("Source passages appear here after you ask a question.")
        else:
            for i, chunk in enumerate(st.session_state.last_sources, start=1):
                score      = chunk.get("score", 0.0)
                score_pct  = f"{score * 100:.1f}%"
                source     = chunk.get("source", "unknown")
                page       = chunk.get("page", "?")
                total_pg   = chunk.get("total_pages", "?")
                chunk_id   = chunk.get("chunk_id", "")
                text       = chunk.get("text", "")

                label = f"[{i}] {source}  —  p.{page}/{total_pg}  ({score_pct} match)"
                with st.expander(label, expanded=(i == 1)):
                    # Colour-coded score badge
                    if score >= 0.6:
                        badge = "🟢"
                    elif score >= 0.35:
                        badge = "🔵"
                    else:
                        badge = "🟡"
                    st.markdown(f"{badge} **Similarity:** {score_pct}  |  **Page:** {page} of {total_pg}")
                    st.markdown(f"> {text}")
                    st.caption(f"Chunk ID: `{chunk_id}`")

    # ── Tab: Summary ──────────────────────────────────────────────────────────
    with tab_sum:
        st.markdown("#### Document summary")
        st.caption(
            "Extractive summarisation using TF-IDF sentence scoring. "
            "For long documents, a map-reduce approach is used automatically."
        )
        if not _index_ready():
            st.info("Process documents first to enable summarisation.")
        else:
            options = ["All documents"] + list(pipeline.ingested_sources)
            sel = st.selectbox("Summarise:", options, key="sum_select")

            if st.button("📋 Generate summary", use_container_width=True, key="btn_sum"):
                filt = None if sel == "All documents" else sel
                with st.spinner("Generating summary…"):
                    try:
                        summ = pipeline.summarize(source_filter=filt)
                        st.session_state.last_summary = summ
                    except ValueError as exc:
                        st.warning(str(exc))
                        st.session_state.last_summary = {}
                    except Exception as exc:
                        st.error(f"Summarisation failed: {exc}")
                        st.session_state.last_summary = {}

            if st.session_state.last_summary:
                s = st.session_state.last_summary
                method = s.get("method", "direct")
                badge  = "🔵 map-reduce" if method == "map-reduce" else "🟢 direct"
                st.markdown(
                    f"**Source:** {s.get('source_filter','—')} · "
                    f"**Chunks used:** {s.get('chunks_used','?')} · "
                    f"**Method:** {badge}"
                )
                st.markdown("---")
                st.markdown(s.get("summary", ""))

    # ── Tab: Key Points ───────────────────────────────────────────────────────
    with tab_kp:
        st.markdown("#### Key points")
        st.caption(
            "The most informative phrases, ranked by TF-IDF relevance. "
            "Generate a summary first to populate this tab."
        )
        if not _index_ready():
            st.info("Process documents first.")
        elif not st.session_state.last_summary:
            st.info("Click **📋 Generate summary** in the Summary tab first.")
        else:
            kp = st.session_state.last_summary.get("key_points", [])
            if kp:
                for point in kp:
                    st.markdown(f"• {point}")
            else:
                st.info("No key points could be extracted.")

    # ── Tab: About ────────────────────────────────────────────────────────────
    with tab_about:
        st.markdown("#### About this application")
        st.markdown(
            """
**B.Tech AI/ML Project** — RAG-Based AI Document Analyser

---

**How it works**

| Step | Component | Library |
|---|---|---|
| Document loading | PDF / TXT / DOCX parser | pypdf, python-docx |
| Chunking | Recursive character splitter | LangChain Text Splitters |
| Embeddings | Dense vector encoder | sentence-transformers |
| Vector search | Cosine similarity index | FAISS CPU |
| Answer generation | Extractive / GGUF / API | llama-cpp / openai |
| UI | Web dashboard | Streamlit |

---

**Concept glossary**

- **Embedding** — A fixed-length numeric vector representing the *meaning* of a text chunk. Used for search only; never generates text.
- **LLM** — A separate model that *reads* retrieved passages and *writes* a coherent answer.
- **FAISS** — Facebook AI Similarity Search; finds the nearest embedding vectors in milliseconds.
- **Cosine similarity** — Score from 0 (unrelated) to 1 (identical meaning).
- **Extractive mode** — Returns the top retrieved passages verbatim when no LLM is configured.

---

**Security**
- API keys are in `.env` only — never entered into this UI.
- Uploaded files are processed in memory; nothing is written to disk.
- Document text is untrusted input — embedded instructions are ignored.

---

`streamlit run app.py` · `python -m pytest tests/ -v`
            """
        )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
fc1, fc2, fc3 = st.columns(3)
p_icon, p_label = _provider_info()
fc1.caption("🎓 B.Tech AI/ML Project — RAG Document Analyser")
fc2.caption("Embeddings: sentence-transformers/all-MiniLM-L6-v2 · FAISS CPU")
fc3.caption(f"LLM: {p_icon} {p_label}")
