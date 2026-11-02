# RAG-Based AI Document Analyser 🔍📄

> **B.Tech Artificial Intelligence & Machine Learning Project**  
> A locally-runnable, production-quality document Q&A system using Retrieval-Augmented Generation.

---

## Problem Statement

Students and researchers frequently need to extract specific information from large collections of PDF reports, research papers, and text documents. Traditional keyword search is brittle and context-unaware. Large Language Models (LLMs) hallucinate facts they were not trained on. This project solves both problems by combining semantic search with document grounding — the system only answers from what is actually in your uploaded documents.

---

## Objectives

1. Enable question answering over user-uploaded PDF, TXT, and DOCX files.
2. Ground every answer in retrieved document passages with source citations.
3. Run fully offline on a student laptop using free, open-source models.
4. Provide an optional LLM integration (local GGUF or OpenAI-compatible API) for better answer quality.
5. Include document summarisation and key-point extraction.
6. Demonstrate core ML concepts: embeddings, vector similarity search, and RAG.

---

## Features

| Feature | Details |
|---|---|
| Multi-format ingestion | PDF (pypdf), TXT (UTF-8/latin-1), DOCX (python-docx) |
| Semantic chunking | Recursive character splitting with configurable size and overlap |
| Local embeddings | `sentence-transformers/all-MiniLM-L6-v2` — free, 22 MB, CPU-only |
| Vector search | FAISS CPU — exact cosine similarity, no server needed |
| Answer generation | Extractive (always works) · Local GGUF · OpenAI-compatible API |
| Source citations | Every answer shows filename, page number, chunk ID, similarity score |
| Low-evidence detection | Honest "not enough evidence" response instead of hallucinated answers |
| Document summary | TF-IDF extractive summarisation with map-reduce for long documents |
| Key-point extraction | Top informative phrases ranked by TF-IDF |
| Duplicate handling | Re-uploading a file replaces its index entries (no stale data) |
| Per-document removal | Remove a single document from the index without re-processing others |
| Professional UI | Streamlit dashboard with status bar, tabs, progress indicators, tooltips |

---

## RAG Workflow

```
User uploads document(s)
        │
        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Document Loader │───▶│  Text Chunker    │───▶│  Embedding Model│
│  PDF/TXT/DOCX   │    │  (overlapping    │    │  all-MiniLM-L6  │
│  → page dicts   │    │   chunks)        │    │  → float32 vecs │
└─────────────────┘    └──────────────────┘    └────────┬────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────┐
                                               │   FAISS Index   │
                                               │  (in-memory,    │
                                               │  optional disk) │
                                               └────────┬────────┘
User asks question                                       │
        │                                               │
        ▼                                               │
┌───────────────┐    ┌──────────────────┐              │
│ Query Encoder │───▶│  Similarity      │◀─────────────┘
│ (same model)  │    │  Search top-K    │
└───────────────┘    └────────┬─────────┘
                               │
                               ▼
                    ┌──────────────────┐    ┌──────────────────┐
                    │  Retrieved Chunks│───▶│  Answer Generator│
                    │  + metadata      │    │  LLM / extractive│
                    └──────────────────┘    └────────┬─────────┘
                                                      │
                                                      ▼
                                           Answer + Source Citations
```

**Key insight:** The embedding model and the answer-generating LLM are two completely different components:
- **Embedding model** → converts text to numbers for *search* (never generates words)
- **LLM** → reads retrieved passages and *writes* an answer (never used for search)

---

## Technology Stack

| Component | Library | Why |
|---|---|---|
| Web UI | Streamlit ≥ 1.32 | Zero-boilerplate Python web apps |
| Chunking | LangChain Text Splitters | Sentence-aware recursive splitting |
| Embeddings | sentence-transformers 2.7+ | Free, offline, fast on CPU |
| Vector DB | faiss-cpu 1.7+ | Millisecond exact similarity search |
| PDF parsing | pypdf 4.2+ | Pure Python, no system deps |
| DOCX parsing | python-docx 1.1+ | Full paragraph/heading access |
| LLM (optional) | llama-cpp-python / openai | Local GGUF or cloud API |
| Env config | python-dotenv | Secrets never in code |
| Numerics | numpy 1.26+ | Array operations |

---

## Folder Structure

```
RAG-Document-Analyser/
├── app.py                        # Streamlit UI (entry point)
├── requirements.txt              # All Python dependencies
├── .env.example                  # Credentials template (copy → .env)
├── README.md                     # This file
├── project_report_outline.md     # Academic report template
│
├── src/                          # All backend logic
│   ├── __init__.py
│   ├── document_loader.py        # PDF/TXT/DOCX → page dicts
│   ├── text_chunker.py           # Overlapping chunk splitting
│   ├── embeddings.py             # sentence-transformers wrapper
│   ├── vector_store.py           # FAISS index + persistence
│   ├── retriever.py              # Query → top-K chunks + tuning docs
│   ├── llm_service.py            # GGUF / OpenAI / extractive router
│   ├── rag_pipeline.py           # End-to-end orchestrator
│   └── summarizer.py             # TF-IDF extractive + map-reduce
│
└── tests/                        # Automated test suite
    ├── __init__.py
    ├── sample_docs/
    │   ├── renewable_energy.txt  # 37-line factual TXT document
    │   ├── ai_healthcare.docx    # 6-section DOCX document
    │   └── climate_change.pdf    # 3-page PDF with known figures
    ├── test_document_loader.py   # 12 unit tests
    ├── test_text_chunker.py      # 9 unit tests
    ├── test_rag_pipeline.py      # 20 unit/integration tests
    ├── test_retriever_and_llm.py # 38 component tests
    └── test_integration_scenarios.py  # 46 end-to-end scenario tests
```

---

## Prerequisites

- Windows 10 or 11 (the exact commands below are PowerShell)
- **Python 3.10, 3.11, or 3.12** — download from [python.org](https://www.python.org/downloads/)
- Git (optional) — for cloning

Verify your Python version:
```powershell
python --version
```

---

## Installation

### 1. Get the project

```powershell
# If you have Git:
git clone https://github.com/your-username/RAG-Document-Analyser.git
cd RAG-Document-Analyser

# Or: unzip the project folder and open a terminal inside it
```

### 2. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> **If you get an execution-policy error:**
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```
> Then run `.\venv\Scripts\Activate.ps1` again.

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~22 MB) automatically to `~/.cache/huggingface/`. This happens once.

---

## Configure an Optional LLM (for generated answers)

The application works **without any LLM** — it returns the most relevant document passages verbatim (extractive mode). To enable AI-generated answers:

```powershell
# Copy the template
Copy-Item .env.example .env
notepad .env
```

**Option A — Local GGUF model (free, 100% offline)**

1. Download a GGUF model:  
   [Mistral-7B-Instruct Q4_K_M (~4 GB)](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf)
2. Install the Python binding:  
   ```powershell
   pip install llama-cpp-python
   ```
3. Set in `.env`:  
   ```
   GGUF_MODEL_PATH=C:\models\mistral-7b-instruct-v0.2.Q4_K_M.gguf
   ```

**Option B — OpenAI-compatible API**

Set in `.env` (choose your provider):
```
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL_NAME=gpt-4o-mini
```

Works with: OpenAI, Groq (`https://api.groq.com/openai/v1`), Together AI, Ollama (`http://localhost:11434/v1`).

> API keys are **never** stored in code or displayed in the UI.

---

## Launch the Application

```powershell
# Ensure venv is active
.\venv\Scripts\Activate.ps1

streamlit run app.py
```

Open **http://localhost:8501** in your browser.

**Workflow:**
1. Upload one or more PDF/TXT/DOCX files using the file uploader
2. Click **⚡ Process documents** to build the vector index
3. Ask questions in the chat interface
4. View source citations in the **🔎 Sources** tab
5. Generate summary and key points in the **📋 Summary** tab

---

## Run the Tests

```powershell
# All 125 tests (unit + integration + scenario):
.\venv\Scripts\python.exe -m pytest tests/ -v

# Fast unit tests only (no embedding model needed for some):
.\venv\Scripts\python.exe -m pytest tests/test_document_loader.py tests/test_text_chunker.py -v

# Integration scenarios only:
.\venv\Scripts\python.exe -m pytest tests/test_integration_scenarios.py -v
```

No API keys or external services are required for any test.

---

## Retrieval Quality Tuning

| Parameter | Default | Lower value | Higher value |
|---|---|---|---|
| Chunk size | 500 chars | More precise, less context | More context, lower precision |
| Chunk overlap | 100 chars | Risk of split facts | Wastes index space |
| Top-K | 5 | Faster, focused | More coverage, higher LLM cost |

**Recommended settings by use case:**

| Use case | Chunk size | Overlap | Top-K |
|---|---|---|---|
| Short fact lookups | 300 | 75 | 3 |
| General Q&A (default) | 500 | 100 | 5 |
| Long-form reasoning | 800 | 150 | 7 |

To use an alternative embedding model (higher quality, slower):
```
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```
**Re-process all documents after changing the embedding model.**

---

## Sample Questions and Expected Behaviour

Using the provided sample documents:

| Question | Expected source | Expected behaviour |
|---|---|---|
| "What was the global solar cost per kWh in 2023?" | renewable_energy.txt | Cites the $0.057 figure |
| "How long does drug discovery take traditionally?" | ai_healthcare.docx | Cites 10-15 years figure |
| "What was the CO2 concentration in May 2023?" | climate_change.pdf p.3 | Cites 421 ppm |
| "How do I bake a chocolate cake?" | — | Low-evidence warning |
| "What is Alzheimer's disease treatment?" | — | Low-evidence (not in docs) |

---

## Known Limitations

| Limitation | Workaround |
|---|---|
| Scanned/image-only PDFs return no text | Use OCR (e.g. pytesseract) as a pre-processing step |
| Very large files (>50 MB) are blocked | Split large documents before uploading |
| FAISS index is in-memory only by default | Set `FAISS_INDEX_DIR=./faiss_store` in `.env` for persistence |
| Extractive answers are verbatim passages | Configure a GGUF or API LLM for synthesised answers |
| Sentence-transformer model downloads on first run | Requires internet on first launch only |
| No multi-user session isolation | For shared deployment, add per-session directories |
| DOCX images/tables are not extracted | Only paragraph text is used |

---

## Future Enhancements

- OCR support for scanned PDFs using `pytesseract`
- Streaming LLM responses in the chat UI
- Conversation memory across multiple questions
- FAISS IVF index for faster search on large collections (>100k chunks)
- Support for additional formats: Markdown, HTML, CSV
- Per-user session isolation for multi-user deployments
- Export answers and citations to PDF/DOCX
- Confidence calibration and answer quality scoring

---

## Running on VS Code

1. Open the project folder: **File → Open Folder**
2. Select the Python interpreter: `Ctrl+Shift+P` → "Python: Select Interpreter" → choose `venv`
3. Open a terminal: `` Ctrl+` `` → run `streamlit run app.py`
4. Install the "Pylance" and "Python" extensions for IntelliSense

---

*Made with Python, Streamlit, sentence-transformers, and FAISS.*  
*B.Tech AI/ML Project — All processing is local; no data is sent to external servers when using extractive mode.*
