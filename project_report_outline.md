# Project Report Outline
## RAG-Based AI Document Analyser

**B.Tech — Artificial Intelligence & Machine Learning**  
**Subject: Major Project / Final Year Project**

---

## Title Page

- Project title: RAG-Based AI Document Analyser
- Student name(s) and roll number(s)
- Department and institution
- Supervisor name
- Academic year
- Date of submission

---

## Declaration

*I/We hereby declare that this project is our original work and has not been submitted to any other institution.*

---

## Certificate

*(Supervisor's certificate of authenticity)*

---

## Acknowledgements

*(Thank supervisor, department, and contributors)*

---

## Abstract (150–200 words)

Retrieval-Augmented Generation (RAG) is a framework that grounds language model outputs in external documents, reducing hallucination and improving factual accuracy. This project implements a complete RAG pipeline as a locally-runnable web application. Users upload PDF, TXT, and DOCX documents; the system extracts text, splits it into overlapping chunks, embeds each chunk using a free sentence-transformers model, and indexes the vectors in a FAISS store. When a user asks a question, the question is embedded identically and the most semantically similar chunks are retrieved. An optional LLM (local GGUF or OpenAI-compatible API) synthesises a grounded answer; without an LLM, the top retrieved passages are returned verbatim with source citations. The system explicitly detects low-relevance retrievals and reports insufficient evidence rather than hallucinating. All processing runs locally on a standard student laptop. The application includes a Streamlit dashboard, document summarisation, key-point extraction, and 125 automated tests covering unit, component, and end-to-end scenarios.

**Keywords:** RAG, LLM, sentence-transformers, FAISS, Streamlit, document Q&A, semantic search, NLP

---

## Table of Contents

1. Introduction
2. Problem Statement
3. Objectives
4. Literature Review
5. Methodology
6. System Architecture
7. Implementation
8. Testing
9. Results and Discussion
10. Limitations
11. Conclusion and Future Work
12. References
13. Appendices

---

## 1. Introduction (2–3 pages)

- The information retrieval challenge in the age of large documents
- How language models transformed NLP
- The hallucination problem and why grounding is essential
- Overview of Retrieval-Augmented Generation
- Motivation for this project
- Scope and organisation of the report

---

## 2. Problem Statement (1 page)

- Students/researchers cannot efficiently query large document collections
- Keyword search fails on semantically equivalent phrasings
- LLMs generate plausible but incorrect facts from training data
- Existing commercial tools require cloud APIs and share data externally
- **Need:** A local, grounded, citation-producing Q&A system for personal documents

---

## 3. Objectives (0.5 page)

1. Implement a complete local RAG pipeline (embed → index → retrieve → answer)
2. Support PDF, TXT, and DOCX with metadata-preserving chunking
3. Use only free, locally-runnable models for embeddings and optionally for generation
4. Produce source citations with document name, page number, and similarity score
5. Detect and communicate insufficient evidence honestly
6. Build a professional, user-friendly Streamlit interface
7. Validate correctness with 125 automated tests

---

## 4. Literature Review (3–4 pages)

### 4.1 Dense Retrieval
- Karpukhin et al. (2020) — Dense Passage Retrieval (DPR)
- Bi-encoder vs cross-encoder architectures
- Sentence-BERT (Reimers & Gurevych, 2019)

### 4.2 Retrieval-Augmented Generation
- Lewis et al. (2020) — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- REALM, RAG-Sequence vs RAG-Token variants
- Limitations of generative-only models (hallucination)

### 4.3 Vector Databases
- FAISS (Johnson et al., 2017) — Billion-scale similarity search
- IndexFlatIP vs IVF vs HNSW trade-offs
- Alternatives: Chroma, Pinecone, Weaviate

### 4.4 Document Processing
- Recursive character splitting in LangChain
- Chunk size vs retrieval precision trade-offs
- Metadata preservation for citations

### 4.5 Open-Source LLMs
- Llama 2, Mistral 7B, Phi-2
- GGUF quantisation for CPU inference (llama.cpp)
- Extractive vs abstractive answer modes

---

## 5. Methodology (2–3 pages)

### 5.1 System Design Philosophy
- Modular, testable components
- Provider-independence (GGUF / OpenAI / extractive)
- No data sent externally in default mode

### 5.2 Document Processing Pipeline
- Format detection by file extension
- PDF text extraction with pypdf (page-level)
- DOCX paragraph grouping (30 paragraphs per virtual page)
- TXT encoding detection (UTF-8 → latin-1 fallback)
- Empty document detection and error reporting

### 5.3 Chunking Strategy
- RecursiveCharacterTextSplitter: paragraphs → sentences → words
- Default: chunk_size=500, overlap=100 (~20%)
- Global chunk indexing for unique IDs
- Metadata: source filename, page number, total_pages, chunk_id

### 5.4 Embedding Model
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- 384-dimensional unit-normalised vectors
- Cosine similarity ≡ inner product on normalised vectors
- Module-level singleton to avoid repeated model loading

### 5.5 FAISS Indexing
- IndexFlatIP (exact inner-product search)
- Suitable for up to ~100k chunks on a student laptop
- Parallel metadata list (chunk dicts) maintains index→metadata mapping
- Optional disk persistence via `faiss.write_index` + pickle

### 5.6 Retrieval and Evidence Filtering
- Query encoded with the same model used for indexing
- Top-K cosine similarity search
- Low-evidence detection: if ALL retrieved chunks < 0.25 similarity
- `min_score` parameter for configurable thresholding

### 5.7 Answer Generation
- Priority: GGUF local model → OpenAI API → extractive fallback
- Prompt engineering: context + citation instruction + honesty directive
- Graceful degradation: LLM failure → extractive with warning

### 5.8 Summarisation
- TF-IDF sentence scoring (no external dependencies)
- Map-reduce for documents with >30 chunks
- Key-point extraction: top phrases by TF-IDF, truncated to 120 chars

---

## 6. System Architecture (2 pages)

### 6.1 Component Diagram

```
┌──────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                  │
│  Upload │ Process │ Chat │ Sources │ Summary │ Key Points │
└──────────────────────┬───────────────────────────────────┘
                        │ calls
                        ▼
┌──────────────────────────────────────────────────────────┐
│                  RAGPipeline (orchestrator)               │
│  ingest() │ query() │ summarize() │ remove_source()       │
└────┬─────────────┬──────────────┬────────────────────────┘
     │             │              │
     ▼             ▼              ▼
document_  text_      embeddings  vector_   retriever  llm_      summarizer
loader     chunker                store                service
```

### 6.2 Data Flow

- **Ingestion:** File → pages → chunks → embeddings → FAISS + metadata
- **Query:** Question → embedding → FAISS search → chunks → LLM → answer
- **Session state:** Streamlit session dict isolates per-browser-tab state

### 6.3 Security Considerations
- No API keys in source code — `.env` file only
- Uploaded files processed in memory (BytesIO), not written to disk
- Document text treated as untrusted input (not executed as code)
- Prompt injection: document text is user context, not system directive

---

## 7. Implementation (3–4 pages)

*(Show key code snippets with explanations)*

### 7.1 Document Loader
*(Show _load_pdf, _load_txt, _load_docx — discuss metadata preservation)*

### 7.2 Text Chunker
*(Show chunk_pages — discuss global_index and chunk_id format)*

### 7.3 Embeddings
*(Show embed_texts — discuss normalisation and batch processing)*

### 7.4 VectorStore
*(Show add_chunks, search — discuss FAISS IndexFlatIP)*

### 7.5 RAGPipeline
*(Show ingest, query, _rebuild_without — discuss duplicate handling)*

### 7.6 LLM Service
*(Show detect_provider, _build_prompt, _answer_extractive)*

### 7.7 Streamlit UI
*(Show session state design, two-step upload/process flow, tab layout)*

---

## 8. Testing (2 pages)

### 8.1 Test Strategy

| Level | File | Count | Coverage |
|---|---|---|---|
| Unit | test_document_loader.py | 12 | Format loading, empty docs, metadata |
| Unit | test_text_chunker.py | 9 | Chunk size, overlap, unique IDs |
| Component | test_rag_pipeline.py | 20 | Ingest, query, summarize, reset |
| Component | test_retriever_and_llm.py | 38 | Retrieval, LLM routing, citations |
| Integration | test_integration_scenarios.py | 46 | 10 full end-to-end scenarios |
| **Total** | | **125** | |

### 8.2 Key Test Scenarios
1. TXT file upload and metadata preservation
2. DOCX file loading with paragraph grouping
3. PDF file loading with 3-page document
4. Multi-document upload and cross-document retrieval
5. Factual question with answer present in documents
6. Source citation correctness (filename, page, score)
7. Question with no relevant evidence (low-evidence detection)
8. Document summarisation and key-point extraction
9. Empty and unsupported file type rejection
10. Session clear and re-index with different documents

### 8.3 Test Results

```
125 passed, 0 failed, 0 warnings
Platform: Windows 10, Python 3.12.10
Runtime: ~55 seconds (embedding model included)
```

---

## 9. Results and Discussion (1–2 pages)

### 9.1 Retrieval Quality
- Questions with strong keyword overlap: cosine similarity 0.6–0.9
- Questions requiring semantic understanding: 0.35–0.6
- Completely unrelated questions: < 0.25 (correctly flagged)

### 9.2 Sample Interaction

**Question:** "What was the cost of utility-scale solar per kWh in 2023?"  
**Retrieved source:** renewable_energy.txt, Page 1, similarity 0.81  
**Passage:** "The global average cost of utility-scale solar fell to $0.057 per kilowatt-hour (kWh) in 2023, making it the cheapest electricity source in history."  
**Answer (extractive):** Returns the passage verbatim with citation.

### 9.3 Performance
- Embedding model load: ~3 seconds (first run only)
- Document processing: ~0.5 seconds per 1000 words
- Query response: <500 ms (retrieving 5 chunks from 100-chunk index)

---

## 10. Limitations (0.5 page)

- Scanned PDFs require OCR pre-processing
- No conversation memory across multiple questions
- FAISS flat index does not scale beyond ~100k chunks efficiently
- Extractive answers are verbatim; an LLM improves fluency
- Single-user session design (no concurrent user isolation)

---

## 11. Conclusion and Future Work (0.5 page)

This project demonstrates a complete, functional, locally-runnable RAG pipeline suitable for document question answering. The system correctly grounds answers in retrieved passages, detects insufficient evidence, and provides source citations. All 125 automated tests pass on a standard Windows laptop without any cloud API access.

**Future directions:** OCR integration, streaming LLM responses, conversation memory, FAISS IVF indexing, and multi-user deployment.

---

## 12. References

1. Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
2. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019.
3. Johnson, J. et al. (2017). *Billion-scale similarity search with GPUs*. IEEE Transactions.
4. Karpukhin, V. et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP 2020.
5. Touvron, H. et al. (2023). *Llama 2: Open Foundation and Fine-Tuned Chat Models*. arXiv 2307.09288.
6. Jiang, A. et al. (2023). *Mistral 7B*. arXiv 2310.06825.

---

## 13. Appendices

### A. Full Folder Tree
*(Paste the project folder tree output)*

### B. Requirements.txt
*(Paste the file)*

### C. Environment Configuration
*(Paste .env.example)*

### D. Sample Test Run Output
*(Paste `pytest tests/ -v` output)*

### E. UI Screenshots
*(Attach screenshots of: file upload, question answering, source citations, summary)*
