# RAG Chatbot — Specification

## 1. Overview
A local **learning project** that lets a user ask natural-language questions about a single internal document (a 10-page PDF) and receive answers grounded in that content, with source citations. The goal is to build and understand a complete RAG pipeline end-to-end — not to handle production-scale document volumes.

## 2. Goals
- Learn and demonstrate the core RAG pipeline: load → chunk → embed → index → retrieve → generate.
- Allow a single user to query a single PDF document via a simple chat UI.
- Ground LLM responses strictly in retrieved document chunks (RAG pattern) to eliminate hallucination.
- Keep the system simple, local, and easy to run/extend (no cloud infra required beyond the OpenAI API).

## 3. Out of Scope (for v1)
- Multiple documents or document types — scope is **one PDF (10 pages)**. DOCX and HTML/Markdown (Confluence) loaders are not implemented in v1.
- Multi-user accounts, authentication, or access control.
- Automatic syncing with Confluence.
- Real-time document updates / incremental re-indexing on file change detection.
- Conversation persistence across sessions (beyond in-memory chat history).
- Deployment to a hosted environment.
- Performance/scale optimization — this is a learning project, not a production system.

## 4. Tech Stack
| Component        | Choice                              |
|-------------------|--------------------------------------|
| Language          | Python 3.12 (managed via `uv`, dev: macOS / Mac mini) |
| Package manager   | `uv` (handles virtualenv creation and dependency installation) |
| Orchestration     | LangChain                           |
| LLM (generation)  | OpenAI API (`gpt-4o-mini`) |
| Embeddings        | OpenAI embeddings API (`text-embedding-3-small`) |
| Vector store      | FAISS (`faiss-cpu`, local, persisted to disk via LangChain's `FAISS.save_local` / `load_local`) |
| UI                | Streamlit                           |
| Document loaders  | LangChain `PyPDFLoader` for PDF only |

> **Note on Python version**: The project targets Python 3.12 (3.13 also acceptable) rather than the newly-released 3.14, since `faiss-cpu` and several LangChain dependencies have well-tested prebuilt wheels for these versions. `uv` will be used to pin and manage this Python version and the virtual environment, so no manual pyenv setup is required — `uv venv --python 3.12` and `uv pip install -r requirements.txt` (or `uv sync` if using a `pyproject.toml`) should work directly on the Mac mini.

## 5. System Architecture

```
[ Single PDF Document ]
   (10 pages)
        │
        ▼
[ Ingestion Script ]
  - Load documents
  - Split into chunks
  - Generate embeddings
  - Build & persist FAISS index
        │
        ▼
[ Vector Store (FAISS, local index files) ]
        │
        ▼
[ Streamlit App ]
  - User enters question
  - Retrieve top-k relevant chunks
  - Build prompt (context + question)
  - Call OpenAI LLM
  - Display answer + source citations
```

## 6. Document Ingestion
- **Input**: A single PDF file (10 pages) placed in `./documents/` (e.g., `./documents/source.pdf`).
- **Process** (run via a standalone Python script, e.g., `ingest.py`):
  1. Load the PDF via `PyPDFLoader`, producing one `Document` per page with `metadata["source"]` (filename) and `metadata["page"]`.
  2. Split documents into chunks using a **header-aware recursive splitter**: first split on document structure (headings, "Task Statement" / section titles, then paragraphs/bullets), then apply a recursive character split within oversized sections. Default **chunk size: 800 characters, overlap: 150 characters**.
  3. Generate embeddings for each chunk via the OpenAI embeddings API (`text-embedding-3-small`).
  4. Build a FAISS index from the chunk embeddings, storing chunk text + metadata (source filename, page number, and section/heading title if available) in LangChain's accompanying docstore.
  5. Persist the FAISS index and docstore to disk (e.g., `./faiss_index/`) via `FAISS.save_local`.
- **Re-running ingestion**: Re-running the script re-processes the PDF and rebuilds the FAISS index from scratch via `FAISS.save_local` (overwriting `./faiss_index/`). v1 does not need deduplication or incremental update logic.
- **Scale note**: A 10-page PDF at ~800 chars/chunk yields roughly 30–50 chunks — small enough to embed and index in seconds, which keeps the iteration loop fast for learning.

### Chunking & Embedding Rationale
Based on a review of a representative sample document (a 10-page certification exam guide with dense, hierarchical structure — domains, task statements, and "Knowledge of" / "Skills in" bullet lists):
- **Chunk size 800 / overlap 150** keeps most individual "Knowledge of" or "Skills in" blocks intact in a single chunk (each typically 300–700 characters), while overlap preserves continuity across list items that span a chunk boundary. This avoids fragmenting a single bullet point mid-sentence.
- **Header-aware splitting first** ensures a chunk doesn't straddle two unrelated Task Statements, which would mix context from different topics and hurt retrieval precision.
- **`text-embedding-3-small`** is used as a cost-effective default suitable for a personal prototype. The source material does contain closely related, easily-confused technical terms (e.g., similarly named tools, "hooks vs. prompt-based enforcement"), so if retrieval quality on these near-duplicate concepts proves insufficient, `text-embedding-3-large` is a drop-in upgrade (re-run `ingest.py` after changing `EMBEDDING_MODEL`, since the FAISS index must be rebuilt with matching embedding dimensions).

### Vector Store: FAISS vs. Chroma
FAISS was chosen per your preference. For this single-user, file-based prototype it's a good fit:
- LangChain's `FAISS` vector store wrapper bundles an in-memory docstore that holds chunk text and metadata alongside the index, so source/page attribution works the same as it would with Chroma.
- `FAISS.save_local(path)` / `FAISS.load_local(path)` persist both the index and docstore to a local directory — no separate database process needed.
- Tradeoff vs. Chroma: FAISS has no native support for deleting/updating individual vectors or filtering by metadata at query time; since v1 always does a full rebuild and doesn't need metadata-filtered search, this isn't a limitation here. If those features become important later (see Future Considerations), Chroma remains an easy swap since retrieval logic is isolated in `rag_pipeline.py`.

## 7. Retrieval & Generation
- On each user query:
  1. Embed the user's question using the same embedding model.
  2. Perform a similarity search against the FAISS index to retrieve the top-k chunks (default k=4, configurable).
  3. Construct a prompt containing: a strict grounding system instruction (below), the retrieved chunks as context (each tagged with its source filename and page number), and the user's question.
  4. Send the prompt to the OpenAI chat completion model (`gpt-4o-mini`).
  5. Return the generated answer along with the list of source documents/chunks used.

### Strict Grounding Rules (System Instruction)
The retrieved document context is the model's ONLY source of truth. The system prompt sent to the LLM must enforce:

1. Answer ONLY from the retrieved document context.
2. Do NOT use prior knowledge, model training data, assumptions, inference, reasoning beyond the provided text, or external sources.
3. If the answer is not explicitly stated in the retrieved context, respond with exactly: `NO INFORMATION AVAILABLE`
4. If only part of the answer is available, respond with exactly: `NO INFORMATION AVAILABLE`
5. Do NOT paraphrase critical facts.
6. Do NOT invent information.
7. Do NOT guess.
8. Do NOT summarize missing information.
9. Do NOT provide recommendations unless explicitly written in the document.
10. Preserve terminology exactly as written in the source whenever possible.
11. If retrieved chunks do not contain enough information to answer with complete confidence, respond with exactly: `NO INFORMATION AVAILABLE`

### Required Answer Format
Every successful answer (i.e., not `NO INFORMATION AVAILABLE`) must include:
- **Answer**
- **Page Number(s)**
- **Exact Supporting Quote** — copied verbatim from the retrieved context

If multiple pages contain supporting information:
- Cite all relevant pages.
- Include the shortest quote that fully supports the answer.

If the response is `NO INFORMATION AVAILABLE`, that exact string is returned with no additional fields.

> **Note on enforcement reliability**: `gpt-4o-mini` may not always produce a truly verbatim quote or the exact literal string `NO INFORMATION AVAILABLE` (it may paraphrase slightly) purely through prompt instructions. The pipeline should include a verification step that checks the returned "Exact Supporting Quote" actually appears (e.g., via substring match) in the retrieved chunk text for the cited page(s). If verification fails, the pipeline should treat the response as `NO INFORMATION AVAILABLE` rather than surfacing an unverified quote. This check is a deterministic guard on top of the prompt-based instructions, not a replacement for them.

## 8. User Interface (Streamlit)
- Single-page chat interface.
- Text input box for the user's question.
- Display of conversation history (question/answer pairs) for the current session (in-memory only).
- Each answer is rendered per the Required Answer Format in §7: **Answer**, **Page Number(s)**, and **Exact Supporting Quote** (or the literal `NO INFORMATION AVAILABLE` if the context is insufficient).
- A "Sources" section/expander lists the source filenames and page numbers of the retrieved chunks used to generate the answer.
- A sidebar showing basic status info (e.g., source PDF filename, number of chunks indexed, vector store location).

## 9. Configuration
A `.env` file (or `config.py`) should hold:
- `OPENAI_API_KEY`
- `PDF_PATH` (default: `./documents/source.pdf`) — path to the single source PDF
- `FAISS_INDEX_DIR` (default: `./faiss_index`)
- `EMBEDDING_MODEL` (default: `text-embedding-3-small`)
- `LLM_MODEL` (default: `gpt-4o-mini`)
- `CHUNK_SIZE` (default: `800`)
- `CHUNK_OVERLAP` (default: `150`)
- `TOP_K` (default: `4`)

## 10. Project Structure
```
rag-chatbot/
├── documents/          # contains the single source PDF (source.pdf)
├── faiss_index/          # persisted FAISS index + docstore (generated)
├── ingest.py             # document ingestion script
├── app.py                # Streamlit chat application
├── rag_pipeline.py       # shared retrieval + generation logic
├── config.py             # configuration loading
├── pyproject.toml        # project metadata + dependencies (uv-managed)
├── uv.lock                # locked dependency versions (generated by uv)
├── requirements.txt      # dependency reference (kept in sync with pyproject.toml)
├── .env                  # API keys and config (not committed)
├── .gitignore            # excludes .env, faiss_index/, documents/*, etc.
├── README.md             # setup & usage instructions (Milestone 8)
├── CLAUDE.md             # project guardrails for AI-assisted development
├── specs.md
└── implementation.md
```

## 11. Non-Functional Requirements
- **Simplicity**: Minimal dependencies; runnable on Mac mini via `uv venv --python 3.12`, `uv pip install -r requirements.txt` (or `uv sync`), then `uv run python ingest.py` and `uv run streamlit run app.py`.
- **Privacy**: Documents and vector store remain local; only query text, retrieved chunk text, and final prompts are sent to OpenAI's API.
- **Performance**: Scoped to a single ~10-page PDF (~30–50 chunks); not optimized or tested for larger document sets — this is a learning project.
- **Extensibility**: Architecture should allow swapping the vector store, embedding model, or LLM provider with minimal changes (isolated in `config.py` and `rag_pipeline.py`).

## 12. Future Considerations (not in v1)
- Support for additional document types (DOCX, HTML/Markdown, Confluence exports).
- Support for multiple documents / larger document sets (folder-based ingestion).
- Confluence API auto-sync for live document updates.
- Incremental re-indexing (detect new/changed/deleted files).
- Conversation history persistence (e.g., SQLite).
- Multi-user support and authentication.
- Switchable LLM/embedding providers (local models via Ollama, etc.).
- Source document preview/highlighting in the UI.