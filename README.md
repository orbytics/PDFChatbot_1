# PDF Chatbot — RAG Pipeline

A local RAG (Retrieval-Augmented Generation) chatbot that answers natural-language questions about a single PDF document, with verbatim source citations. Built with LangChain, OpenAI, FAISS, and Streamlit.

## Quick Start

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd PDFChatbot_1

# 2. Create virtualenv and install dependencies
uv venv --python 3.12
uv pip install -r requirements.txt

# 3. Add your API key
echo "OPENAI_API_KEY=sk-..." > .env

# 4. Add your PDF
cp /path/to/your/document.pdf documents/source.pdf

# 5. Index the document
uv run python ingest.py

# 6. Launch the chat UI
uv run streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.12 (managed via `uv`) |
| [uv](https://docs.astral.sh/uv/) | any recent version |
| OpenAI API key | — |

---

## Setup

### 1. Install `uv` (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Create the virtual environment

```bash
uv venv --python 3.12
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 4. Create `.env`

`.env` contents:

```
OPENAI_API_KEY=sk-...
```

All other settings have working defaults (see [Configuration](#configuration)).

---

## Adding Your Document

Place a single PDF at `documents/source.pdf` (default path, configurable via `PDF_PATH` in `.env`).

```bash
cp /path/to/your-document.pdf documents/source.pdf
```

---

## Indexing (`ingest.py`)

Run once after adding or replacing the PDF. Re-running always rebuilds the index from scratch.

```bash
uv run python ingest.py
```

Expected output:
```
Loaded 10 pages from ./documents/source.pdf

Chunk count  : 41
Avg length   : 463 chars
Min / Max    : 39 / 800 chars
Embedding 41 chunks with text-embedding-3-small ...
Index saved to ./faiss_index/ (41 vectors)
```

The FAISS index is saved to `faiss_index/` (gitignored — not committed).

---

## Running the Chat UI (`app.py`)

```bash
uv run streamlit run app.py
```

Open http://localhost:8501. The sidebar shows the index status, chunk count, and active model/config.

**Answer format** — every response includes:
- **Answer** — grounded strictly in the document
- **Page Number(s)** — source page(s)
- **Exact Supporting Quote** — verbatim text from the retrieved chunk

If the document does not contain the answer, the response is `NO INFORMATION AVAILABLE`.

---

## Configuration

All settings can be overridden in `.env`. Defaults match `config.py`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `PDF_PATH` | `./documents/source.pdf` | Path to the source PDF |
| `FAISS_INDEX_DIR` | `./faiss_index` | Directory for the persisted FAISS index |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `CHUNK_SIZE` | `800` | Max characters per chunk |
| `CHUNK_OVERLAP` | `150` | Character overlap between chunks |
| `TOP_K` | `4` | Number of chunks retrieved per query |

**Changing models**: update `EMBEDDING_MODEL` or `LLM_MODEL` in `.env`, then re-run `ingest.py` (the FAISS index must be rebuilt when the embedding model changes).

---

## Project Structure

```
.
├── documents/          # Place source.pdf here (gitignored)
├── faiss_index/        # Generated index files — rebuilt by ingest.py (gitignored)
├── app.py              # Streamlit chat UI
├── ingest.py           # PDF loading, chunking, embedding, FAISS index build
├── rag_pipeline.py     # Retrieval + generation logic (reusable)
├── config.py           # Centralised config loaded from .env
├── requirements.txt    # Pinned dependencies
├── specs.md            # What/Why — full specification
├── implementation.md   # How/Milestones — build plan
└── CLAUDE.md           # AI-assisted development guardrails
```

---

## How It Works

```
PDF → load (PyPDFLoader) → chunk (header-aware + RecursiveCharacterTextSplitter)
    → embed (text-embedding-3-small) → FAISS index (persisted to disk)

Query → embed → similarity search (top-k chunks) → prompt (strict grounding rules)
      → gpt-4o-mini → parse JSON → verify quote → return answer + citation
```

The quote verification guard ensures the "Exact Supporting Quote" is a verbatim substring of a retrieved chunk. If the LLM paraphrases or fabricates a quote, the response is overridden to `NO INFORMATION AVAILABLE`.
