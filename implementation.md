# RAG Chatbot — Implementation Plan

This document breaks down the build into ordered milestones. Each milestone produces a working, testable increment. Follow them in order — later milestones depend on earlier ones. Refers to `specs.md` for the *why/what* behind each decision.

## Workflow Rules
- **Approval gate**: Do not begin the next milestone until the user has reviewed the current milestone's work and given explicit approval to proceed.
- **Branching**: Each milestone is developed on its own branch, created from `main` (or the previous approved branch, per the user's preference at the time). Suggested naming: `milestone-0-setup`, `milestone-1-config`, `milestone-2-document-loading`, etc.
- **Pushing**: The user pushes each milestone's branch to GitHub themselves once the milestone is complete and the "Done when" criteria are met. Merging into `main` happens after approval.

---

## Milestone 0: Project Setup
**Goal**: A working Python environment and project skeleton, ready for dependencies.

1. Create project folder `rag-chatbot/` and initialize with `uv init` (or manually create `pyproject.toml`).
2. Pin the interpreter: `uv venv --python 3.12`.
3. Create the folder structure from specs.md §10:
   - `documents/` (place the 10-page sample PDF here as `source.pdf`)
   - `faiss_index/` (empty, will be generated)
   - empty placeholder files: `config.py`, `ingest.py`, `rag_pipeline.py`, `app.py`
4. Create `requirements.txt` (already provided) and install: `uv pip install -r requirements.txt`.
5. Create `.env` with `OPENAI_API_KEY=...`. Add the provided `.gitignore` (covers `.env`, `faiss_index/`, `documents/*`, Python/uv artifacts) — your internal documents likely shouldn't be pushed to GitHub, especially if the repo is public.
6. Initialize git: `git init`, create the GitHub repo, add the remote, and make an initial commit (`git add .`, `git commit -m "Project setup"`).

**Branch**: `git checkout -b milestone-0-setup`

**Done when**: `uv run python -c "import langchain, faiss, streamlit, openai"` runs without errors.

> **Dependency fix (actual)**: `requirements.txt` pinned `langchain-text-splitters==0.3.5`, but `langchain==0.3.18` requires `>=0.3.6`. Bumped to `0.3.11` (the version resolved by `uv` for this combination). Keep this in mind if upgrading LangChain in future.

**Push**: commit (`git add . && git commit -m "Milestone 0: project setup"`), then push the branch to GitHub.

**⏸ Wait for approval before starting Milestone 1.**

---

## Milestone 1: Configuration Module (`config.py`)
**Goal**: Centralized, typed access to all settings from specs.md §9.

1. Load environment variables from `.env` using `python-dotenv`.
2. Define constants/variables for: `OPENAI_API_KEY`, `PDF_PATH`, `FAISS_INDEX_DIR`, `EMBEDDING_MODEL`, `LLM_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, each with the defaults specified in specs.md.
3. Add a basic validation check (e.g., raise a clear error at import time if `OPENAI_API_KEY` is missing).

**Branch**: `git checkout -b milestone-1-config`

**Done when**: `uv run python -c "import config; print(config.EMBEDDING_MODEL)"` prints `text-embedding-3-small`.

**Push**: commit (`git add . && git commit -m "Milestone 1: configuration module"`), then push the branch to GitHub.

**⏸ Wait for approval before starting Milestone 2.**

---

## Milestone 2: Document Loading
**Goal**: Load the single source PDF into LangChain `Document` objects with page-level metadata.

1. Place the 10-page sample PDF (the exam-guide PDF) at `./documents/source.pdf`.
2. In `ingest.py`, write a `load_documents(pdf_path)` function that:
   - Loads the PDF via `PyPDFLoader(pdf_path)`.
   - Returns a list of `Document` objects, one per page, each with `metadata["source"]` set to the filename and `metadata["page"]` set to the page number.
3. Add a `print(f"Loaded {len(docs)} pages from {pdf_path}")` for sanity checking.

**Branch**: `git checkout -b milestone-2-document-loading`

**Done when**: Running the loader on `documents/source.pdf` prints `Loaded 10 pages from ...` and you can inspect a sample `Document`'s `page_content` and `metadata`.

**Push**: commit (`git add . && git commit -m "Milestone 2: document loading"`), then push the branch to GitHub.

**⏸ Wait for approval before starting Milestone 3.**

---

## Milestone 3: Chunking
**Goal**: Implement the header-aware recursive splitter from specs.md §6.

1. In `ingest.py`, write a `split_documents(docs)` function that:
   - First splits on markdown/heading-like patterns (e.g., using `MarkdownHeaderTextSplitter` or a custom regex pass for "Task Statement"/heading lines) to produce section-level `Document`s, preserving the heading text in `metadata["section"]`.
   - Then applies `RecursiveCharacterTextSplitter` with `chunk_size=config.CHUNK_SIZE`, `chunk_overlap=config.CHUNK_OVERLAP` within any section still larger than `CHUNK_SIZE`.
   - Returns the final list of chunked `Document`s with metadata intact (`source`, `page`, `section`).
2. Sanity-check on the sample PDF: print chunk count, average chunk length, and a couple of sample chunks to confirm sections aren't being split mid-bullet.

**Branch**: `git checkout -b milestone-3-chunking`

**Actual implementation notes**:
- The header regex (`Task\s+Statement\s+[\d.]+\s*:[^\n]*`) must run **before** newline normalization, because newlines are the only reliable boundary between a section title and its body. A two-phase normalization approach was used: `_normalize_spaces` (double-space collapse + watermark strip) at load time, `_normalize_chunk` (newline flattening) per chunk after splitting.
- `_normalize_chunk` flattens line-wrap newlines (except before bullet `-` markers and paragraph breaks) so that LLM-generated quotes, which naturally have no newlines, can substring-match the stored chunk text in the verification guard.
- `current_section` is tracked across pages so that continuation chunks (body of a Task Statement that overflows to the next page) inherit the correct section label.
- A `MIN_CHUNK_LEN = 20` filter removes orphaned bullet markers (e.g., a lone `"-"`) and bare domain headers with no body, both of which arise from page-boundary PDF layout artifacts.
- Result on the sample PDF: **41 chunks**, avg 463 chars, min 39, max 800, all 10 pages covered.

**Done when**: Chunk count is reasonable (not 1 giant chunk, not hundreds of tiny ones) and spot-checked chunks look semantically coherent.

**Push**: commit (`git add . && git commit -m "Milestone 3: header-aware chunking"`), then push the branch to GitHub.

**⏸ Wait for approval before starting Milestone 4.**

---

## Milestone 4: Embedding + FAISS Index Build
**Goal**: Turn chunks into a persisted FAISS index, per specs.md §6.

1. In `ingest.py`, write a `build_index(chunks)` function that:
   - Initializes `OpenAIEmbeddings(model=config.EMBEDDING_MODEL)`.
   - Builds a `FAISS` vector store from the chunks via `FAISS.from_documents(chunks, embeddings)`.
   - Persists it with `vectorstore.save_local(config.FAISS_INDEX_DIR)`.
2. Wire `ingest.py`'s `main()` to: load → split → build_index, with progress print statements (file count, chunk count, "Index saved to ...").
3. Run `uv run python ingest.py` end-to-end on the sample documents.

**Branch**: `git checkout -b milestone-4-faiss-index`

**Done when**: `faiss_index/` contains the generated index files (`index.faiss`, `index.pkl`), and re-running `ingest.py` successfully rebuilds them from scratch.

**Push**: commit (`git add . && git commit -m "Milestone 4: embedding + FAISS index build"`), then push the branch to GitHub. Note: `faiss_index/` is gitignored, so only `ingest.py` and code changes are pushed.

**⏸ Wait for approval before starting Milestone 5.**

---

## Milestone 5: Retrieval + Generation Pipeline (`rag_pipeline.py`)
**Goal**: A reusable function that takes a question and returns an answer + sources, per specs.md §7.

1. Write a `load_vectorstore()` function that loads the FAISS index via `FAISS.load_local(config.FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)`.
2. Write a `get_answer(question)` function that:
   - Retrieves top-k chunks via `vectorstore.similarity_search(question, k=config.TOP_K)`.
   - Builds a prompt using the **Strict Grounding Rules** system instruction from specs.md §7 (answer only from context, no prior knowledge/inference, exact `NO INFORMATION AVAILABLE` if not fully supported), with each retrieved chunk tagged by its source filename and page number.
   - Requests the **Required Answer Format** from specs.md §7 (Answer, Page Number(s), Exact Supporting Quote) as JSON, and parses/validates it with a `pydantic` model (`RAGAnswer`: `answer`, `page_numbers: list[int]`, `quote: str | None`, `no_information: bool`).
   - Calls `ChatOpenAI(model=config.LLM_MODEL)` (or raw OpenAI client) to generate the response.
3. Implement the **quote verification guard** (specs.md §7 note): if the response is not `NO INFORMATION AVAILABLE`, check that the "Exact Supporting Quote" is a substring of the corresponding retrieved chunk's `page_content`. If verification fails, override the response to `NO INFORMATION AVAILABLE`.
4. Return a dict: `{"answer": str, "page_numbers": [...], "quote": str | None, "sources": [{"source": ..., "page": ..., "section": ...}, ...], "no_information": bool}`.
5. Test directly via `uv run python -c "from rag_pipeline import get_answer; print(get_answer('...'))"` with:
   - A question clearly answerable from the sample document (expect Answer + Page Number(s) + verified Quote).
   - A question NOT covered by the sample document (expect `NO INFORMATION AVAILABLE`).

**Branch**: `git checkout -b milestone-5-rag-pipeline`

**Done when**: Both test cases above behave as expected, and the verification guard correctly catches at least one deliberately-induced unverifiable quote (e.g., by temporarily asking the model to paraphrase).

**Push**: commit (`git add . && git commit -m "Milestone 5: retrieval + generation pipeline with strict grounding"`), then push the branch to GitHub.

**⏸ Wait for approval before starting Milestone 6.**

---

## Milestone 6: Streamlit UI (`app.py`)
**Goal**: The chat interface from specs.md §8.

1. Set up the basic Streamlit page (`st.title`, `st.chat_input`).
2. On each user input:
   - Call `rag_pipeline.get_answer(question)`.
   - Append the question/result dict to `st.session_state.history`.
3. Render the conversation history using `st.chat_message` for both user and assistant turns.
4. For each assistant turn, render the result per the **Required Answer Format**:
   - If `no_information` is `True`, display the literal `NO INFORMATION AVAILABLE`.
   - Otherwise, display **Answer**, **Page Number(s)**, and **Exact Supporting Quote** as distinct, clearly labeled fields (e.g., quote in a blockquote or code block).
5. Add a "Sources" expander listing `source` (and `page`/`section` if present) for each retrieved chunk.
6. Add a sidebar showing: number of indexed documents/chunks (read from FAISS index size) and the `FAISS_INDEX_DIR` path.
7. Handle the "no index found" case gracefully (e.g., a friendly message telling the user to run `ingest.py` first).

**Branch**: `git checkout -b milestone-6-streamlit-ui`

**Done when**: `uv run streamlit run app.py` opens a working chat UI; asking a question grounded in the sample documents returns an Answer/Page Number(s)/Quote response with correct sources, and an out-of-scope question returns `NO INFORMATION AVAILABLE`.

**Push**: commit (`git add . && git commit -m "Milestone 6: Streamlit chat UI"`), then push the branch to GitHub.

**⏸ Wait for approval before starting Milestone 7.**

---

## Milestone 7: End-to-End Test Pass
**Goal**: Validate the full flow against the sample PDF.

1. Confirm `documents/source.pdf` is the 10-page PDF and `faiss_index/` was built from it (re-run `ingest.py` if needed).
2. Run `uv run streamlit run app.py` and test:
   - A question clearly answerable from the PDF → correct answer + correct page number(s) + verified quote.
   - A question NOT covered by the PDF → `NO INFORMATION AVAILABLE`.
   - A question where the answer spans multiple pages → correct multi-page citation.
   - A multi-turn conversation → check that history displays correctly (note: each call to `get_answer` in v1 is independent per specs.md scope; conversational memory is not required).
3. Note any retrieval quality issues (e.g., wrong chunks retrieved) — if frequent, revisit `CHUNK_SIZE`/`TOP_K`/`EMBEDDING_MODEL` per the rationale in specs.md §6.

**Branch**: `git checkout -b milestone-7-e2e-test`

**Actual findings**:
- **Watermark bug found and fixed**: The page-footer `"Anthropic, PBC · Confidential Need to Know (NTK)"` appeared in 9 of 41 chunks. In one case the LLM returned it as the answer. Fixed by adding a regex strip to `_normalize_spaces` in `ingest.py` at load time. Zero watermarked chunks after rebuild.
- **Multi-page aggregation limit**: A question asking for all 5 domain weights returned `NO INFORMATION AVAILABLE` — each domain is a separate chunk and `TOP_K=4` can only retrieve 4. This is **correct strict-grounding behavior** (rule 4: "if only part of the answer is available → NO INFORMATION AVAILABLE"), not a bug. Questions requiring aggregation across more than `TOP_K` chunks will always hit this limit.
- **Test results**: (1) clearly answerable → answer + verified quote ✓, (2) out-of-scope → `no_information: True` ✓, (3) multi-page (pages 4–6) → answer + verified quote ✓, (4) multi-turn simulation → correct independent responses ✓.

**Done when**: You're satisfied with answer quality, citation accuracy, and `NO INFORMATION AVAILABLE` behavior on the sample PDF.

**Push**: commit (`git add . && git commit -m "Milestone 7: end-to-end test pass"`), then push the branch to GitHub. Commit any config tweaks made during testing.

**⏸ Wait for approval before starting Milestone 8.**

---

## Milestone 8: Polish & Documentation
**Goal**: Make the project easy to pick up again later.

1. Write a `README.md` covering: setup (`uv venv`, `uv pip install`), adding documents, running `ingest.py`, running `app.py`, and configuration options (`.env` reference).
2. Add basic error handling: missing `.env`/API key, empty `documents/` folder, OpenAI API errors (rate limits, invalid key) surfaced as user-friendly messages in the UI.
3. Review `config.py` defaults one more time against specs.md to confirm nothing drifted during implementation.

**Branch**: `git checkout -b milestone-8-polish`

**Done when**: A fresh clone + `uv` setup + README instructions gets a new user to a working chat in under 10 minutes (assuming they have their own documents and API key).

**Push**: commit (`git add . && git commit -m "Milestone 8: polish, error handling, README"`), then push the branch to GitHub.

**⏸ Wait for approval before merging to `main`.**

---

## Summary of Build Order
1. Project setup (`uv`, folders, deps)
2. `config.py`
3. Document loading
4. Chunking
5. Embedding + FAISS index build (`ingest.py` complete)
6. Retrieval + generation (`rag_pipeline.py`)
7. Streamlit UI (`app.py`)
8. End-to-end test with real documents
9. Polish + README

## Future Considerations
Not part of this build plan — see specs.md §12 (Confluence sync, incremental indexing, conversation persistence, multi-user, alternate providers, source previews).