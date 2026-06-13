# CLAUDE.md

## Core Architecture
- **Source of Truth:** `specs.md` (What/Why), `implementation.md` (How/Milestones).
- **Code Conventions:** - All tunables (chunk size, overlap, top-k, models, paths) must live in `config.py` or `.env`.
  - Isolate provider-specific calls (OpenAI/FAISS) to keep `rag_pipeline.py` core agnostic.
  - Pin all new dependencies in `requirements.txt`.

## Workflow & Guardrails
- **Branching:** Work on one milestone at a time on branch `milestone-N-<name>`, based off the prior approved branch.
- **Strict Stop:** Stop when a milestone's "Done when" criteria are met. Do not merge, push, or advance to the next milestone without explicit user approval.
- **Strict Grounding (RAG):** System prompt in `rag_pipeline.py` must enforce `specs.md` §7 (exact phrase `NO INFORMATION AVAILABLE` if context is insufficient; zero inference). Output must always contain: Answer, Page Number(s), Exact Supporting Quote.

## Environment & Hygiene
- **Exclusions:** Never stage or commit `.env`, `faiss_index/`, or `documents/`.
- **Security:** Zero logging/printing of raw API keys or full document contents.
- **Conflicts:** Pause and ask if instructions conflict with `specs.md` or milestone ordering.