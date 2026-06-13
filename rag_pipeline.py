import json
from typing import Any

from pydantic import BaseModel, ValidationError
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document, HumanMessage, SystemMessage

import config

# ---------------------------------------------------------------------------
# Strict grounding system prompt — specs.md §7
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a document Q&A assistant. Follow these rules without exception:

1. Answer ONLY from the retrieved document context provided below.
2. Do NOT use prior knowledge, model training data, assumptions, inference, reasoning beyond the provided text, or external sources.
3. If the answer is not explicitly stated in the retrieved context, set no_information=true.
4. If only part of the answer is available, set no_information=true.
5. Do NOT paraphrase critical facts.
6. Do NOT invent information.
7. Do NOT guess.
8. Do NOT summarize missing information.
9. Do NOT provide recommendations unless explicitly written in the document.
10. Preserve terminology exactly as written in the source whenever possible.
11. If retrieved chunks do not contain enough information to answer with complete confidence, set no_information=true.

Respond ONLY in valid JSON matching this exact schema:
{
  "no_information": boolean,
  "answer": string or null,
  "page_numbers": array of integers (1-indexed page numbers from the source tags),
  "quote": string or null
}

Rules for the quote field:
- Copy the supporting text VERBATIM from the retrieved context — do not alter a single word or character.
- The quote must be the shortest passage that fully supports the answer.
- If no_information is true, set answer=null, page_numbers=[], quote=null."""


# ---------------------------------------------------------------------------
# Pydantic response model
# ---------------------------------------------------------------------------
class RAGAnswer(BaseModel):
    no_information: bool = False
    answer: str | None = None
    page_numbers: list[int] = []
    quote: str | None = None


# ---------------------------------------------------------------------------
# Module-level singleton — avoids reloading the index on every call
# ---------------------------------------------------------------------------
_vectorstore: FAISS | None = None


def load_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
        )
        _vectorstore = FAISS.load_local(
            config.FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def get_index_size() -> int:
    return load_vectorstore().index.ntotal


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_context(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        page = doc.metadata.get("page_label") or str(doc.metadata.get("page", 0) + 1)
        source = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section", "")
        header = (
            f"[Source: {source}, Page: {page}"
            + (f", Section: {section}" if section else "")
            + "]"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _verify_quote(quote: str, docs: list[Document]) -> bool:
    """True if quote appears verbatim as a substring in any retrieved chunk."""
    return any(quote in doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_answer(question: str) -> dict[str, Any]:
    vs = load_vectorstore()
    docs = vs.similarity_search(question, k=config.TOP_K)
    context = _build_context(docs)

    user_message = f"RETRIEVED CONTEXT:\n{context}\n\nQUESTION: {question}\n\nRespond in JSON only."

    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        openai_api_key=config.OPENAI_API_KEY,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    # Parse and validate
    try:
        parsed = RAGAnswer(**json.loads(response.content))
    except (json.JSONDecodeError, ValidationError):
        parsed = RAGAnswer(no_information=True)

    # Quote verification guard (specs.md §7)
    if not parsed.no_information:
        if not parsed.quote or not _verify_quote(parsed.quote, docs):
            parsed = RAGAnswer(no_information=True)

    sources = [
        {
            "source": d.metadata.get("source", "unknown"),
            "page": d.metadata.get("page_label") or str(d.metadata.get("page", 0) + 1),
            "section": d.metadata.get("section", ""),
        }
        for d in docs
    ]

    return {
        "answer": parsed.answer,
        "page_numbers": parsed.page_numbers,
        "quote": parsed.quote,
        "sources": sources,
        "no_information": parsed.no_information,
    }


# ---------------------------------------------------------------------------
# Manual test (run via: uv run python rag_pipeline.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pprint

    tests = [
        ("IN-SCOPE", "What is the hub-and-spoke architecture in multi-agent systems?"),
        ("OUT-OF-SCOPE", "What is the capital of France?"),
        ("GUARD TEST", None),  # handled specially below
    ]

    for label, question in tests:
        if label == "GUARD TEST":
            # Directly test the guard with a fabricated non-matching quote
            from langchain.schema import Document as D
            fake_doc = D(page_content="The sky is blue.", metadata={})
            result = _verify_quote("The sky is green.", [fake_doc])
            print(f"\n=== GUARD TEST ===")
            print(f"Quote 'The sky is green.' in 'The sky is blue.' → {result} (expected False)")
            result2 = _verify_quote("The sky is blue.", [fake_doc])
            print(f"Quote 'The sky is blue.' in 'The sky is blue.' → {result2} (expected True)")
        else:
            print(f"\n=== {label} ===")
            print(f"Q: {question}")
            answer = get_answer(question)
            pprint.pprint(answer)
