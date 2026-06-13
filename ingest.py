import re
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import config

# Matches Task Statement / Domain headers (single-space normalized text).
# [^\n]* stops at the first newline so the title doesn't bleed into content.
_HEADER_RE = re.compile(
    r"((?:Task\s+Statement\s+[\d.]+\s*:[^\n]*|Domain\s+\d+[^\n]*:))",
    re.IGNORECASE,
)


def _normalize_spaces(text: str) -> str:
    """Phase 1 (load time): collapse double-space PDF artifact, keep newlines."""
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _normalize_chunk(text: str) -> str:
    """Phase 2 (post-split): collapse line-wrap newlines so LLM quotes match."""
    # space-only lines between words: "word\n \nword" → "word word"
    text = re.sub(r"(?<=[^\n])\n[ \t]+\n(?=[^\n])", " ", text)
    # remaining line-wrap newlines (not before bullet "-" or paragraph break)
    text = re.sub(r"(?<!\n)\n(?![\n\-])", " ", text)
    # strip trailing lone bullet markers (PDF page-boundary artifact)
    text = re.sub(r"\n-\s*$", "", text)
    # tidy up
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(pdf_path: str) -> list[Document]:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = os.path.basename(pdf_path)
        doc.page_content = _normalize_spaces(doc.page_content)
    print(f"Loaded {len(docs)} pages from {pdf_path}")
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    section_docs: list[Document] = []
    current_section = ""  # carry forward across page boundaries

    for doc in docs:
        page_meta = dict(doc.metadata)
        text = doc.page_content
        parts = _HEADER_RE.split(text)

        if len(parts) == 1:
            # No header on this page — content continues prior section
            content = _normalize_chunk(text)
            if content:
                section_docs.append(Document(
                    page_content=content,
                    metadata={**page_meta, "section": current_section},
                ))
        else:
            # parts = [pre_text, header1, body1, header2, body2, ...]
            pre = _normalize_chunk(parts[0])
            if pre:
                section_docs.append(Document(
                    page_content=pre,
                    metadata={**page_meta, "section": current_section},
                ))
            for i in range(1, len(parts), 2):
                header = parts[i].strip()
                body = _normalize_chunk(parts[i + 1] if i + 1 < len(parts) else "")
                current_section = header
                full_text = f"{header}\n{body}".strip() if body else header
                if full_text:
                    section_docs.append(Document(
                        page_content=full_text,
                        metadata={**page_meta, "section": header},
                    ))

    # Phase 2: recursive character split for any chunk exceeding CHUNK_SIZE
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    MIN_CHUNK_LEN = 20  # filter out orphaned bullets / bare headers
    final_chunks: list[Document] = []
    for doc in section_docs:
        if len(doc.page_content) > config.CHUNK_SIZE:
            final_chunks.extend(
                c for c in splitter.split_documents([doc])
                if len(c.page_content.strip()) >= MIN_CHUNK_LEN
            )
        elif len(doc.page_content.strip()) >= MIN_CHUNK_LEN:
            final_chunks.append(doc)

    return final_chunks


if __name__ == "__main__":
    docs = load_documents(config.PDF_PATH)
    chunks = split_documents(docs)

    lengths = [len(c.page_content) for c in chunks]
    print(f"\nChunk count  : {len(chunks)}")
    print(f"Avg length   : {sum(lengths) // len(lengths)} chars")
    print(f"Min / Max    : {min(lengths)} / {max(lengths)} chars")

    print("\n--- Sample chunk 1 ---")
    print(f"metadata : {chunks[0].metadata}")
    print(f"content  : {chunks[0].page_content[:400]}")

    print("\n--- Sample chunk (mid-document) ---")
    mid = len(chunks) // 2
    print(f"metadata : {chunks[mid].metadata}")
    print(f"content  : {chunks[mid].page_content[:400]}")

    print("\n--- Sample chunk (last) ---")
    print(f"metadata : {chunks[-1].metadata}")
    print(f"content  : {chunks[-1].page_content[:400]}")
