import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document


def load_documents(pdf_path: str) -> list[Document]:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = os.path.basename(pdf_path)
    print(f"Loaded {len(docs)} pages from {pdf_path}")
    return docs


if __name__ == "__main__":
    import config
    docs = load_documents(config.PDF_PATH)
    sample = docs[0]
    print(f"\n--- Page 1 sample ---")
    print(f"metadata : {sample.metadata}")
    print(f"content  : {sample.page_content[:300]!r}")
