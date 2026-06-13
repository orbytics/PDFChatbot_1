import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY is not set. Add it to your .env file."
    )

PDF_PATH: str = os.getenv("PDF_PATH", "./documents/source.pdf")
FAISS_INDEX_DIR: str = os.getenv("FAISS_INDEX_DIR", "./faiss_index")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K: int = int(os.getenv("TOP_K", "4"))
