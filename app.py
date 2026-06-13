import os
import streamlit as st
import rag_pipeline
import config

st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="centered")


# ---------------------------------------------------------------------------
# Render helper
# ---------------------------------------------------------------------------
def _render_answer(result: dict) -> None:
    if result["no_information"]:
        st.error("NO INFORMATION AVAILABLE")
        st.caption(
            "The retrieved context does not contain enough information to answer this question."
        )
    else:
        st.markdown(f"**Answer**\n\n{result['answer']}")
        pages = ", ".join(str(p) for p in result["page_numbers"])
        st.markdown(f"**Page Number(s):** {pages}")
        st.markdown("**Exact Supporting Quote**")
        st.markdown(f"> {result['quote']}")

    with st.expander("Sources", expanded=False):
        for src in result["sources"]:
            section = src.get("section", "")
            label = f"📄 {src['source']} — Page {src['page']}"
            if section:
                label += f" · *{section[:60]}{'…' if len(section) > 60 else ''}*"
            st.markdown(label)


# ---------------------------------------------------------------------------
# Sidebar — status info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Status")

    index_exists = os.path.isfile(
        os.path.join(config.FAISS_INDEX_DIR, "index.faiss")
    ) and os.path.isfile(os.path.join(config.FAISS_INDEX_DIR, "index.pkl"))

    if index_exists:
        try:
            chunk_count = rag_pipeline.get_index_size()
            st.success("Index loaded")
            st.metric("Chunks indexed", chunk_count)
        except Exception as e:
            st.error(f"Failed to load index: {e}")
            index_exists = False
    else:
        st.warning("No index found")

    st.divider()
    st.caption(f"**Source PDF:** {os.path.basename(config.PDF_PATH)}")
    st.caption(f"**Index dir:** `{config.FAISS_INDEX_DIR}`")
    st.caption(f"**LLM:** {config.LLM_MODEL}")
    st.caption(f"**Embeddings:** {config.EMBEDDING_MODEL}")
    st.caption(f"**Top-k:** {config.TOP_K}")


# ---------------------------------------------------------------------------
# Main — chat interface
# ---------------------------------------------------------------------------
st.title("📄 PDF Chatbot")
st.caption(
    "Ask questions about the indexed document. Answers are grounded strictly in the source."
)

if not index_exists:
    st.error(
        "No FAISS index found. Run `uv run python ingest.py` first to index your document."
    )
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

# Render conversation history
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        _render_answer(turn["result"])

# Chat input
if question := st.chat_input("Ask a question about the document…"):
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            result = rag_pipeline.get_answer(question)
        _render_answer(result)

    st.session_state.history.append({"question": question, "result": result})
