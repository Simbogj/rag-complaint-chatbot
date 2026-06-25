# Task 4: Streamlit chat interface for the complaint RAG system

"""Run with:
    streamlit run app.py
"""

from __future__ import annotations

import time
from typing import Any
import streamlit as st

from src.config import DEFAULT_TOP_K, TARGET_PRODUCTS, VECTOR_STORE_PATH
from src.rag_core import RAGPipeline

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def init_session_state() -> None:
    """Initialize Streamlit session state for the chat history."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

@st.cache_resource(show_spinner="Loading RAG pipeline…")
def load_pipeline(use_fallback: bool) -> RAGPipeline:
    """Instantiate the RAG pipeline (fallback uses local LLM)."""
    return RAGPipeline(use_fallback=use_fallback)

def source_to_dict(source: dict) -> dict[str, Any]:
    """The rag_core returns plain dicts; this is a pass‑through helper."""
    return source

def render_sources(sources: list[dict[str, Any]]) -> None:
    """Display retrieved complaint excerpts beneath the answer."""
    if not sources:
        st.info("No complaint excerpts were retrieved for this answer.")
        return
    with st.expander(f"Sources ({len(sources)} excerpts)", expanded=True):
        for idx, src in enumerate(sources, start=1):
            meta = src.get("metadata", {})
            distance = src.get("distance")
            score = f" | relevance: {1 - distance:.3f}" if distance is not None else ""
            st.markdown(
                f"**Source {idx}** — {meta.get('product_category', 'Unknown')} | Issue: {meta.get('issue', 'N/A')}{score}"
            )
            st.caption(src.get("text", ""))
            if idx < len(sources):
                st.divider()

def stream_text(text: str, delay: float = 0.015) -> None:
    """Render answer text token‑by‑token for a smoother UI experience."""
    placeholder = st.empty()
    displayed = ""
    for token in text.split():
        displayed = f"{displayed} {token}".strip()
        placeholder.markdown(displayed)
        time.sleep(delay)

# ------------------------------------------------------------
# Main app
# ------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="CrediTrust Complaint Analyst", page_icon="💬", layout="wide")
    init_session_state()

    st.title("CrediTrust Complaint Analyst")
    st.caption(
        "Ask plain‑English questions about customer complaints. "
        "Answers are synthesized from retrieved complaint narratives."
    )

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        product_filter = st.selectbox(
            "Product category",
            ["All products"] + list(TARGET_PRODUCTS),
            help="Filter retrieval to a single financial product category.",
        )
        top_k = st.slider(
            "Chunks to retrieve (top‑k)",
            min_value=1,
            max_value=10,
            value=DEFAULT_TOP_K,
        )
        use_fallback = st.checkbox(
            "Fallback mode (no LLM)",
            value=True,
            help="Use extractive summaries when the LLM backend is unavailable.",
        )
        enable_streaming = st.checkbox(
            "Stream responses",
            value=True,
            help="Reveal the answer progressively for a smoother chat experience.",
        )
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.subheader("Vector store")
        try:
            pipeline = load_pipeline(use_fallback)
            chunk_count = pipeline.collection.count()
            st.metric("Indexed chunks", f"{chunk_count:,}")
            st.caption(f"Store path: `{VECTOR_STORE_PATH}`")
        except Exception as exc:
            st.error(f"Could not load pipeline: {exc}")
            return

        st.divider()
        st.markdown("**Example questions**")
        st.markdown(
            "- Why are people unhappy with credit cards?\\n"
            "- What fraud issues appear in money transfers?\\n"
            "- What billing disputes affect savings accounts?"
        )

    # Chat history display
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"])

    # User input
    question = st.chat_input("Ask about customer complaints…")
    if not question:
        return

    # Record user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Prepare query parameters
    category = None if product_filter == "All products" else product_filter

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving complaints and generating answer…"):
            try:
                response = pipeline.ask(query=question, product_category=category, top_k=top_k)
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                return
        if enable_streaming:
            stream_text(response["answer"])
        else:
            st.markdown(response["answer"])
        source_dicts = [source_to_dict(s) for s in response["sources"]]
        render_sources(source_dicts)

    # Append assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": response["answer"], "sources": source_dicts})

if __name__ == "__main__":
    main()
