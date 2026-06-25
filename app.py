"""
<<<<<<< HEAD
Task 4: Streamlit chat interface for the complaint RAG system.
=======
Task 4: Streamlit chat interface for the complaint RAG assistant.
>>>>>>> task-2/chunking-embeddings

Run:
    streamlit run app.py
"""

from __future__ import annotations

<<<<<<< HEAD
import time
from typing import Any

import streamlit as st

from src.config import DEFAULT_TOP_K, TARGET_PRODUCTS, VECTOR_STORE_PATH
from src.rag import RAGPipeline
from src.retriever import RetrievedChunk


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


@st.cache_resource(show_spinner="Loading RAG pipeline...")
def load_pipeline(use_fallback: bool) -> RAGPipeline:
    return RAGPipeline(use_fallback=use_fallback)


def source_to_dict(source: RetrievedChunk) -> dict[str, Any]:
    return {
        "text": source.text,
        "metadata": source.metadata,
        "distance": source.distance,
        "id": source.id,
    }


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.info("No complaint excerpts were retrieved for this answer.")
        return

    with st.expander(f"Sources ({len(sources)} excerpts)", expanded=True):
        for idx, source in enumerate(sources, start=1):
            meta = source.get("metadata", {})
            distance = source.get("distance")
            score_text = f" | relevance: {1 - distance:.3f}" if distance is not None else ""

            st.markdown(
                f"**Source {idx}** — {meta.get('product_category', 'Unknown')} "
                f"| Issue: {meta.get('issue', 'N/A')}{score_text}"
            )
            st.caption(source.get("text", ""))
            if idx < len(sources):
                st.divider()


def stream_text(text: str, delay: float = 0.015) -> None:
    """Render answer text with a lightweight token-by-token effect."""
    placeholder = st.empty()
    displayed = ""
    for token in text.split():
        displayed = f"{displayed} {token}".strip()
        placeholder.markdown(displayed)
        time.sleep(delay)
=======
import streamlit as st

from src.generator import ComplaintGenerator
from src.rag import ComplaintRAG
from src.retriever import ComplaintRetriever

PRODUCT_FILTERS = [
    "All products",
    "Credit Card",
    "Personal Loan",
    "Savings Account",
    "Money Transfer",
]


@st.cache_resource(show_spinner="Loading vector store and models...")
def load_rag_pipeline(max_chunks: int = 5_000) -> ComplaintRAG:
    retriever = ComplaintRetriever.from_prebuilt_store(max_chunks=max_chunks)
    return ComplaintRAG(retriever=retriever, generator=ComplaintGenerator(), top_k=5)


def render_sources(sources) -> None:
    if not sources:
        st.info("No source excerpts were retrieved for this answer.")
        return

    st.subheader("Sources")
    for idx, source in enumerate(sources, start=1):
        meta = source.metadata
        header = (
            f"Source {idx} — {meta.get('product_category', 'Unknown product')} | "
            f"{meta.get('issue', 'Unknown issue')} | "
            f"score {source.score:.3f}"
        )
        with st.expander(header):
            st.write(source.text)
            st.caption(
                f"Company: {meta.get('company', 'N/A')} | "
                f"State: {meta.get('state', 'N/A')} | "
                f"Complaint ID: {meta.get('complaint_id', 'N/A')}"
            )
>>>>>>> task-2/chunking-embeddings


def main() -> None:
    st.set_page_config(
        page_title="CrediTrust Complaint Analyst",
        page_icon="💬",
        layout="wide",
    )
<<<<<<< HEAD
    init_session_state()

    st.title("CrediTrust Complaint Analyst")
    st.caption(
        "Ask plain-English questions about customer complaints. "
        "Answers are synthesized from retrieved complaint narratives."
    )

    with st.sidebar:
        st.header("Settings")
        product_filter = st.selectbox(
            "Product category",
            ["All products"] + list(TARGET_PRODUCTS),
            help="Filter retrieval to a single financial product category.",
        )
        top_k = st.slider(
            "Chunks to retrieve (top-k)",
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
            chunk_count = pipeline.retriever.chunk_count
            st.metric("Indexed chunks", f"{chunk_count:,}")
            if chunk_count == 0:
                st.warning(
                    "No chunks found. Run `python -m src.build_vector_store` or "
                    "`python -m src.load_prebuilt_store` first."
                )
            else:
                st.caption(f"Store path: `{VECTOR_STORE_PATH}`")
        except Exception as exc:
            st.error(f"Could not load pipeline: {exc}")
            return

        st.divider()
        st.markdown("**Example questions**")
        st.markdown(
            "- Why are people unhappy with credit cards?\n"
            "- What fraud issues appear in money transfers?\n"
            "- What billing disputes affect savings accounts?"
        )
=======
    st.title("CrediTrust Complaint Analyst")
    st.caption(
        "Ask plain-English questions about customer complaints. "
        "Answers are grounded in retrieved complaint excerpts."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Settings")
        product_filter = st.selectbox("Product filter", PRODUCT_FILTERS)
        max_chunks = st.slider("Indexed chunks (dev sample)", 1000, 20000, 5000, step=1000)
        if st.button("Clear conversation", type="secondary"):
            st.session_state.messages = []
            st.rerun()

    rag = load_rag_pipeline(max_chunks=max_chunks)
    category = None if product_filter == "All products" else product_filter
>>>>>>> task-2/chunking-embeddings

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])

    question = st.chat_input("Ask about customer complaints...")
<<<<<<< HEAD
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    category = None if product_filter == "All products" else product_filter

    with st.chat_message("assistant"):
        with st.spinner("Retrieving complaints and generating answer..."):
            try:
                response = pipeline.ask(
                    question=question,
                    product_category=category,
                    top_k=top_k,
                )
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                return

        if enable_streaming:
            stream_text(response.answer)
        else:
            st.markdown(response.answer)

        source_dicts = [source_to_dict(s) for s in response.sources]
        render_sources(source_dicts)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response.answer,
            "sources": source_dicts,
        }
    )
=======
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching complaints and drafting an answer..."):
                response = rag.ask(question, product_category=category)
            st.markdown(response.answer)
            render_sources(response.sources)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.answer,
                "sources": response.sources,
            }
        )
>>>>>>> task-2/chunking-embeddings


if __name__ == "__main__":
    main()
