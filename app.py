"""
Task 4: Streamlit chat interface for the complaint RAG assistant.

Run:
    streamlit run app.py
"""

from __future__ import annotations

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


def main() -> None:
    st.set_page_config(
        page_title="CrediTrust Complaint Analyst",
        page_icon="💬",
        layout="wide",
    )
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

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])

    question = st.chat_input("Ask about customer complaints...")
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


if __name__ == "__main__":
    main()
