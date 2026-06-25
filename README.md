# RAG Complaint Chatbot

Internal AI tool for analyzing CFPB customer complaints using Retrieval-Augmented Generation (RAG).

## Project structure

```
rag-complaint-chatbot/
├── data/processed/          # Cleaned complaint datasets
├── src/                     # Pipeline modules
├── vector_store/            # ChromaDB index (Task 2 output)
├── notebooks/               # EDA notebooks
├── tests/                   # Unit tests
└── app.py                   # Streamlit UI (Task 4)
```

## Branches

| Branch | Task |
|--------|------|
| `task-1/eda-preprocessing` | EDA and text cleaning |
| `task-2/chunking-embeddings` | Sampling, chunking, vector store |
| `task-3/rag-pipeline` | RAG retriever + LLM + evaluation |
| `task-4/chat-ui` | Interactive chat interface |

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Task 1 — EDA & preprocessing

Run the notebook or use `src/preprocess.py`. Output: `data/processed/complaints_clean.csv`.

## Task 2 — Chunking, embeddings & vector store

Build a stratified sample (~12K complaints), chunk narratives, embed with `all-MiniLM-L6-v2`, and persist to ChromaDB:

```bash
python -m src.build_vector_store
```

Options:

```bash
python -m src.build_vector_store --sample-size 12000 --batch-size 64
python -m src.build_vector_store --dry-run   # sample + chunk only, no embeddings
```

**Outputs:**
- `data/processed/sampled_complaints.csv` — stratified sample
- `vector_store/chromadb/` — persisted ChromaDB index
- `vector_store/chromadb/build_stats.json` — build metadata

**Design choices:**
- **Sampling:** sklearn stratified split preserves product-category proportions across Credit Card, Personal Loan, Savings Account, and Money Transfer.
- **Chunking:** 500 characters with 50-character overlap (matches the pre-built full-scale index spec).
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` — fast, 384-dim, strong semantic search for short-to-medium financial text.

## Task 3 — RAG core & evaluation

Uses the pre-built parquet embeddings indexed into `vector_store/prebuilt_chromadb/`:

```bash
python -m src.evaluate_rag --max-chunks 5000
```

**Modules:** `src/retriever.py`, `src/prompts.py`, `src/generator.py`, `src/rag.py`

## Task 4 — Chat UI

```bash
streamlit run app.py
```

The UI shows generated answers with expandable source excerpts and a product-category filter.

## Tests

```bash
pytest tests/ -v
```
