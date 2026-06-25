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

## Task 3 — RAG pipeline & evaluation

### Load pre-built full-scale vector store (recommended for Tasks 3–4)

Place `complaint_embeddings.parquet` in `data/` and load it into ChromaDB:

```bash
python -m src.load_prebuilt_store
```

Or use the Task 2 index at `vector_store/chromadb/` after running `python -m src.build_vector_store`.

### Ask questions programmatically

```python
from src.rag import RAGPipeline

pipeline = RAGPipeline(use_fallback=False)
response = pipeline.ask(
    "Why are people unhappy with credit cards?",
    product_category="Credit Card",
    top_k=5,
)
print(response.answer)
for source in response.sources:
    print(source.metadata, source.text[:120])
```

### Run qualitative evaluation

```bash
python -m src.evaluate_rag --fallback
python -m src.evaluate_rag --mock --fallback   # offline demo without vector store / torch
python -m src.evaluate_rag --output reports/rag_evaluation.md
```

Outputs:
- `reports/rag_evaluation.md` — evaluation table for your final report
- `reports/rag_evaluation.json` — raw answers and retrieved sources

**RAG components:**
- **Retriever:** embeds queries with `all-MiniLM-L6-v2`, searches ChromaDB (cosine similarity, `top_k=5`)
- **Prompt:** analyst template grounded in retrieved excerpts only
- **Generator:** `google/flan-t5-base` via Hugging Face (`--fallback` for offline extractive summaries)

## Task 4 — Interactive chat UI

Launch the Streamlit app:

```bash
streamlit run app.py
```

**Features:**
- Natural-language question input with chat history
- Product category filter (Credit Card, Personal Loan, Savings Account, Money Transfer)
- Retrieved source excerpts shown below each answer for verification
- Adjustable `top-k` retrieval count
- **Clear** button to reset the conversation
- Optional response streaming for smoother UX
- Fallback mode when the LLM backend is unavailable

## Tests

```bash
pytest tests/ -v
```
