# RAG Complaint Chatbot — CrediTrust Financial

Internal AI tool that transforms CFPB customer complaint narratives into actionable insights using Retrieval-Augmented Generation (RAG).

## Project structure

```
rag-complaint-chatbot/
├── app.py                          # Task 4 — Streamlit chat UI
├── data/README.md                  # How to obtain local datasets
├── notebooks/01_eda_preprocessing.ipynb   # Task 1 — EDA notebook
├── reports/
│   ├── task1_eda_summary.md        # Task 1 — written findings
│   └── rag_evaluation.md           # Task 3 — evaluation table
├── src/
│   ├── preprocess.py               # Task 1 — cleaning pipeline
│   ├── eda.py                      # Task 1 — chunked EDA helpers
│   ├── sampling.py                 # Task 2 — stratified sampling
│   ├── chunking.py                 # Task 2 — text chunking
│   ├── build_vector_store.py       # Task 2 — embedding + ChromaDB
│   ├── load_prebuilt_store.py      # Task 3 — load full parquet index
│   ├── config.py                   # Shared configuration
│   ├── retriever.py                # Task 3 — semantic retriever
│   ├── prompts.py                  # Task 3 — analyst prompt template
│   ├── generator.py                # Task 3 — LLM answer generation
│   ├── rag.py                      # Task 3 — end-to-end RAG pipeline
│   └── evaluate_rag.py             # Task 3 — qualitative evaluation
├── tests/                          # Unit tests (Tasks 1–4)
└── vector_store/                   # ChromaDB index (generated locally)
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Place the CFPB CSV at `data/raw/complaints.csv` (see `data/README.md`). **Do not commit data files.**

---

## Task 1 — EDA & preprocessing

```bash
python -m src.preprocess
# or run notebooks/01_eda_preprocessing.ipynb
```

| File | Purpose |
|------|---------|
| `src/preprocess.py` | Memory-safe chunked load, filter, clean, save |
| `src/eda.py` | Chunked narrative availability & product stats |
| `notebooks/01_eda_preprocessing.ipynb` | Interactive EDA with charts |
| `reports/task1_eda_summary.md` | Written findings (2–3 paragraphs) |

**Outputs (local, gitignored):** `data/filtered_complaints.csv`, `data/processed/complaints_clean.csv`

---

## Task 2 — Chunking, embeddings & vector store

```bash
python -m src.build_vector_store
python -m src.build_vector_store --dry-run   # validate without embeddings
```

| File | Purpose |
|------|---------|
| `src/sampling.py` | Stratified 12K sample across product categories |
| `src/chunking.py` | 500-char chunks, 50-char overlap |
| `src/build_vector_store.py` | Embed with `all-MiniLM-L6-v2`, index in ChromaDB |

**Output (local):** `vector_store/chromadb/`

---

## Task 3 — RAG pipeline & evaluation

```bash
# Load pre-built full-scale index (recommended)
python -m src.load_prebuilt_store

# Run qualitative evaluation
python -m src.evaluate_rag --mock --fallback
python -m src.evaluate_rag --output reports/rag_evaluation.md
```

| File | Purpose |
|------|---------|
| `src/retriever.py` | Embed query → ChromaDB top-k retrieval |
| `src/prompts.py` | Grounded analyst prompt template |
| `src/generator.py` | `google/flan-t5-base` generation + fallback |
| `src/rag.py` | `RAGPipeline.ask(question, product_category, top_k)` |
| `src/evaluate_rag.py` | 10-question evaluation runner |
| `reports/rag_evaluation.md` | Quality-scored evaluation table |

---

## Task 4 — Interactive chat UI

```bash
streamlit run app.py
```

Features: chat history, product filter, source excerpts, streaming, clear button, fallback mode.

---

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

## Branches

| Branch | Task |
|--------|------|
| `task-1/eda-preprocessing` | EDA and preprocessing |
| `task-2/chunking-embeddings` | Vector store pipeline |
| `task-3/rag-pipeline` | RAG core + evaluation |
| `task-4/chat-ui` | Streamlit interface |
| `main` | All tasks merged |
