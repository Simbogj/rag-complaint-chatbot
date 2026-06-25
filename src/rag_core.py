import os
import logging
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline.

    Loads a persisted ChromaDB collection, embeds queries with the same model used for indexing,
    retrieves top‑k relevant chunks, builds a prompt, and generates an answer using a language model.
    """

    def __init__(
        self,
        vector_store_path: str = "vector_store/chromadb",
        collection_name: str = "complaints",
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        generator_model_name: str = "google/flan-t5-base",
        device: str = "cpu",
    ) -> None:
        # Embedding model (same as Task 2)
        self.embedder = SentenceTransformer(embed_model_name, device=device)
        logger.info("Loaded embedding model %s on %s", embed_model_name, device)

        # Initialise ChromaDB client (persisted DuckDB+Parquet implementation)
        self.client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=os.path.abspath(vector_store_path),
                anonymized_telemetry=False,
            )
        )
        self.collection = self.client.get_collection(name=collection_name)
        logger.info("Connected to ChromaDB collection '%s' at %s", collection_name, vector_store_path)

        # Generation model (local HuggingFace pipeline)
        tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)
        self.generator = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device.startswith("cuda") else -1,
        )
        logger.info("Loaded generator model %s", generator_model_name)

    def _embed_query(self, query: str) -> List[float]:
        return self.embedder.encode([query]).tolist()[0]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        product_category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top‑k chunks for *query*.

        Optional *product_category* filters results via ChromaDB's `where` clause.
        Returns a list of dicts with keys `text`, `metadata`, and `distance`.
        """
        query_emb = self._embed_query(query)
        where_clause = None
        if product_category:
            where_clause = {"product_category": {"$eq": product_category}}
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )
        sources: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            sources.append({"text": doc, "metadata": meta, "distance": dist})
        logger.debug("Retrieved %d sources for query '%s'", len(sources), query)
        return sources

    @staticmethod
    def _build_prompt(query: str, sources: List[Dict[str, Any]]) -> str:
        """Construct the prompt used by the generator.

        The prompt follows the template required for the evaluation.
        """
        context_parts = []
        for idx, src in enumerate(sources, start=1):
            snippet = src["text"].replace("\n", " ")
            context_parts.append(f"Source {idx}: {snippet}")
        context = "\n".join(context_parts)
        prompt = (
            "You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints. "
            "Use the following retrieved complaint excerpts to formulate your answer. "
            "If the context doesn't contain the answer, state that you don't have enough information.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
        return prompt

    def ask(
        self,
        query: str,
        product_category: Optional[str] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Run the full RAG pipeline and return answer + sources.
        """
        sources = self.retrieve(query, top_k=top_k, product_category=product_category)
        prompt = self._build_prompt(query, sources)
        # Generation – limit length to keep responses concise
        generated = self.generator(
            prompt,
            max_length=250,
            do_sample=False,
            temperature=0.0,
        )
        answer = generated[0]["generated_text"].strip()
        logger.info("Generated answer for query '%s'", query)
        return {"answer": answer, "sources": sources}

# Helper for script execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple CLI for the RAG pipeline.")
    parser.add_argument("query", type=str, help="User question")
    parser.add_argument("--category", type=str, default=None, help="Product category filter")
    parser.add_argument("--top_k", type=int, default=5, help="Number of retrieved chunks")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    result = pipeline.ask(args.query, product_category=args.category, top_k=args.top_k)
    print("Answer:\n", result["answer"], "\n")
    print("Sources:")
    for src in result["sources"]:
        print("-", src["metadata"], "| distance", round(src["distance"], 3))
