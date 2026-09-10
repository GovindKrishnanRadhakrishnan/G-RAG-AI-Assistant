"""
retriever.py
------------
AdvancedRetriever — the main orchestration pipeline for Advanced RAG.

Pipeline (executed for every user query):
  1. Query Rewriting    → make query retrieval-friendly
  2. Hybrid Search      → BM25 + ChromaDB dense search, fused with RRF
  3. Cross-Encoder Re-ranking → re-score top-10 with ms-marco cross-encoder
  4. Context Compression → strip irrelevant sentences from each chunk
  5. Answer Generation  → call Ollama with context + conversation memory
  6. Source Attribution → attach doc name + page to every answer
  7. Evaluation         → faithfulness + relevance scores

Usage:
    from sentence_transformers import SentenceTransformer

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    embed_fn    = lambda text: embed_model.encode(text).tolist()

    retriever = AdvancedRetriever(
        persist_directory="vector_db",
        embed_fn=embed_fn,
    )
    result = retriever.query("What is the attention mechanism in transformers?")
    print(result["answer"])
    print(result["sources"])
    print(result["evaluation"])
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ..ollama import OllamaClient, OllamaError
from .vector_store import VectorStore
from .hybrid_retriever import HybridRetriever
from .query_rewriter import QueryRewriter
from .reranker import CrossEncoderReranker
from .compressor import ContextCompressor
from .memory import ConversationMemory
from .evaluator import ResponseEvaluator


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """\
You are a helpful document assistant for G-RAG. Answer the user's question using
ONLY the information in the provided context. If the answer is not in the
context, say "I don't have enough information to answer that."

{history_section}
Context:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class AdvancedRetriever:
    """
    End-to-end Advanced RAG pipeline.

    Parameters
    ----------
    persist_directory : str
        Path to the ChromaDB persistence directory (default: "vector_db").
    embed_fn : Callable[[str], List[float]]
        Function mapping a text string to an embedding vector.
    ollama_host : str
        Base URL for the Ollama API.
    ollama_model : str
        Ollama model name for answer generation and evaluation.
    top_k_retrieval : int
        Chunks fetched by hybrid search before re-ranking.
    top_k_final : int
        Chunks retained after cross-encoder re-ranking.
    compression_threshold : float
        Cosine similarity cutoff for sentence-level compression.
    memory_turns : int
        Number of conversation turns to keep in memory.
    cross_encoder_model : str
        HuggingFace model name for the cross-encoder re-ranker.
    embed_model_name : str
        Model name passed to ContextCompressor's bi-encoder.
    run_evaluation : bool
        Whether to call the evaluator (adds ~2 LLM calls per query).
    """

    def __init__(
        self,
        persist_directory: str = "vector_db",
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        ollama_host: str = "http://localhost:11434",
        ollama_model: str = "phi3:mini" ,
        top_k_retrieval: int = 10,
        top_k_final: int = 5,
        compression_threshold: float = 0.30,
        memory_turns: int = 5,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        embed_model_name: str = "all-MiniLM-L6-v2",
        run_evaluation: bool = True,
    ) -> None:
        self.ollama_host = ollama_host.rstrip("/")
        self.ollama_model = ollama_model
        self.top_k_retrieval = top_k_retrieval
        self.top_k_final = top_k_final
        self.run_evaluation = run_evaluation

        # --- Sub-components ---
        self.vector_store = VectorStore(persist_directory)
        self.ollama = OllamaClient(host=self.ollama_host, model=ollama_model)

        # Default embed_fn: use sentence-transformers bi-encoder
        if embed_fn is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(embed_model_name)
            embed_fn = lambda text: _model.encode(text).tolist()

        self.embed_fn = embed_fn

        self.hybrid = HybridRetriever(self.vector_store, embed_fn, top_k=top_k_retrieval)
        self.rewriter = QueryRewriter(client=self.ollama)
        self.reranker = CrossEncoderReranker(cross_encoder_model)
        self.compressor = ContextCompressor(embed_model_name, compression_threshold)
        self.memory = ConversationMemory(memory_turns)
        self.evaluator = ResponseEvaluator(client=self.ollama)

    # ------------------------------------------------------------------
    # Document ingestion (delegates to VectorStore)
    # ------------------------------------------------------------------

    def ingest(self, documents: List[Dict]) -> None:
        """
        Ingest pre-processed document chunks into the vector store.

        Each item in *documents* must be the dict produced by DocumentProcessor:
            {"text": str, "metadata": {"chunk_hash": str, ...}}

        After ingestion the BM25 index is refreshed automatically.
        """
        embeddings = [self.embed_fn(doc["text"]) for doc in documents]
        self.vector_store.add_documents(documents, embeddings)
        self.hybrid.refresh_index()

    # ------------------------------------------------------------------
    # Query pipeline
    # ------------------------------------------------------------------

    def query(self, question: str) -> Dict:
        """
        Run the full Advanced RAG pipeline for *question*.

        Returns
        -------
        {
            "question"       : original question,
            "rewritten_query": query after rewriting,
            "answer"         : generated answer string,
            "sources"        : list of {source_file, page, snippet},
            "evaluation"     : {faithfulness: float, relevance: float},
            "chunks"         : list of final compressed chunks (for debugging),
        }
        """
        import time
        start_time = time.time()
        print(f"\n[RAG Pipeline] Starting query: '{question}'")

        # ---- 1. Query Rewriting ----
        t0 = time.time()
        print("[RAG Pipeline] Step 1/6: Rewriting query via Ollama...")
        rewritten = self.rewriter.rewrite(question)
        print(f"[RAG Pipeline] -> Rewritten to: '{rewritten}' (took {time.time() - t0:.2f}s)")

        # ---- 2. Hybrid Search (BM25 + Dense + RRF) ----
        retrieval_start = time.time()
        t0 = time.time()
        print("[RAG Pipeline] Step 2/6: Running Hybrid Search (BM25 + Dense ChromaDB)...")
        candidates = self.hybrid.retrieve(rewritten)
        print(f"[RAG Pipeline] -> Found {len(candidates)} candidate chunks (took {time.time() - t0:.2f}s)")

        if not candidates:
            print("[RAG Pipeline] -> No candidates found. Aborting pipeline.")
            return self._empty_result(question, rewritten)

        # ---- 3. Cross-Encoder Re-ranking ----
        t0 = time.time()
        print("[RAG Pipeline] Step 3/6: Re-ranking chunks with Cross-Encoder...")
        reranked = self.reranker.rerank(rewritten, candidates, top_n=self.top_k_final)
        print(f"[RAG Pipeline] -> Re-ranked to top-{len(reranked)} chunks (took {time.time() - t0:.2f}s)")

        # ---- 4. Context Compression ----
        t0 = time.time()
        print("[RAG Pipeline] Step 4/6: Compressing context sentences...")
        compressed = self.compressor.compress(rewritten, reranked)
        print(f"[RAG Pipeline] -> Context compression completed (took {time.time() - t0:.2f}s)")
        
        # Record RAG retrieval duration
        retrieval_duration = time.time() - retrieval_start
        from .metrics import RAG_RETRIEVAL_DURATION
        RAG_RETRIEVAL_DURATION.observe(retrieval_duration)

        # ---- 5. Build context string + sources ----
        context_parts: List[str] = []
        sources: List[Dict] = []

        for i, chunk in enumerate(compressed, start=1):
            meta = chunk.get("metadata", {})
            src_file = meta.get("source_file", "Unknown")
            page     = meta.get("page", "?")
            text     = chunk["text"]

            context_parts.append(f"[Source {i}] {text}")
            sources.append({
                "source_file": src_file,
                "page"       : page,
                "snippet"    : chunk.get("original_text", text)[:300],
            })

        context_str = "\n\n".join(context_parts)

        # ---- 6. Conversation memory ----
        history_str = self.memory.get_context_string()
        history_section = (
            f"Conversation history:\n{history_str}\n\n" if history_str else ""
        )

        # ---- 7. Answer generation via Ollama ----
        t0 = time.time()
        print("[RAG Pipeline] Step 5/6: Generating answer via Ollama...")
        prompt = ANSWER_PROMPT.format(
            history_section=history_section,
            context=context_str,
            question=question,
        )
        answer = self._call_ollama(prompt)
        
        # Record LLM inference duration
        inference_duration = time.time() - t0
        print(f"[RAG Pipeline] -> Answer generated (took {inference_duration:.2f}s)")
        from .metrics import LLM_INFERENCE_DURATION
        LLM_INFERENCE_DURATION.observe(inference_duration)

        # ---- 8. Store turn in memory ----
        self.memory.add_exchange(question, answer)

        # ---- 9. Evaluation ----
        evaluation = {"faithfulness": 0.5, "relevance": 0.5}
        if self.run_evaluation:
            t0 = time.time()
            print("[RAG Pipeline] Step 6/6: Evaluating faithfulness & relevance via Ollama...")
            evaluation = self.evaluator.evaluate(question, answer, context_str)
            print(f"[RAG Pipeline] -> Evaluation complete (took {time.time() - t0:.2f}s)")

        print(f"[RAG Pipeline] Total pipeline execution time: {time.time() - start_time:.2f}s\n")

        return {
            "question"       : question,
            "rewritten_query": rewritten,
            "answer"         : answer,
            "sources"        : sources,
            "evaluation"     : evaluation,
            "chunks"         : compressed,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str, timeout: int = 120) -> str:
        """Send a prompt to Ollama and return the generated text."""
        try:
            return self.ollama.generate(
                prompt,
                temperature=0.7,
                num_predict=512,
                timeout=timeout,
            )
        except OllamaError as exc:
            return f"[Error calling Ollama: {exc}]"

    def _empty_result(self, question: str, rewritten: str) -> Dict:
        """Return a safe empty result when no documents are in the store."""
        return {
            "question"       : question,
            "rewritten_query": rewritten,
            "answer"         : "No documents are ready yet. Please upload a PDF and process it first.",
            "sources"        : [],
            "evaluation"     : {"faithfulness": 0.0, "relevance": 0.0},
            "chunks"         : [],
        }

    def clear_memory(self) -> None:
        """Reset conversation history."""
        self.memory.clear()
