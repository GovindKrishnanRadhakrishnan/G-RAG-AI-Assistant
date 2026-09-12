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

import re
from pathlib import Path
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
Use only the context to answer the question.
Give a short factual answer. Include useful details from the context when available (full name, role, location, skills, experience).
If the context does not contain the answer, say: I don't have enough information to answer that.

{history_section}Context:
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
        ollama_model: str = "llama3.2:1b",
        top_k_retrieval: int = 6,
        top_k_final: int = 3,
        compression_threshold: float = 0.30,
        memory_turns: int = 5,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        embed_model_name: str = "all-MiniLM-L6-v2",
        run_evaluation: bool = False,
        enable_query_rewrite: bool = False,
        enable_rerank: bool = False,
        enable_compression: bool = False,
        llm_num_predict: int = 192,
        ollama_keep_alive: str = "10m",
    ) -> None:
        self.ollama_host = ollama_host.rstrip("/")
        self.ollama_model = ollama_model
        self.top_k_retrieval = top_k_retrieval
        self.top_k_final = top_k_final
        self.run_evaluation = run_evaluation
        self.enable_query_rewrite = enable_query_rewrite
        self.enable_rerank = enable_rerank
        self.enable_compression = enable_compression
        self.llm_num_predict = llm_num_predict

        # --- Sub-components ---
        self.vector_store = VectorStore(persist_directory)
        self.ollama = OllamaClient(
            host=self.ollama_host,
            model=ollama_model,
            keep_alive=ollama_keep_alive,
        )

        # Single shared bi-encoder for embeddings (+ optional compression)
        self._embed_model = None
        if embed_fn is None:
            from sentence_transformers import SentenceTransformer

            self._embed_model = SentenceTransformer(embed_model_name)
            embed_fn = lambda text: self._embed_model.encode(text).tolist()

        self.embed_fn = embed_fn

        self.hybrid = HybridRetriever(self.vector_store, embed_fn, top_k=top_k_retrieval)
        self.rewriter = QueryRewriter(client=self.ollama)
        # Lazy-load cross-encoder only when re-ranking is enabled
        self.reranker = (
            CrossEncoderReranker(cross_encoder_model, lazy=True)
            if enable_rerank
            else None
        )
        self.compressor = (
            ContextCompressor(
                embed_model_name,
                compression_threshold,
                model=self._embed_model,
            )
            if enable_compression
            else None
        )
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
        texts = [doc["text"] for doc in documents]
        if self._embed_model is not None:
            # Batch encode — much faster than one-by-one
            vectors = self._embed_model.encode(texts, show_progress_bar=False)
            embeddings = [v.tolist() for v in vectors]
        else:
            embeddings = [self.embed_fn(text) for text in texts]
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

        # ---- 1. Query Rewriting (optional — costs an extra Ollama call) ----
        t0 = time.time()
        if self.enable_query_rewrite:
            print("[RAG Pipeline] Step 1/6: Rewriting query via Ollama...")
            rewritten = self.rewriter.rewrite(question)
        else:
            print("[RAG Pipeline] Step 1/6: Query rewrite skipped")
            rewritten = question
        retrieval_query = self._retrieval_query(question, rewritten)
        print(
            f"[RAG Pipeline] -> Rewritten to: '{rewritten}' | "
            f"retrieval query: '{retrieval_query}' (took {time.time() - t0:.2f}s)"
        )

        # ---- 2. Hybrid Search (BM25 + Dense + RRF) ----
        retrieval_start = time.time()
        t0 = time.time()
        print("[RAG Pipeline] Step 2/6: Running Hybrid Search (BM25 + Dense ChromaDB)...")
        candidates = self.hybrid.retrieve(retrieval_query)
        print(f"[RAG Pipeline] -> Found {len(candidates)} candidate chunks (took {time.time() - t0:.2f}s)")

        if not candidates:
            print("[RAG Pipeline] -> No candidates found. Aborting pipeline.")
            return self._empty_result(question, rewritten)

        # ---- 3. Cross-Encoder Re-ranking (optional) ----
        t0 = time.time()
        if self.enable_rerank and self.reranker is not None:
            print("[RAG Pipeline] Step 3/6: Re-ranking chunks with Cross-Encoder...")
            reranked = self.reranker.rerank(question, candidates, top_n=self.top_k_final)
        else:
            print("[RAG Pipeline] Step 3/6: Fast lexical selection (re-rank off)")
            reranked = self._fast_select(question, candidates, self.top_k_final)
        print(f"[RAG Pipeline] -> Kept top-{len(reranked)} chunks (took {time.time() - t0:.2f}s)")

        # ---- 4. Context Compression (optional) ----
        t0 = time.time()
        if self.enable_compression and self.compressor is not None:
            print("[RAG Pipeline] Step 4/6: Compressing context sentences...")
            compressed = self.compressor.compress(question, reranked)
        else:
            print("[RAG Pipeline] Step 4/6: Compression skipped")
            compressed = [
                {**chunk, "original_text": chunk.get("text", "")}
                for chunk in reranked
            ]
        print(f"[RAG Pipeline] -> Context ready (took {time.time() - t0:.2f}s)")
        
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
            src_name = Path(str(src_file)).name

            context_parts.append(f"[Source {i} | {src_name} | page {page}]\n{text}")
            sources.append({
                "source_file": src_file,
                "page"       : page,
                "snippet"    : chunk.get("original_text", text)[:300],
            })

        context_str = "\n\n".join(context_parts)

        # ---- 6. Conversation memory ----
        # Skip history for short questions — small models get confused and
        # ignore the retrieved document context.
        history_str = self.memory.get_context_string()
        use_history = bool(history_str) and len(question.split()) > 6
        history_section = (
            f"Conversation history:\n{history_str}\n\n" if use_history else ""
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
        evaluation: Dict = {}
        if self.run_evaluation:
            t0 = time.time()
            print("[RAG Pipeline] Step 6/6: Evaluating faithfulness & relevance via Ollama...")
            evaluation = self.evaluator.evaluate(question, answer, context_str)
            print(f"[RAG Pipeline] -> Evaluation complete (took {time.time() - t0:.2f}s)")
        else:
            print("[RAG Pipeline] Step 6/6: Evaluation skipped (RUN_EVALUATION=false)")

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
                temperature=0.2,
                num_predict=self.llm_num_predict,
                timeout=timeout,
            )
        except OllamaError as exc:
            msg = str(exc)
            if "system memory" in msg.lower() or "memory" in msg.lower():
                return (
                    "Ollama does not have enough free memory to run the configured model. "
                    f"Details: {msg}. "
                    "Close other apps, or set OLLAMA_MODEL to a smaller model "
                    "(for example tinyllama) in your .env file, then restart the app."
                )
            return (
                f"We couldn't generate an answer with Ollama ({self.ollama_model}). "
                f"Details: {msg}"
            )

    @staticmethod
    def _fast_select(question: str, candidates: List[Dict], top_n: int) -> List[Dict]:
        """
        Cheap substitute for cross-encoder re-ranking.

        Boosts chunks (and filenames) that contain the user's keywords so
        short name queries like 'who is govind' prefer the right CV.
        """
        tokens = [
            t
            for t in re.findall(r"[a-z0-9]{3,}", (question or "").lower())
            if t
            not in {
                "who",
                "what",
                "where",
                "when",
                "why",
                "how",
                "the",
                "and",
                "for",
                "are",
                "was",
                "with",
                "about",
            }
        ]

        def score(chunk: Dict) -> tuple:
            text = (chunk.get("text") or "").lower()
            fname = str(chunk.get("metadata", {}).get("source_file", "")).lower()
            text_hits = sum(1 for t in tokens if t in text)
            file_hits = sum(2 for t in tokens if t in fname)
            rrf = float(chunk.get("rrf_score") or 0.0)
            return (text_hits + file_hits, rrf)

        return sorted(candidates, key=score, reverse=True)[:top_n]

    @staticmethod
    def _retrieval_query(question: str, rewritten: str) -> str:
        """
        Prefer a retrieval query that keeps the user's original words.

        Small LLMs often rewrite short questions poorly (dropping names like
        "govind"), which breaks BM25 keyword matching.
        """
        q = (question or "").strip()
        rw = (rewritten or "").strip()
        if not rw or rw.lower() == q.lower():
            return q

        # Short / name-like questions: always keep the original phrasing first
        if len(q.split()) <= 8:
            return f"{q} {rw}"

        orig_tokens = set(re.findall(r"[A-Za-z0-9]{2,}", q.lower()))
        rew_tokens = set(re.findall(r"[A-Za-z0-9]{2,}", rw.lower()))
        dropped = orig_tokens - rew_tokens
        if dropped:
            return f"{q} {rw}"
        return rw

    def _empty_result(self, question: str, rewritten: str) -> Dict:
        """Return a safe empty result when no documents are in the store."""
        return {
            "question"       : question,
            "rewritten_query": rewritten,
            "answer"         : "No documents are ready yet. Please upload a PDF and process it first.",
            "sources"        : [],
            "evaluation"     : {},
            "chunks"         : [],
        }

    def clear_memory(self) -> None:
        """Reset conversation history."""
        self.memory.clear()

    def reset_documents(self) -> None:
        """Delete every stored document chunk and clear conversation memory."""
        self.vector_store.reset()
        self.hybrid.refresh_index()
        self.memory.clear()
