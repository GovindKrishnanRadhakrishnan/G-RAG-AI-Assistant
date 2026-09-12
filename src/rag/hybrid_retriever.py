"""
hybrid_retriever.py
-------------------
Combines BM25 keyword search with ChromaDB dense-vector search,
using Reciprocal Rank Fusion (RRF) to merge the two ranked lists.

Algorithm (RRF):
    score(doc) = sum( 1 / (k + rank_in_list) )   for each list
    where k = 60  (standard constant that dampens high-rank bias)

Usage:
    store   = VectorStore("vector_db")
    hybrid  = HybridRetriever(store, embed_fn=my_embed_fn, top_k=10)
    results = hybrid.retrieve(query="What is BERT?")
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Callable, List, Dict

from rank_bm25 import BM25Okapi

from .vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Common English function words that drown out names in short questions
# like "who is govind".
_BM25_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for",
    "of", "as", "by", "with", "from", "into", "about", "is", "are", "was",
    "were", "be", "been", "being", "am", "do", "does", "did", "doing", "have",
    "has", "had", "having", "who", "whom", "whose", "what", "which", "when",
    "where", "why", "how", "this", "that", "these", "those", "it", "its",
    "he", "she", "they", "them", "their", "we", "you", "your", "i", "me",
    "my", "our", "can", "could", "should", "would", "will", "just", "also",
    "than", "then", "there", "here", "not", "no", "yes",
}


def _tokenize(text: str) -> List[str]:
    """Lowercase whitespace-split tokeniser for BM25, with stopword removal."""
    tokens = re.sub(r"[^\w\s]", "", text.lower()).split()
    kept = [t for t in tokens if t not in _BM25_STOPWORDS and len(t) > 1]
    # If everything was a stopword, fall back to raw tokens so search still runs
    return kept if kept else tokens


# ---------------------------------------------------------------------------
# BM25 retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Wraps rank_bm25.BM25Okapi around a list of document chunks.

    The index is rebuilt every time `build_index` is called so it stays
    in sync with fresh documents ingested into ChromaDB.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._docs: List[Dict] = []

    def build_index(self, documents: List[Dict]) -> None:
        """Build (or rebuild) the BM25 index from a list of chunk dicts."""
        self._docs = documents
        corpus = [_tokenize(doc["text"]) for doc in documents]
        if corpus:
            self._bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Return up to top_k chunks ranked by BM25 score.
        Each result has: text, metadata, bm25_score.
        """
        if self._bm25 is None or not self._docs:
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        ranked = sorted(
            zip(scores, self._docs),
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        return [
            {**doc, "bm25_score": float(score)}
            for score, doc in ranked
            if score > 0  # skip zero-score docs
        ]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict]],
    id_key: str = "text",
    k: int = 60,
) -> List[Dict]:
    """
    Merge multiple ranked result lists with RRF.

    Args:
        ranked_lists : list of ranked lists, each list ordered best-first.
        id_key       : dict key used as the unique document identifier.
        k            : RRF constant (default 60, per the original paper).

    Returns:
        A single list of dicts sorted by fused RRF score (descending).
        Each dict has an extra key ``rrf_score``.
    """
    rrf_scores: Dict[str, float] = defaultdict(float)
    doc_store: Dict[str, Dict] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            uid = doc[id_key]
            rrf_scores[uid] += 1.0 / (k + rank)
            if uid not in doc_store:
                doc_store[uid] = doc

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [{**doc_store[uid], "rrf_score": score} for uid, score in fused]


# ---------------------------------------------------------------------------
# Hybrid retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """
    Orchestrates BM25 + ChromaDB dense search and merges results via RRF.

    Parameters
    ----------
    vector_store : VectorStore
        The persistent ChromaDB store.
    embed_fn : Callable[[str], List[float]]
        Any function that turns a text string into an embedding vector.
        Typically a SentenceTransformer or Ollama embedding call.
    top_k : int
        Number of results to fetch from each sub-retriever before fusion.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embed_fn: Callable[[str], List[float]],
        top_k: int = 10,
    ) -> None:
        self.vector_store = vector_store
        self.embed_fn = embed_fn
        self.top_k = top_k
        self.bm25 = BM25Retriever()
        self._index_built = False

    def _ensure_index(self) -> None:
        """Lazily build the BM25 index from all docs currently in ChromaDB."""
        if not self._index_built:
            all_docs = self.vector_store.get_all_documents()
            self.bm25.build_index(all_docs)
            self._index_built = True

    def refresh_index(self) -> None:
        """Force-rebuild the BM25 index (call after ingesting new documents)."""
        all_docs = self.vector_store.get_all_documents()
        self.bm25.build_index(all_docs)
        self._index_built = True

    def retrieve(self, query: str) -> List[Dict]:
        """
        Run hybrid retrieval for *query* and return fused RRF-ranked results.

        Returns a list of chunk dicts, each containing:
            text, metadata, rrf_score
        """
        # 1. Ensure BM25 index is ready
        self._ensure_index()

        # 2. BM25 keyword search
        bm25_results = self.bm25.retrieve(query, top_k=self.top_k)

        # 3. Dense vector search (ChromaDB)
        query_embedding = self.embed_fn(query)
        dense_results = self.vector_store.distance_search(
            query_embedding, top_k=self.top_k
        )

        # 4. Fuse with RRF
        fused = reciprocal_rank_fusion(
            [bm25_results, dense_results],
            id_key="text",
            k=60,
        )

        return fused[: self.top_k]
