"""
compressor.py
-------------
Context compression: given a retrieved chunk, keep only the sentences
that are semantically relevant to the query.

Strategy:
  1. Split chunk text into sentences (using a simple regex).
  2. Encode {query} and each sentence with a bi-encoder (SentenceTransformer).
  3. Compute cosine similarity between the query embedding and each sentence.
  4. Keep sentences whose similarity >= threshold (default 0.30).
  5. If nothing survives, return the first 3 sentences as a fallback.

Usage:
    compressor = ContextCompressor(threshold=0.3)
    result = compressor.compress(query="What is BERT?",
                                 chunks=reranked_chunks)
"""

from __future__ import annotations

import re
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using punctuation boundaries.
    Handles common abbreviations roughly — good enough for paragraph chunks.
    """
    # Split on . ! ? followed by whitespace + capital letter or end-of-string
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'(])", text)
    # Filter out blank/too-short fragments
    return [p.strip() for p in parts if len(p.strip()) > 15]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D numpy arrays."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class ContextCompressor:
    """
    Removes irrelevant sentences from retrieved chunks.

    Parameters
    ----------
    model_name : str
        Bi-encoder model for sentence embeddings.
        "all-MiniLM-L6-v2" is fast (~80 MB) and accurate enough.
    threshold : float
        Minimum cosine similarity to keep a sentence (0 – 1).
    model : SentenceTransformer | None
        Optional shared embedding model (avoids loading a second copy).
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.30,
        model: SentenceTransformer | None = None,
    ) -> None:
        self.threshold = threshold
        self._model = model if model is not None else SentenceTransformer(model_name)

    def _compress_text(self, query: str, text: str) -> str:
        """Return a compressed version of *text* keeping only relevant sentences."""
        sentences = _split_sentences(text)
        if not sentences:
            return text  # Nothing to split — return as-is

        # Encode query + all sentences in one pass
        all_texts = [query] + sentences
        embeddings = self._model.encode(all_texts, convert_to_numpy=True)
        query_emb = embeddings[0]
        sent_embs = embeddings[1:]

        # Score each sentence
        scores = [_cosine_similarity(query_emb, s) for s in sent_embs]

        # Keep sentences above threshold
        kept = [s for s, sc in zip(sentences, scores) if sc >= self.threshold]

        # Fallback: if nothing kept, use first 3 sentences
        if not kept:
            kept = sentences[:3]

        return " ".join(kept)

    def compress(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """
        Compress every chunk in *chunks* for the given *query*.

        Returns a new list of chunk dicts with the "text" field replaced
        by the compressed version. Metadata and scores are preserved.
        """
        compressed = []
        for chunk in chunks:
            original_text = chunk.get("text", "")
            # Short chunks (typical CV/resume sections) should stay intact.
            if len(original_text) < 700:
                compressed_text = original_text
            else:
                compressed_text = self._compress_text(query, original_text)
            compressed.append({
                **chunk,
                "text": compressed_text,
                "original_text": original_text,  # keep for source display
            })
        return compressed
