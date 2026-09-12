"""
reranker.py
-----------
Cross-encoder re-ranking using the sentence-transformers library.

After hybrid retrieval returns ~10 candidate chunks, this module
scores every (query, chunk) pair with a cross-encoder and returns
the top-N chunks by that score.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~85 MB, downloaded from HuggingFace on first use
  - Trained on MS MARCO passage ranking, strong general-purpose re-ranker

Usage:
    reranker = CrossEncoderReranker()
    top5 = reranker.rerank(query="What is BERT?", chunks=candidates, top_n=5)
"""

from __future__ import annotations

from typing import List, Dict

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Re-ranks a list of retrieved chunks using a cross-encoder model.

    Parameters
    ----------
    model_name : str
        Any sentence-transformers CrossEncoder model name.
    device : str | None
        PyTorch device string ("cpu", "cuda", …).
        None lets sentence-transformers pick automatically.
    lazy : bool
        If True, delay model load until the first rerank() call.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        lazy: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: CrossEncoder | None = None
        if not lazy:
            self._model = CrossEncoder(model_name, device=device)

    def _ensure_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_n: int = 5,
    ) -> List[Dict]:
        """
        Score every (query, chunk_text) pair, sort descending, return top_n.

        Parameters
        ----------
        query  : The user query (or rewritten query).
        chunks : List of chunk dicts — each must have a "text" key.
        top_n  : Number of chunks to return after re-ranking.

        Returns
        -------
        List of chunk dicts with an added key ``cross_encoder_score``.
        """
        if not chunks:
            return []

        # Build input pairs for the cross-encoder
        pairs = [(query, chunk["text"]) for chunk in chunks]

        # Score all pairs in a single batched forward pass
        scores = self._ensure_model().predict(pairs, show_progress_bar=False)

        # Attach score to each chunk and sort
        scored = sorted(
            zip(scores, chunks),
            key=lambda x: float(x[0]),
            reverse=True,
        )

        return [
            {**chunk, "cross_encoder_score": float(score)}
            for score, chunk in scored[:top_n]
        ]
