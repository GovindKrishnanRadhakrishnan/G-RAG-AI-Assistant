"""
query_rewriter.py
-----------------
Uses the local Ollama LLM to rewrite a user query before retrieval,
making it more specific, keyword-rich, and retrieval-friendly.

If Ollama is unavailable or returns an empty response the original
query is returned unchanged (graceful degradation).
"""

from __future__ import annotations

import re
from typing import Optional

from ..ollama import OllamaClient, OllamaError


REWRITE_PROMPT = """\
You are a search query optimisation assistant.

Your task is to rewrite the user's question into a precise, keyword-rich
search query that will help retrieve the most relevant text chunks from a
document database.

Rules:
1. Keep the core intent of the original question.
2. Expand abbreviations where helpful.
3. Add relevant technical synonyms if appropriate.
4. Output ONLY the rewritten query — no explanations, no preamble.

Original question: {query}

Rewritten query:"""


class QueryRewriter:
    """Rewrites a user query using the local Ollama LLM."""

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "llama3.2:1b",
        timeout: int = 30,
        client: Optional[OllamaClient] = None,
    ) -> None:
        self.client = client or OllamaClient(
            host=ollama_host, model=model, timeout=timeout
        )

    def rewrite(self, query: str) -> str:
        """Return a retrieval-optimised version of *query*."""
        if not query or not query.strip():
            return query

        original = query.strip()
        # Short questions (e.g. "who is govind") are already good search queries.
        # Rewriting them with a small local model often drops the key name.
        if len(original.split()) <= 6:
            return original

        prompt = REWRITE_PROMPT.format(query=original)

        try:
            rewritten = self.client.generate(
                prompt,
                temperature=0.3,
                num_predict=100,
                timeout=self.client.timeout,
            )
            if rewritten and len(rewritten) < 500:
                rewritten = re.sub(
                    r"^(rewritten query\s*[:：]\s*)",
                    "",
                    rewritten,
                    flags=re.IGNORECASE,
                ).strip()
                if not rewritten:
                    return original

                # If the rewrite drops distinctive tokens from the original, keep original
                orig_tokens = set(re.findall(r"[A-Za-z0-9]{3,}", original.lower()))
                rew_tokens = set(re.findall(r"[A-Za-z0-9]{3,}", rewritten.lower()))
                if orig_tokens - rew_tokens:
                    return original
                return rewritten
        except OllamaError:
            pass

        return original
