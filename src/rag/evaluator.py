"""
evaluator.py
------------
Lightweight evaluation of RAG responses using the local Ollama LLM.

Produces two scores per response:
  - faithfulness  : Is the answer grounded in the retrieved context?  (0.0 – 1.0)
  - relevance     : Does the answer address the user's question?       (0.0 – 1.0)
"""

from __future__ import annotations

import re
from typing import Dict, Optional

from ..ollama import OllamaClient, OllamaError


FAITHFULNESS_PROMPT = """\
You are an impartial evaluator. Your task is to judge whether the given answer
is fully supported by the provided context — i.e., does the answer contain only
information that can be found or inferred from the context?

Context:
{context}

Answer:
{answer}

Rate the faithfulness of the answer on a scale from 0.0 (completely unfaithful /
hallucinated) to 1.0 (entirely supported by context).

Output ONLY a single decimal number between 0.0 and 1.0, nothing else.
"""

RELEVANCE_PROMPT = """\
You are an impartial evaluator. Your task is to judge how well the given answer
addresses the user's question.

Question: {question}

Answer:
{answer}

Rate the relevance of the answer on a scale from 0.0 (completely irrelevant) to
1.0 (perfectly answers the question).

Output ONLY a single decimal number between 0.0 and 1.0, nothing else.
"""


class ResponseEvaluator:
    """Scores an LLM response for faithfulness and relevance."""

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "llama3.2:1b",
        timeout: int = 60,
        client: Optional[OllamaClient] = None,
    ) -> None:
        self.client = client or OllamaClient(
            host=ollama_host, model=model, timeout=timeout
        )

    def _parse_score(self, text: str, fallback: float = 0.5) -> float:
        """Extract the first float in [0, 1] from *text*."""
        match = re.search(r"\b(0(\.\d+)?|1(\.0+)?)\b", text)
        if match:
            score = float(match.group())
            return max(0.0, min(1.0, score))
        return fallback

    def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
        max_context_chars: int = 3000,
    ) -> Dict[str, float]:
        """Evaluate an answer against the question and retrieved context."""
        ctx = context[:max_context_chars] if len(context) > max_context_chars else context

        faithfulness_score = 0.5
        relevance_score = 0.5

        try:
            faith_raw = self.client.generate(
                FAITHFULNESS_PROMPT.format(context=ctx, answer=answer),
                temperature=0.0,
                num_predict=10,
            )
            faithfulness_score = self._parse_score(faith_raw)
        except OllamaError:
            pass

        try:
            rel_raw = self.client.generate(
                RELEVANCE_PROMPT.format(question=question, answer=answer),
                temperature=0.0,
                num_predict=10,
            )
            relevance_score = self._parse_score(rel_raw)
        except OllamaError:
            pass

        return {
            "faithfulness": round(faithfulness_score, 2),
            "relevance": round(relevance_score, 2),
        }
