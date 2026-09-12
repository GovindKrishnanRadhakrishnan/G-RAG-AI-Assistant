"""
memory.py
---------
Lightweight conversation memory that stores the last N exchanges
(user + assistant pairs) and formats them for injection into LLM prompts.

Usage:
    memory = ConversationMemory(max_turns=5)
    memory.add_exchange("What is BERT?", "BERT is a transformer model...")
    memory.add_exchange("How was it trained?", "It was trained with MLM...")

    print(memory.get_context_string())
    # Human: What is BERT?
    # Assistant: BERT is a transformer model...
    # Human: How was it trained?
    # Assistant: It was trained with MLM...
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List


class ConversationMemory:
    """
    Ring-buffer conversation history.

    Stores up to *max_turns* complete exchanges (one exchange = one
    user message + one assistant message). Older exchanges are dropped
    automatically once the buffer is full.

    Parameters
    ----------
    max_turns : int
        Maximum number of (user, assistant) pairs to retain.
    """

    def __init__(self, max_turns: int = 5) -> None:
        self.max_turns = max_turns
        # Each element: {"role": "human"|"assistant", "content": str}
        self._history: Deque[Dict[str, str]] = deque(maxlen=max_turns * 2)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_exchange(self, user_message: str, assistant_message: str) -> None:
        """Append one complete exchange to the history."""
        self._history.append({"role": "human", "content": user_message.strip()})
        self._history.append({"role": "assistant", "content": assistant_message.strip()})

    def clear(self) -> None:
        """Wipe all stored history."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_history(self) -> List[Dict[str, str]]:
        """Return the full history as a list of role/content dicts."""
        return list(self._history)

    def get_context_string(self) -> str:
        """
        Format history as a human-readable dialogue string suitable for
        injection into an LLM prompt.

        Example output:
            Human: What is BERT?
            Assistant: BERT is a pre-trained transformer...
        """
        if not self._history:
            return ""

        lines: List[str] = []
        for msg in self._history:
            label = "Human" if msg["role"] == "human" else "Assistant"
            lines.append(f"{label}: {msg['content']}")
        return "\n".join(lines)

    def get_last_n_exchanges(self, n: int) -> "ConversationMemory":
        """Return a new ConversationMemory containing only the last *n* exchanges."""
        sub = ConversationMemory(max_turns=n)
        recent = list(self._history)[-(n * 2):]
        for i in range(0, len(recent) - 1, 2):
            sub.add_exchange(recent[i]["content"], recent[i + 1]["content"])
        return sub

    def is_empty(self) -> bool:
        """True if no exchanges have been stored yet."""
        return len(self._history) == 0

    def __len__(self) -> int:
        """Number of individual messages (turns × 2)."""
        return len(self._history)
