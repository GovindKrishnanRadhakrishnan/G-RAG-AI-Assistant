"""Thin Ollama HTTP client used by rewrite, generation, and evaluation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests


class OllamaError(Exception):
    """Raised when Ollama cannot be reached or returns an unexpected response."""


class OllamaClient:
    """Minimal wrapper around Ollama's `/api/generate` endpoint."""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "phi3:mini",
        timeout: int = 120,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        num_predict: int = 512,
        timeout: Optional[int] = None,
    ) -> str:
        """Send a prompt to Ollama and return the generated text."""
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": num_predict,
                    },
                },
                timeout=timeout if timeout is not None else self.timeout,
            )
            response.raise_for_status()
            data: Dict[str, Any] = response.json()
            return str(data.get("response", "")).strip()
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc

    def is_reachable(self, timeout: float = 2.0) -> bool:
        """Return True if the Ollama host responds to `/api/tags`."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=timeout)
            return response.status_code == 200
        except requests.RequestException:
            return False
