"""
Groq LLM interface for TopoRAG.

Groq provides fast, free-tier inference for llama-3.1-8b-instant and similar models.
Used for speculative query generation (never for evaluation).

Free tier limits (as of 2025):
  llama-3.1-8b-instant  : 6000 req/day, 30 req/min
  llama-3.3-70b-versatile: 1000 req/day, 30 req/min
"""

import os
import time
import logging
from typing import Optional, List, Dict

from .base import BaseLLM

logger = logging.getLogger(__name__)


class GroqLLM(BaseLLM):
    """
    Groq API client.

    Args:
        model_name: Groq model (default: llama-3.1-8b-instant — free, fast, good quality)
        temperature: Sampling temperature
        max_tokens: Max tokens in response
        retry: Number of retries on rate-limit / transient errors
        api_key: Groq API key (defaults to GROQ_API_KEY env var)
    """

    RATE_LIMIT_PAUSE = 2.5  # seconds between requests (safe for 30 req/min limit)

    def __init__(
        self,
        model_name: str = "llama-3.1-8b-instant",
        temperature: float = 0.3,
        max_tokens: int = 256,
        retry: int = 4,
        api_key: Optional[str] = None,
    ):
        super().__init__(model_name, temperature, max_tokens, retry)

        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY in .env or pass api_key=..."
            )

        try:
            from groq import Groq
        except ImportError:
            raise ImportError("Run: pip install groq")

        self.client = Groq(api_key=self.api_key)
        self._last_call = 0.0

    def _wait_rate_limit(self):
        """Ensure minimum gap between API calls."""
        elapsed = time.time() - self._last_call
        if elapsed < self.RATE_LIMIT_PAUSE:
            time.sleep(self.RATE_LIMIT_PAUSE - elapsed)
        self._last_call = time.time()

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.generate_with_messages(messages)

    def generate_with_messages(self, messages: List[Dict[str, str]]) -> str:
        last_error = None
        for attempt in range(self.retry + 1):
            try:
                self._wait_rate_limit()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                wait = min(60, 5 * (attempt + 1))
                logger.warning(f"Groq error (attempt {attempt + 1}/{self.retry + 1}): {e}. Waiting {wait}s...")
                time.sleep(wait)

        raise RuntimeError(f"Groq API failed after {self.retry + 1} attempts: {last_error}")
