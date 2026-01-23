"""
OpenAI LLM interface for TopoRAG.

This replicates the GFM-RAG setup which uses GPT-4o-mini for:
1. Answer generation (QA evaluation)
2. OpenIE extraction (KG construction) - not used in TopoRAG

GFM-RAG settings:
- Model: gpt-4o-mini
- Temperature: 0.0 (deterministic)
- Timeout: 60 seconds
"""

import os
import time
import logging
from typing import List, Dict, Optional

from .base import BaseLLM

logger = logging.getLogger(__name__)


class OpenAILLM(BaseLLM):
    """
    OpenAI ChatGPT interface.

    Follows GFM-RAG implementation for consistency.

    Args:
        model_name: Model name (default: gpt-4o-mini as in GFM-RAG)
        temperature: Sampling temperature (default: 0.0 for deterministic)
        max_tokens: Maximum tokens to generate
        retry: Number of retries on failure
        timeout: API timeout in seconds
    """

    # Token limits by model
    TOKEN_LIMITS = {
        "gpt-4o-mini": 128000,
        "gpt-4o": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16384,
    }

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        retry: int = 3,
        timeout: int = 60,
        api_key: Optional[str] = None,
    ):
        super().__init__(model_name, temperature, max_tokens, retry)
        self.timeout = timeout

        # Get API key from environment if not provided
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Import here to avoid dependency issues
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)

        self.max_context_tokens = self.TOKEN_LIMITS.get(model_name, 128000)

    def _count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text from a prompt."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        return self.generate_with_messages(messages)

    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """
        Generate text from messages.

        Follows GFM-RAG implementation with retry logic.
        """
        # Check token limit
        total_content = "\n".join(m["content"] for m in messages)
        if self._count_tokens(total_content) > self.max_context_tokens:
            logger.warning(
                f"Input may exceed token limit ({self.max_context_tokens}). "
                "Consider truncating."
            )

        last_error = None
        for attempt in range(self.retry + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=self.timeout,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                last_error = e
                logger.warning(f"OpenAI API error (attempt {attempt + 1}): {e}")
                if attempt < self.retry:
                    time.sleep(min(30, 2 ** attempt))  # Exponential backoff

        raise RuntimeError(f"OpenAI API failed after {self.retry + 1} attempts: {last_error}")
