"""
Groq LLM interface for TopoRAG.

Groq provides FREE API access to fast LLM inference:
- llama-3.3-70b-versatile (recommended)
- llama-3.1-8b-instant (faster)
- mixtral-8x7b-32768
- gemma2-9b-it

Get free API key at: https://console.groq.com/

This is a cost-effective alternative to OpenAI for development/testing.
"""

import os
import time
import logging
from typing import List, Dict, Optional

from .base import BaseLLM

logger = logging.getLogger(__name__)


class GroqLLM(BaseLLM):
    """
    Groq LLM interface.

    Groq offers free API access with rate limits.
    Excellent for development and testing TopoRAG.

    Args:
        model_name: Model to use (default: llama-3.1-70b-versatile)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        retry: Number of retries on failure
        api_key: Groq API key (or set GROQ_API_KEY env var)

    Available models:
        - llama-3.3-70b-versatile: Best quality (recommended)
        - llama-3.1-8b-instant: Faster, good for testing
        - mixtral-8x7b-32768: Good balance
        - gemma2-9b-it: Google's model
    """

    AVAILABLE_MODELS = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-3.2-90b-vision-preview",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ]

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        retry: int = 3,
        api_key: Optional[str] = None,
    ):
        super().__init__(model_name, temperature, max_tokens, retry)

        # Get API key
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key required. Get a FREE key at https://console.groq.com/ "
                "and set GROQ_API_KEY environment variable or pass api_key parameter."
            )

        # Import Groq client
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "Groq package not installed. Install with: pip install groq"
            )

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
        """Generate text from messages."""
        last_error = None

        for attempt in range(self.retry + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Handle rate limiting
                if "rate_limit" in error_str or "429" in error_str:
                    wait_time = min(60, 10 * (attempt + 1))
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"Groq API error (attempt {attempt + 1}): {e}")
                    if attempt < self.retry:
                        time.sleep(2 ** attempt)

        raise RuntimeError(f"Groq API failed after {self.retry + 1} attempts: {last_error}")


class GroqBatchLLM(GroqLLM):
    """
    Groq LLM with batching support for efficient query generation.

    Handles rate limits by batching requests with delays.
    """

    def __init__(
        self,
        model_name: str = "llama-3.1-8b-instant",  # Faster model for batch
        batch_delay: float = 1.0,  # Delay between requests (rate limit)
        **kwargs,
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.batch_delay = batch_delay

    def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        show_progress: bool = True,
    ) -> List[str]:
        """
        Generate responses for multiple prompts.

        Args:
            prompts: List of prompts
            system_prompt: Shared system prompt
            show_progress: Whether to show progress

        Returns:
            List of generated responses
        """
        responses = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(prompts, desc="Generating")
            except ImportError:
                iterator = prompts
        else:
            iterator = prompts

        for prompt in iterator:
            try:
                response = self.generate(prompt, system_prompt=system_prompt)
                responses.append(response)
            except Exception as e:
                logger.error(f"Failed to generate for prompt: {e}")
                responses.append("")

            # Rate limit delay
            time.sleep(self.batch_delay)

        return responses
