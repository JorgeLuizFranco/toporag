"""
Base LLM interface for TopoRAG.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Union


@dataclass
class LLMResponse:
    """Response from an LLM."""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None  # tokens used


class BaseLLM(ABC):
    """
    Abstract base class for LLM interfaces.

    All LLM implementations should inherit from this class.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        retry: int = 3,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry = retry

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        """
        Generate text from a list of messages.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts

        Returns:
            Generated text
        """
        pass

    def generate_answer(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate an answer given a question and context.

        This is the main method for RAG answer generation.

        Args:
            question: The question to answer
            context: Retrieved context/documents
            system_prompt: Optional system instruction

        Returns:
            Generated answer
        """
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant that answers questions based on the provided context. "
                "Answer the question concisely based only on the given information. "
                "If the answer cannot be found in the context, say so."
            )

        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

        return self.generate(prompt, system_prompt=system_prompt)
