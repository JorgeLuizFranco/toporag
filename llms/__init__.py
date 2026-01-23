"""
LLM interfaces for TopoRAG.

Supports multiple backends:
- OpenAI (GPT-4o-mini, GPT-4o) - used by GFM-RAG
- Groq (free tier available) - llama-3.1-70b, mixtral
- Local models via Ollama
"""

from .base import BaseLLM
from .openai_llm import OpenAILLM
from .groq_llm import GroqLLM

__all__ = [
    "BaseLLM",
    "OpenAILLM",
    "GroqLLM",
    "get_llm",
]


def get_llm(
    provider: str = "groq",
    model_name: str = None,
    **kwargs,
) -> BaseLLM:
    """
    Factory function to get an LLM instance.

    Args:
        provider: LLM provider ("openai", "groq", "ollama")
        model_name: Model name (provider-specific)
        **kwargs: Additional arguments for the LLM

    Returns:
        LLM instance

    Examples:
        # Use Groq (free)
        llm = get_llm("groq", model_name="llama-3.3-70b-versatile")

        # Use OpenAI (GFM-RAG default)
        llm = get_llm("openai", model_name="gpt-4o-mini")
    """
    if provider == "openai":
        model_name = model_name or "gpt-4o-mini"
        return OpenAILLM(model_name=model_name, **kwargs)
    elif provider == "groq":
        model_name = model_name or "llama-3.3-70b-versatile"
        return GroqLLM(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")
