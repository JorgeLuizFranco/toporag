"""
LLM interfaces for TopoRAG.

Backends:
- groq   : free API, llama-3.1-8b-instant — recommended for query generation
- openai : gpt-4o-mini — highest quality, paid
- local  : HuggingFace model on GPU — slow, low quality for multi-hop prompts
"""

from .base import BaseLLM, CachedLLM
from .openai_llm import OpenAILLM
from .local_llm import LocalLLM
from .groq_llm import GroqLLM

__all__ = [
    "BaseLLM",
    "CachedLLM",
    "OpenAILLM",
    "LocalLLM",
    "GroqLLM",
    "get_llm",
]


def get_llm(
    provider: str = "local",
    model_name: str = None,
    use_cache: bool = True,
    cache_dir: str = ".cache",
    **kwargs,
) -> BaseLLM:
    """
    Factory for LLM instances. Wraps with CachedLLM by default so
    every generation is cached to SQLite — generate once, reuse forever.

    Args:
        provider   : "local" (default, offline), "groq" (free API), "openai" (paid)
        model_name : override the default model for the provider
        use_cache  : wrap with CachedLLM (default True)
        cache_dir  : directory for SQLite cache files

    Model defaults:
        local  -> Qwen/Qwen2.5-3B-Instruct  (fp16 GPU, ~6 GB VRAM)
                  pass device="cpu" + model_name="Qwen/Qwen2.5-7B-Instruct" for higher quality
        groq   -> llama-3.1-8b-instant       (free API)
        openai -> gpt-4o-mini                (paid, highest quality)

    Examples:
        # Best offline option for 6-8 GB GPU:
        llm = get_llm("local")

        # Higher quality, uses CPU RAM (needs ~14 GB):
        llm = get_llm("local", model_name="Qwen/Qwen2.5-7B-Instruct", device="cpu")

        # Free API fallback:
        llm = get_llm("groq")
    """
    if provider == "local":
        model_name = model_name or "Qwen/Qwen2.5-3B-Instruct"
        llm = LocalLLM(model_name=model_name, **kwargs)
    elif provider == "groq":
        model_name = model_name or "llama-3.1-8b-instant"
        llm = GroqLLM(model_name=model_name, **kwargs)
    elif provider == "openai":
        model_name = model_name or "gpt-4o-mini"
        llm = OpenAILLM(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown provider: '{provider}'. Choose: local, groq, openai")

    if use_cache:
        return CachedLLM(llm, cache_dir=cache_dir)
    return llm
