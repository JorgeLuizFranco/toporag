"""
Local LLM interface for TopoRAG using Hugging Face Transformers.

Recommended models by VRAM:
  GPU 6-8 GB  : Qwen/Qwen2.5-3B-Instruct   (fp16, ~6 GB VRAM)  ← default
  GPU 8+ GB   : Qwen/Qwen2.5-7B-Instruct   (fp16, ~14 GB VRAM, or use CPU below)
  CPU only    : Qwen/Qwen2.5-7B-Instruct   (fp16, ~14 GB RAM) — slow but fine for one-time cache
  CPU light   : Qwen/Qwen2.5-3B-Instruct   (fp16, ~6 GB RAM)

Query generation is ONE-TIME and results are cached to SQLite.
Speed does not matter — use the biggest model that fits.
"""

import logging
import torch
from typing import Optional, List, Dict

from .base import BaseLLM

logger = logging.getLogger(__name__)


class LocalLLM(BaseLLM):
    """
    Local HuggingFace model for offline query generation.

    Args:
        model_name : HuggingFace model id
        device     : "cuda", "cpu", or "auto" (let accelerate decide)
        max_tokens : max new tokens to generate
        temperature: sampling temperature (0 = greedy)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "auto",
        max_tokens: int = 256,
        temperature: float = 0.3,
        **kwargs,
    ):
        super().__init__(model_name, temperature, max_tokens, retry=1)
        self.device_arg = device

        print(f"Loading local model: {model_name} on device={device} ...")
        print("(This is a one-time load — queries will be cached afterwards)")

        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Determine dtype: fp16 for GPU, fp32 for CPU
        if device == "cpu":
            dtype = torch.float32
            device_map = "cpu"
        elif device == "cuda" or device == "auto":
            dtype = torch.float16
            device_map = "auto"
        else:
            dtype = torch.float16
            device_map = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
        )
        print(f"  Model loaded. Memory: {self._mem_str()}")

    def _mem_str(self) -> str:
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"{used:.1f}/{total:.1f} GB VRAM"
        return "CPU"

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.generate_with_messages(messages)

    def generate_with_messages(self, messages: List[Dict[str, str]]) -> str:
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback for models without a chat template
            formatted = "\n".join(
                f"{'System' if m['role'] == 'system' else 'User'}: {m['content']}"
                for m in messages
            ) + "\nAssistant:"

        outputs = self.pipe(formatted)
        full_text = outputs[0]["generated_text"]

        # Strip the prompt prefix, return only new tokens
        if full_text.startswith(formatted):
            return full_text[len(formatted):].strip()
        return full_text.strip()

    def free(self):
        """Release GPU memory after query generation is done."""
        import gc
        del self.pipe
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  Local model freed from memory.")
