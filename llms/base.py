import logging
import sqlite3
import hashlib
from typing import Optional, List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseLLM:
    """Base class for LLM interfaces."""

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

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from a prompt."""
        raise NotImplementedError

    def generate_with_messages(self, messages: List[Dict[str, str]]) -> str:
        """Generate text from messages."""
        raise NotImplementedError


class CachedLLM(BaseLLM):
    """
    Wrapper for any LLM that caches responses in SQLite.
    
    This saves money and time by not re-querying the API for known prompts.
    """

    def __init__(self, llm: BaseLLM, cache_dir: str = ".cache"):
        super().__init__(
            llm.model_name, llm.temperature, llm.max_tokens, llm.retry
        )
        self.llm = llm
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Use model-specific DB file for physical separation
        safe_name = "".join(c if c.isalnum() or c in ".-_" else "_" for c in self.model_name)
        self.db_path = self.cache_dir / f"cache_{safe_name}.db"
        
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS cache (
                id TEXT PRIMARY KEY,
                prompt TEXT,
                response TEXT,
                model TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def _get_cache_key(self, prompt: str) -> str:
        """Generate a unique key for the prompt and model config."""
        content = f"{self.model_name}_{self.temperature}_{prompt}"
        return hashlib.md5(content.encode()).hexdigest()

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Check cache before generating."""
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System: {system_prompt}\nUser: {prompt}"

        key = self._get_cache_key(full_prompt)

        # Check cache
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT response FROM cache WHERE id=?", (key,))
        result = c.fetchone()
        conn.close()

        if result:
            return result[0]

        # Generate fresh
        response = self.llm.generate(prompt, system_prompt)

        # Save to cache
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO cache (id, prompt, response, model) VALUES (?, ?, ?, ?)",
            (key, full_prompt, response, self.model_name)
        )
        conn.commit()
        conn.close()

        return response

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
