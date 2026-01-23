"""
Text Embedding utilities for TopoRAG.

Following GFM-RAG's approach using sentence-transformers for clean,
efficient embeddings with proper batch processing and GPU support.

GFM-RAG uses:
- all-mpnet-v2 (default, 768-dim, fast)
- NV-Embed-v2 (32k context, for long documents)

For TopoRAG, we also support:
- facebook/contriever (used by LP-RAG, HippoRAG)
- BAAI/bge-large-en-v1.5 (strong retrieval model)
"""

import torch
import numpy as np
from typing import List, Optional, Union
from dataclasses import dataclass


@dataclass
class EmbeddingModelConfig:
    """Configuration for embedding models."""
    name: str
    dimension: int
    max_seq_length: int
    normalize: bool = True
    query_instruction: Optional[str] = None
    passage_instruction: Optional[str] = None


# Pre-configured models
EMBEDDING_MODELS = {
    "all-mpnet-v2": EmbeddingModelConfig(
        name="sentence-transformers/all-mpnet-base-v2",
        dimension=768,
        max_seq_length=512,
        normalize=True,
    ),
    "contriever": EmbeddingModelConfig(
        name="facebook/contriever",
        dimension=768,
        max_seq_length=512,
        normalize=False,
    ),
    "bge-large": EmbeddingModelConfig(
        name="BAAI/bge-large-en-v1.5",
        dimension=1024,
        max_seq_length=512,
        normalize=True,
        query_instruction="Represent this sentence for searching relevant passages: ",
    ),
    "e5-large": EmbeddingModelConfig(
        name="intfloat/e5-large-v2",
        dimension=1024,
        max_seq_length=512,
        normalize=True,
        query_instruction="query: ",
        passage_instruction="passage: ",
    ),
}


class TextEmbeddingModel:
    """
    Text embedding model following GFM-RAG's clean interface.

    Uses sentence-transformers for efficient batch processing with:
    - Automatic GPU usage
    - Proper batching
    - Progress bars
    - L2 normalization option

    Args:
        model_name: Model name or key from EMBEDDING_MODELS
        normalize: Whether to L2-normalize embeddings
        batch_size: Batch size for encoding
        device: Device to use (auto-detected if None)

    Example:
        >>> embedder = TextEmbeddingModel("all-mpnet-v2")
        >>> embeddings = embedder.encode(["Hello world", "Another text"])
        >>> embeddings.shape
        torch.Size([2, 768])
    """

    def __init__(
        self,
        model_name: str = "all-mpnet-v2",
        normalize: bool = True,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        # Get config if using preset
        if model_name in EMBEDDING_MODELS:
            config = EMBEDDING_MODELS[model_name]
            self.model_name = config.name
            self.dimension = config.dimension
            self.max_seq_length = config.max_seq_length
            self.query_instruction = config.query_instruction
            self.passage_instruction = config.passage_instruction
            normalize = config.normalize if normalize else False
        else:
            self.model_name = model_name
            self.dimension = None  # Will be set after loading
            self.max_seq_length = 512
            self.query_instruction = None
            self.passage_instruction = None

        self.normalize = normalize
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self._load_model()

    def _load_model(self):
        """Load the sentence-transformer model."""
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            self.model_name,
            trust_remote_code=True,
        )
        self.model.to(self.device)

        # Get dimension from model if not set
        if self.dimension is None:
            self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: Union[str, List[str]],
        is_query: bool = False,
        show_progress: bool = True,
        convert_to_tensor: bool = True,
    ) -> torch.Tensor:
        """
        Encode texts into embeddings.

        Args:
            texts: Single text or list of texts
            is_query: If True, use query instruction (for asymmetric models)
            show_progress: Whether to show progress bar
            convert_to_tensor: Return torch.Tensor (else numpy)

        Returns:
            Embeddings tensor of shape (num_texts, dimension)
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        # Apply instruction prefix if configured
        if is_query and self.query_instruction:
            texts = [self.query_instruction + t for t in texts]
        elif not is_query and self.passage_instruction:
            texts = [self.passage_instruction + t for t in texts]

        # Encode using sentence-transformers
        embeddings = self.model.encode(
            texts,
            device=self.device,
            normalize_embeddings=self.normalize,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=convert_to_tensor,
        )

        return embeddings

    def encode_queries(
        self,
        queries: List[str],
        **kwargs,
    ) -> torch.Tensor:
        """Encode query texts (with query instruction if applicable)."""
        return self.encode(queries, is_query=True, **kwargs)

    def encode_passages(
        self,
        passages: List[str],
        **kwargs,
    ) -> torch.Tensor:
        """Encode passage/document texts."""
        return self.encode(passages, is_query=False, **kwargs)


def cosine_similarity(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine similarity between two sets of embeddings.

    Args:
        embeddings1: (n, d) tensor
        embeddings2: (m, d) tensor

    Returns:
        (n, m) similarity matrix
    """
    # Normalize if not already normalized
    embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=-1)
    embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=-1)

    return torch.matmul(embeddings1, embeddings2.T)


def find_top_k(
    query_embeddings: torch.Tensor,
    corpus_embeddings: torch.Tensor,
    k: int = 10,
) -> tuple:
    """
    Find top-k most similar items from corpus for each query.

    Args:
        query_embeddings: (num_queries, dim) tensor
        corpus_embeddings: (num_corpus, dim) tensor
        k: Number of top items to return

    Returns:
        Tuple of (indices, scores) with shapes (num_queries, k)
    """
    similarities = cosine_similarity(query_embeddings, corpus_embeddings)
    top_scores, top_indices = torch.topk(similarities, k=min(k, similarities.shape[1]), dim=-1)

    return top_indices, top_scores


# Convenience function for quick embedding
def embed_texts(
    texts: List[str],
    model_name: str = "all-mpnet-v2",
    **kwargs,
) -> torch.Tensor:
    """
    Quick function to embed texts.

    Args:
        texts: List of texts to embed
        model_name: Embedding model to use
        **kwargs: Additional arguments for TextEmbeddingModel

    Returns:
        Embeddings tensor
    """
    embedder = TextEmbeddingModel(model_name, **kwargs)
    return embedder.encode(texts)
