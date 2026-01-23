"""
TopoRAG Pipeline: End-to-end retrieval workflow.

This module implements the complete TopoRAG pipeline:

1. Document Processing:
   - Chunk extraction (from LP-RAG)
   - Chunk embedding (Contriever)

2. Topological Lifting:
   - Build chunk graph (k-NN similarity)
   - Apply static lifting (k-NN, cycle, clique)
   - Produce topological complex with cells

3. Query Generation:
   - Generate speculative queries for each cell
   - Create query-cell association for supervision

4. Training:
   - Train LP-TNN with BCE loss
   - Negative sampling over cells

5. Inference:
   - Encode user query
   - Retrieve top-k cells via link prediction
   - Extract chunks from cells
   - Generate answer with LLM
"""

import json
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

from .liftings.base import TopologicalComplex, Cell
from .liftings import KNNLifting, CycleLifting, CliqueLifting
from .speculative_queries import SpeculativeQueryGenerator, SpeculativeQuery
from .models.lp_tnn import LPTNN, LPTNNTrainer
from .utils.embedding import TextEmbeddingModel


@dataclass
class TopoRAGConfig:
    """Configuration for TopoRAG pipeline."""

    # Embedding (supports: "contriever", "all-mpnet-v2", "bge-large", "e5-large")
    embed_model: str = "contriever"
    embed_dim: int = 768

    # Lifting
    lifting_type: str = "knn"  # "knn", "cycle", "clique"
    knn_k: int = 3
    max_cycle_length: int = 6
    max_clique_size: int = 5

    # Model
    hidden_dim: int = 256
    cell_encoder_type: str = "deepset"
    link_predictor_type: str = "mlp"

    # Training
    learning_rate: float = 1e-4
    num_negative_samples: int = 5
    batch_size: int = 32
    num_epochs: int = 10

    # Inference
    top_k_cells: int = 5

    # Paths
    output_dir: str = "./toporag_output"


class TopoRAGPipeline:
    """
    Main TopoRAG pipeline.

    Implements the full workflow from paperidea.tex:
    I1: Lift chunks to higher-order structures
    I2: Generate speculative queries for cells
    I3: Query-cell link prediction for retrieval
    """

    def __init__(self, config: TopoRAGConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self._init_embedding_model()
        self._init_lifting()
        self._init_model()

        # State
        self.chunk_texts: List[str] = []
        self.chunk_features: Optional[torch.Tensor] = None
        self.complex: Optional[TopologicalComplex] = None
        self.queries_by_cell: Dict[int, List[SpeculativeQuery]] = {}

    def _init_embedding_model(self):
        """Initialize text embedding model."""
        self.embedder = TextEmbeddingModel(
            model_name=self.config.embed_model,
            device=str(self.device),
        )
        # Update embed_dim from model if using preset
        if self.embedder.dimension:
            self.config.embed_dim = self.embedder.dimension

    def _init_lifting(self):
        """Initialize lifting transform."""
        if self.config.lifting_type == "knn":
            self.lifting = KNNLifting(k=self.config.knn_k)
        elif self.config.lifting_type == "cycle":
            self.lifting = CycleLifting(max_cycle_length=self.config.max_cycle_length)
        elif self.config.lifting_type == "clique":
            self.lifting = CliqueLifting(max_clique_size=self.config.max_clique_size)
        else:
            raise ValueError(f"Unknown lifting type: {self.config.lifting_type}")

    def _init_model(self):
        """Initialize LP-TNN model."""
        self.model = LPTNN(
            embed_dim=self.config.embed_dim,
            hidden_dim=self.config.hidden_dim,
            cell_encoder_type=self.config.cell_encoder_type,
            link_predictor_type=self.config.link_predictor_type,
        )
        self.model.to(self.device)

    def embed_texts(self, texts: List[str], is_query: bool = False) -> torch.Tensor:
        """
        Embed texts using the configured embedding model.

        Args:
            texts: List of text strings
            is_query: Whether these are query texts (uses query instruction if applicable)

        Returns:
            (num_texts, embed_dim) tensor
        """
        embeddings = self.embedder.encode(
            texts,
            is_query=is_query,
            show_progress=len(texts) > 10,
        )
        # Ensure CPU tensor for downstream processing
        if isinstance(embeddings, torch.Tensor):
            return embeddings.cpu()
        return torch.tensor(embeddings)

    def build_chunk_graph(
        self,
        chunk_features: torch.Tensor,
        k: int = 10,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Build chunk graph using k-NN similarity.

        Args:
            chunk_features: (num_chunks, embed_dim) tensor
            k: Number of neighbors
            threshold: Similarity threshold for edges

        Returns:
            (2, num_edges) edge_index tensor
        """
        from sklearn.neighbors import NearestNeighbors

        features_np = chunk_features.numpy()

        # Fit k-NN
        knn = NearestNeighbors(n_neighbors=min(k + 1, len(features_np)), metric="cosine")
        knn.fit(features_np)

        distances, indices = knn.kneighbors(features_np)

        # Convert to edge list
        edges = []
        for i in range(len(features_np)):
            for j, dist in zip(indices[i], distances[i]):
                if i != j and (1 - dist) > threshold:  # cosine similarity
                    edges.append([i, j])

        if edges:
            edge_index = torch.tensor(edges).T
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return edge_index

    def process_chunks(self, chunk_texts: List[str]) -> TopologicalComplex:
        """
        Process chunks: embed, build graph, lift to complex.

        Args:
            chunk_texts: List of chunk text strings

        Returns:
            TopologicalComplex with cells
        """
        self.chunk_texts = chunk_texts

        # Embed chunks
        print(f"Embedding {len(chunk_texts)} chunks...")
        self.chunk_features = self.embed_texts(chunk_texts)

        # Build chunk graph
        print("Building chunk graph...")
        edge_index = self.build_chunk_graph(self.chunk_features)
        print(f"  Graph has {edge_index.shape[1]} edges")

        # Apply lifting
        print(f"Applying {self.config.lifting_type} lifting...")
        self.complex = self.lifting(self.chunk_features, edge_index)

        # Summary
        for dim in sorted(self.complex.cells_by_dim.keys()):
            cells = self.complex.get_cells(dim)
            print(f"  Dimension {dim}: {len(cells)} cells")

        return self.complex

    def generate_speculative_queries(
        self,
        llm,
        num_queries_per_cell: int = 2,
    ) -> Dict[int, List[SpeculativeQuery]]:
        """
        Generate speculative queries for cells.

        Args:
            llm: Language model for generation
            num_queries_per_cell: Queries to generate per cell

        Returns:
            Dict mapping cell_id -> list of queries
        """
        if self.complex is None:
            raise ValueError("Must call process_chunks first")

        generator = SpeculativeQueryGenerator(
            llm=llm,
            num_queries_per_cell=num_queries_per_cell,
        )

        print("Generating speculative queries for cells...")
        self.queries_by_cell = generator.generate_for_complex(
            self.complex,
            self.chunk_texts,
        )

        total_queries = sum(len(qs) for qs in self.queries_by_cell.values())
        print(f"  Generated {total_queries} queries for {len(self.queries_by_cell)} cells")

        return self.queries_by_cell

    def prepare_training_data(
        self,
    ) -> List[Tuple[torch.Tensor, int]]:
        """
        Prepare training data from speculative queries.

        Returns:
            List of (query_embedding, cell_idx) tuples
        """
        if not self.queries_by_cell:
            raise ValueError("Must call generate_speculative_queries first")

        # Get all higher-order cells
        cells = self.complex.get_all_higher_order_cells()
        cell_id_to_idx = {cell.cell_id: i for i, cell in enumerate(cells)}

        # Embed query texts
        all_query_texts = []
        all_cell_indices = []

        for cell_id, queries in self.queries_by_cell.items():
            if cell_id in cell_id_to_idx:
                for query in queries:
                    all_query_texts.append(query.query_text)
                    all_cell_indices.append(cell_id_to_idx[cell_id])

        print(f"Embedding {len(all_query_texts)} query texts...")
        query_embeddings = self.embed_texts(all_query_texts, is_query=True)

        # Create training pairs
        train_data = [
            (query_embeddings[i], all_cell_indices[i])
            for i in range(len(all_query_texts))
        ]

        return train_data

    def train(
        self,
        train_data: List[Tuple[torch.Tensor, int]],
        num_epochs: Optional[int] = None,
    ) -> List[float]:
        """
        Train the LP-TNN model.

        Args:
            train_data: List of (query_embedding, cell_idx) tuples
            num_epochs: Number of epochs (uses config if not specified)

        Returns:
            List of loss values per epoch
        """
        if self.complex is None:
            raise ValueError("Must call process_chunks first")

        num_epochs = num_epochs or self.config.num_epochs
        cells = self.complex.get_all_higher_order_cells()

        trainer = LPTNNTrainer(
            model=self.model,
            learning_rate=self.config.learning_rate,
            num_negative_samples=self.config.num_negative_samples,
            device=self.device,
        )

        losses = []
        for epoch in range(num_epochs):
            loss = trainer.train_epoch(
                train_data,
                self.chunk_features.to(self.device),
                cells,
                batch_size=self.config.batch_size,
            )
            losses.append(loss)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")

        return losses

    def retrieve(
        self,
        query_text: str,
        top_k: Optional[int] = None,
    ) -> Tuple[List[Cell], List[int], torch.Tensor]:
        """
        Retrieve chunks for a query.

        Args:
            query_text: User query text
            top_k: Number of cells to retrieve

        Returns:
            Tuple of (top_cells, chunk_indices, scores)
        """
        if self.complex is None:
            raise ValueError("Must call process_chunks first")

        top_k = top_k or self.config.top_k_cells
        cells = self.complex.get_all_higher_order_cells()

        # Embed query (use is_query=True for asymmetric models like BGE)
        query_emb = self.embed_texts([query_text], is_query=True)[0].to(self.device)

        # Retrieve cells
        self.model.eval()
        top_cells, scores = self.model.retrieve(
            query_emb,
            self.chunk_features.to(self.device),
            cells,
            top_k=top_k,
        )

        # Extract chunk indices
        chunk_indices = self.model.get_retrieved_chunks(
            query_emb,
            self.chunk_features.to(self.device),
            cells,
            top_k=top_k,
        )

        return top_cells, chunk_indices, scores

    def get_context(
        self,
        query_text: str,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Get retrieval context as a string for LLM generation.

        Args:
            query_text: User query
            top_k: Number of cells to retrieve

        Returns:
            Context string with retrieved chunks
        """
        _, chunk_indices, _ = self.retrieve(query_text, top_k)

        context_parts = []
        for i, idx in enumerate(chunk_indices):
            context_parts.append(f"[{i + 1}] {self.chunk_texts[idx]}")

        return "\n\n".join(context_parts)

    def save(self, output_dir: Optional[str] = None):
        """Save pipeline state."""
        output_dir = Path(output_dir or self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(self.model.state_dict(), output_dir / "model.pt")

        # Save config
        with open(output_dir / "config.json", "w") as f:
            json.dump(vars(self.config), f, indent=2)

        # Save chunks and embeddings
        if self.chunk_texts:
            with open(output_dir / "chunks.json", "w") as f:
                json.dump(self.chunk_texts, f)

        if self.chunk_features is not None:
            torch.save(self.chunk_features, output_dir / "chunk_features.pt")

        print(f"Saved pipeline to {output_dir}")

    def load(self, output_dir: str):
        """Load pipeline state."""
        output_dir = Path(output_dir)

        # Load model
        self.model.load_state_dict(torch.load(output_dir / "model.pt"))
        self.model.to(self.device)

        # Load chunks
        with open(output_dir / "chunks.json") as f:
            self.chunk_texts = json.load(f)

        # Load embeddings
        self.chunk_features = torch.load(output_dir / "chunk_features.pt")

        # Rebuild complex
        edge_index = self.build_chunk_graph(self.chunk_features)
        self.complex = self.lifting(self.chunk_features, edge_index)

        print(f"Loaded pipeline from {output_dir}")


def create_pipeline(
    lifting_type: str = "knn",
    **kwargs,
) -> TopoRAGPipeline:
    """
    Factory function to create a TopoRAG pipeline.

    Args:
        lifting_type: Type of lifting ("knn", "cycle", "clique")
        **kwargs: Additional config options

    Returns:
        Configured TopoRAGPipeline
    """
    config = TopoRAGConfig(lifting_type=lifting_type, **kwargs)
    return TopoRAGPipeline(config)
