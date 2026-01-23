"""
LP-TNN: Link Prediction Topological Neural Network for TopoRAG.

This is the core model that:
1. Encodes chunks and cells using topological message passing
2. Scores query-cell pairs for retrieval

From paperidea.tex:
"At inference time, given a new query q_new, we augment the chunk complex
with the query node and apply a link prediction topological neural network
(LP-TNN) to compute scores psi_theta(q_new, sigma) for query-cells links."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from ..liftings.base import Cell, TopologicalComplex
from .cell_encoder import CellEncoder, DeepSetCellEncoder, AttentionCellEncoder
from .link_predictor import QueryCellLinkPredictor, NCNLinkPredictor


@dataclass
class LPTNNOutput:
    """Output from LP-TNN forward pass."""

    query_embedding: torch.Tensor
    cell_embeddings: torch.Tensor
    scores: torch.Tensor
    cells: List[Cell]


class LPTNN(nn.Module):
    """
    Link Prediction Topological Neural Network.

    This model performs query-cell link prediction for retrieval:
    1. Encodes the query using a text encoder
    2. Encodes cells using a cell encoder (DeepSet, Attention, etc.)
    3. Scores query-cell pairs using a link predictor

    Training:
    - Uses speculative queries as supervision
    - BCE loss with negative sampling

    Inference:
    - Given a new query, retrieve top-k cells
    - Return chunks from those cells for LLM generation

    Args:
        embed_dim: Dimension of chunk/query embeddings
        hidden_dim: Hidden dimension for encoders
        cell_encoder_type: Type of cell encoder ('simple', 'deepset', 'attention')
        link_predictor_type: Type of link predictor ('mlp', 'dot', 'ncn')
        num_encoder_layers: Number of layers in cell encoder
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        cell_encoder_type: str = "deepset",
        link_predictor_type: str = "mlp",
        num_encoder_layers: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Query encoder (projects to hidden space)
        self.query_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Cell encoder
        if cell_encoder_type == "simple":
            self.cell_encoder = CellEncoder(
                input_dim=embed_dim,
                output_dim=hidden_dim,
                aggregation="mean",
            )
        elif cell_encoder_type == "deepset":
            self.cell_encoder = DeepSetCellEncoder(
                input_dim=embed_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=num_encoder_layers,
            )
        elif cell_encoder_type == "attention":
            self.cell_encoder = AttentionCellEncoder(
                input_dim=embed_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
            )
        else:
            raise ValueError(f"Unknown cell encoder type: {cell_encoder_type}")

        # Link predictor
        if link_predictor_type == "mlp":
            self.link_predictor = QueryCellLinkPredictor(
                query_dim=hidden_dim,
                cell_dim=hidden_dim,
                hidden_dim=hidden_dim,
                scoring="mlp",
            )
        elif link_predictor_type == "dot":
            self.link_predictor = QueryCellLinkPredictor(
                query_dim=hidden_dim,
                cell_dim=hidden_dim,
                scoring="dot",
            )
        elif link_predictor_type == "ncn":
            self.link_predictor = NCNLinkPredictor(
                embed_dim=hidden_dim,
                hidden_dim=hidden_dim,
            )
        else:
            raise ValueError(f"Unknown link predictor type: {link_predictor_type}")

    def encode_query(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """Encode a query embedding."""
        return self.query_encoder(query_embedding)

    def encode_cells(
        self,
        chunk_features: torch.Tensor,
        cells: List[Cell],
    ) -> torch.Tensor:
        """Encode a list of cells."""
        return self.cell_encoder(chunk_features, cells)

    def forward(
        self,
        query_embedding: torch.Tensor,
        chunk_features: torch.Tensor,
        cells: List[Cell],
    ) -> LPTNNOutput:
        """
        Forward pass: compute query-cell scores.

        Args:
            query_embedding: (embed_dim,) or (num_queries, embed_dim) query embedding
            chunk_features: (num_chunks, embed_dim) chunk embeddings
            cells: List of Cell objects to score

        Returns:
            LPTNNOutput containing embeddings and scores
        """
        # Encode query
        q_emb = self.encode_query(query_embedding)

        # Encode cells
        c_embs = self.encode_cells(chunk_features, cells)

        # Score query-cell pairs
        scores = self.link_predictor(q_emb, c_embs)

        return LPTNNOutput(
            query_embedding=q_emb,
            cell_embeddings=c_embs,
            scores=scores,
            cells=cells,
        )

    def retrieve(
        self,
        query_embedding: torch.Tensor,
        chunk_features: torch.Tensor,
        cells: List[Cell],
        top_k: int = 5,
    ) -> Tuple[List[Cell], torch.Tensor]:
        """
        Retrieve top-k cells for a query.

        Args:
            query_embedding: (embed_dim,) query embedding
            chunk_features: (num_chunks, embed_dim) chunk embeddings
            cells: List of Cell objects
            top_k: Number of cells to retrieve

        Returns:
            Tuple of (top_cells, scores)
        """
        with torch.no_grad():
            output = self.forward(query_embedding, chunk_features, cells)

        # Get top-k
        k = min(top_k, len(cells))
        top_scores, top_indices = torch.topk(output.scores, k=k)

        top_cells = [cells[i] for i in top_indices.tolist()]

        return top_cells, top_scores

    def get_retrieved_chunks(
        self,
        query_embedding: torch.Tensor,
        chunk_features: torch.Tensor,
        cells: List[Cell],
        top_k: int = 5,
        deduplicate: bool = True,
    ) -> List[int]:
        """
        Retrieve chunks by first retrieving cells then extracting chunks.

        Args:
            query_embedding: Query embedding
            chunk_features: Chunk embeddings
            cells: List of cells
            top_k: Number of cells to retrieve
            deduplicate: Whether to remove duplicate chunk indices

        Returns:
            List of chunk indices from top-k cells
        """
        top_cells, _ = self.retrieve(query_embedding, chunk_features, cells, top_k)

        # Extract chunk indices from cells
        chunk_indices = []
        for cell in top_cells:
            chunk_indices.extend(list(cell.chunk_indices))

        if deduplicate:
            # Preserve order while removing duplicates
            seen = set()
            unique_indices = []
            for idx in chunk_indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)
            chunk_indices = unique_indices

        return chunk_indices


class LPTNNTrainer:
    """
    Trainer for LP-TNN.

    Uses speculative queries as supervision signal.
    Training objective: BCE loss with negative sampling.

    From paperidea.tex:
    "The scoring TNN psi_theta(q,sigma) is trained to assign high scores
    to associated query-cell pairs and low scores otherwise, using a
    standard binary cross-entropy objective with negative sampling."
    """

    def __init__(
        self,
        model: LPTNN,
        learning_rate: float = 1e-4,
        num_negative_samples: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.num_negative_samples = num_negative_samples

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

    def sample_negatives(
        self,
        positive_cell_idx: int,
        num_cells: int,
        num_samples: int,
    ) -> List[int]:
        """Sample negative cell indices."""
        negatives = []
        while len(negatives) < num_samples:
            idx = torch.randint(0, num_cells, (1,)).item()
            if idx != positive_cell_idx and idx not in negatives:
                negatives.append(idx)
        return negatives

    def train_step(
        self,
        query_embeddings: torch.Tensor,
        chunk_features: torch.Tensor,
        cells: List[Cell],
        positive_cell_indices: List[int],
    ) -> float:
        """
        Single training step.

        Args:
            query_embeddings: (num_queries, embed_dim) query embeddings
            chunk_features: (num_chunks, embed_dim) chunk embeddings
            cells: List of all Cell objects
            positive_cell_indices: List of positive cell indices for each query

        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()

        query_embeddings = query_embeddings.to(self.device)
        chunk_features = chunk_features.to(self.device)

        # Encode cells
        cell_embeddings = self.model.encode_cells(chunk_features, cells)

        total_loss = 0.0
        num_queries = query_embeddings.shape[0]

        for i in range(num_queries):
            q_emb = self.model.encode_query(query_embeddings[i])

            # Positive sample
            pos_idx = positive_cell_indices[i]
            pos_score = self.model.link_predictor(q_emb, cell_embeddings[pos_idx:pos_idx+1])

            # Negative samples
            neg_indices = self.sample_negatives(pos_idx, len(cells), self.num_negative_samples)
            neg_scores = self.model.link_predictor(
                q_emb,
                cell_embeddings[neg_indices],
            )

            # BCE loss
            pos_labels = torch.ones(1, device=self.device)
            neg_labels = torch.zeros(len(neg_indices), device=self.device)

            scores = torch.cat([pos_score, neg_scores])
            labels = torch.cat([pos_labels, neg_labels])

            loss = self.criterion(scores, labels)
            total_loss += loss

        total_loss /= num_queries
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()

    def train_epoch(
        self,
        train_data: List[Tuple[torch.Tensor, int]],
        chunk_features: torch.Tensor,
        cells: List[Cell],
        batch_size: int = 32,
    ) -> float:
        """
        Train for one epoch.

        Args:
            train_data: List of (query_embedding, positive_cell_idx) tuples
            chunk_features: Chunk embeddings
            cells: List of cells
            batch_size: Batch size

        Returns:
            Average loss for the epoch
        """
        total_loss = 0.0
        num_batches = 0

        # Shuffle data
        indices = torch.randperm(len(train_data))

        for start in range(0, len(train_data), batch_size):
            end = min(start + batch_size, len(train_data))
            batch_indices = indices[start:end]

            batch_queries = torch.stack([train_data[i][0] for i in batch_indices])
            batch_pos_indices = [train_data[i][1] for i in batch_indices]

            loss = self.train_step(
                batch_queries,
                chunk_features,
                cells,
                batch_pos_indices,
            )

            total_loss += loss
            num_batches += 1

        return total_loss / num_batches

    def evaluate(
        self,
        eval_data: List[Tuple[torch.Tensor, int]],
        chunk_features: torch.Tensor,
        cells: List[Cell],
        top_k: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.

        Args:
            eval_data: List of (query_embedding, ground_truth_cell_idx) tuples
            chunk_features: Chunk embeddings
            cells: List of cells
            top_k: Number of cells to retrieve for evaluation

        Returns:
            Dict with metrics (recall@k, mrr, etc.)
        """
        self.model.eval()
        chunk_features = chunk_features.to(self.device)

        hits = 0
        mrr_sum = 0.0

        with torch.no_grad():
            for query_emb, gt_cell_idx in eval_data:
                query_emb = query_emb.to(self.device)

                output = self.model.forward(query_emb, chunk_features, cells)

                # Rank all cells
                ranks = torch.argsort(output.scores, descending=True)

                # Find rank of ground truth
                gt_rank = (ranks == gt_cell_idx).nonzero(as_tuple=True)[0].item()

                if gt_rank < top_k:
                    hits += 1

                mrr_sum += 1.0 / (gt_rank + 1)

        n = len(eval_data)
        return {
            f"recall@{top_k}": hits / n,
            "mrr": mrr_sum / n,
        }
