"""
Link Predictors for TopoRAG.

These modules score query-cell pairs for retrieval.
The scoring function psi_theta(q, sigma) predicts the relevance
of cell sigma to query q.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class QueryCellLinkPredictor(nn.Module):
    """
    Simple link predictor using dot product or bilinear scoring.

    Computes scores as: psi(q, c) = q^T W c  or  psi(q, c) = q^T c

    Args:
        query_dim: Dimension of query embeddings
        cell_dim: Dimension of cell embeddings
        hidden_dim: Hidden dimension for MLP variant
        scoring: Scoring function type ('dot', 'bilinear', 'mlp')
    """

    def __init__(
        self,
        query_dim: int,
        cell_dim: int,
        hidden_dim: int = 128,
        scoring: str = "mlp",
    ):
        super().__init__()
        self.query_dim = query_dim
        self.cell_dim = cell_dim
        self.hidden_dim = hidden_dim
        self.scoring = scoring

        if scoring == "bilinear":
            self.weight = nn.Parameter(torch.randn(query_dim, cell_dim))
            nn.init.xavier_uniform_(self.weight)
        elif scoring == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(query_dim + cell_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1),
            )

    def forward(
        self,
        query_embeddings: torch.Tensor,
        cell_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute link prediction scores.

        Args:
            query_embeddings: (num_queries, query_dim) or (query_dim,)
            cell_embeddings: (num_cells, cell_dim)

        Returns:
            Scores tensor:
            - If query_embeddings is 1D: (num_cells,) scores
            - If query_embeddings is 2D: (num_queries, num_cells) scores
        """
        # Handle single query case
        if query_embeddings.dim() == 1:
            query_embeddings = query_embeddings.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        num_queries = query_embeddings.shape[0]
        num_cells = cell_embeddings.shape[0]

        if self.scoring == "dot":
            # Simple dot product: (num_queries, num_cells)
            scores = torch.matmul(query_embeddings, cell_embeddings.T)

        elif self.scoring == "bilinear":
            # Bilinear: q^T W c
            # (num_queries, query_dim) @ (query_dim, cell_dim) @ (cell_dim, num_cells)
            transformed = torch.matmul(query_embeddings, self.weight)  # (num_queries, cell_dim)
            scores = torch.matmul(transformed, cell_embeddings.T)  # (num_queries, num_cells)

        elif self.scoring == "mlp":
            # MLP on concatenated embeddings
            # Expand for broadcasting
            q_expanded = query_embeddings.unsqueeze(1).expand(-1, num_cells, -1)  # (nq, nc, qd)
            c_expanded = cell_embeddings.unsqueeze(0).expand(num_queries, -1, -1)  # (nq, nc, cd)

            # Concatenate
            combined = torch.cat([q_expanded, c_expanded], dim=-1)  # (nq, nc, qd+cd)

            # Apply MLP
            scores = self.mlp(combined).squeeze(-1)  # (nq, nc)

        if squeeze_output:
            scores = scores.squeeze(0)

        return scores

    def predict_links(
        self,
        query_embedding: torch.Tensor,
        cell_embeddings: torch.Tensor,
        top_k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict top-k cell links for a query.

        Args:
            query_embedding: (query_dim,) tensor
            cell_embeddings: (num_cells, cell_dim) tensor
            top_k: Number of top cells to return

        Returns:
            Tuple of (cell_indices, scores) for top-k cells
        """
        with torch.no_grad():
            scores = self.forward(query_embedding, cell_embeddings)
            top_scores, top_indices = torch.topk(scores, k=min(top_k, len(scores)))

        return top_indices, top_scores


class NCNLinkPredictor(nn.Module):
    """
    Neural Common Neighbor-based Link Predictor.

    Inspired by NCN/NCNC methods that use common neighbor information.
    For TopoRAG, we adapt this to use common chunks between query and cell.

    The idea: cells that share more semantic similarity with the query
    (measured via intermediate chunks) should score higher.

    Args:
        embed_dim: Embedding dimension
        hidden_dim: Hidden dimension
        num_layers: Number of MLP layers
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Feature transformation for query
        self.query_transform = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Feature transformation for cell
        self.cell_transform = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Scoring MLP
        layers = []
        in_dim = hidden_dim * 3  # query + cell + element-wise product
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU() if i < num_layers - 1 else nn.Identity(),
            ])
            in_dim = out_dim
        self.score_mlp = nn.Sequential(*layers)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        cell_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute NCN-style link prediction scores.

        Args:
            query_embeddings: (num_queries, embed_dim) or (embed_dim,)
            cell_embeddings: (num_cells, embed_dim)

        Returns:
            (num_queries, num_cells) or (num_cells,) scores
        """
        if query_embeddings.dim() == 1:
            query_embeddings = query_embeddings.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        num_queries = query_embeddings.shape[0]
        num_cells = cell_embeddings.shape[0]

        # Transform features
        q_feat = self.query_transform(query_embeddings)  # (nq, hidden)
        c_feat = self.cell_transform(cell_embeddings)  # (nc, hidden)

        # Expand for pairwise combination
        q_expanded = q_feat.unsqueeze(1).expand(-1, num_cells, -1)
        c_expanded = c_feat.unsqueeze(0).expand(num_queries, -1, -1)

        # Element-wise product captures interaction
        interaction = q_expanded * c_expanded

        # Concatenate all features
        combined = torch.cat([q_expanded, c_expanded, interaction], dim=-1)

        # Score
        scores = self.score_mlp(combined).squeeze(-1)

        if squeeze_output:
            scores = scores.squeeze(0)

        return scores


class TransformerLinkPredictor(nn.Module):
    """
    Transformer-based link predictor.

    Uses cross-attention between query and cell embeddings.
    More expressive but computationally heavier.

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Cross-attention layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Scoring head
        self.score_head = nn.Linear(embed_dim, 1)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        cell_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute scores using cross-attention.

        Args:
            query_embeddings: (num_queries, embed_dim) or (embed_dim,)
            cell_embeddings: (num_cells, embed_dim)

        Returns:
            (num_queries, num_cells) or (num_cells,) scores
        """
        if query_embeddings.dim() == 1:
            query_embeddings = query_embeddings.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Treat cells as memory, query as target
        # Shape: (num_queries, 1, embed_dim) for query
        # Shape: (num_queries, num_cells, embed_dim) for cells (repeated)
        num_queries = query_embeddings.shape[0]
        num_cells = cell_embeddings.shape[0]

        q = query_embeddings.unsqueeze(1)  # (nq, 1, d)
        c = cell_embeddings.unsqueeze(0).expand(num_queries, -1, -1)  # (nq, nc, d)

        # Cross-attention: query attends to cells
        attended = self.transformer(q, c)  # (nq, 1, d)

        # Score each cell based on attended query representation
        # Compute similarity between attended query and each cell
        attended = attended.expand(-1, num_cells, -1)  # (nq, nc, d)
        scores = (attended * c).sum(dim=-1)  # (nq, nc)

        if squeeze_output:
            scores = scores.squeeze(0)

        return scores
