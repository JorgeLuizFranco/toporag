"""
Cell Encoders for TopoRAG.

Cells are groups of chunks, and we need to compute embeddings for them.
These encoders take chunk embeddings and aggregate them to produce cell embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from ..liftings.base import Cell, TopologicalComplex


class CellEncoder(nn.Module):
    """
    Base cell encoder that aggregates chunk embeddings.

    Simple aggregation methods: mean, sum, max.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        aggregation: str = "mean",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregation = aggregation

        # Optional projection layer
        self.projection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(
        self,
        chunk_features: torch.Tensor,
        cells: List[Cell],
    ) -> torch.Tensor:
        """
        Compute cell embeddings by aggregating chunk embeddings.

        Args:
            chunk_features: (num_chunks, input_dim) tensor
            cells: List of Cell objects to encode

        Returns:
            (num_cells, output_dim) tensor of cell embeddings
        """
        cell_embeddings = []

        for cell in cells:
            indices = list(cell.chunk_indices)
            chunk_embs = chunk_features[indices]  # (k, input_dim)

            if self.aggregation == "mean":
                cell_emb = chunk_embs.mean(dim=0)
            elif self.aggregation == "sum":
                cell_emb = chunk_embs.sum(dim=0)
            elif self.aggregation == "max":
                cell_emb = chunk_embs.max(dim=0).values
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")

            cell_embeddings.append(cell_emb)

        cell_embeddings = torch.stack(cell_embeddings)  # (num_cells, input_dim)
        return self.projection(cell_embeddings)


class DeepSetCellEncoder(nn.Module):
    """
    DeepSet-based cell encoder.

    Uses a permutation-invariant DeepSet architecture:
    f(X) = rho(sum(phi(x_i)))

    This is more expressive than simple aggregation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # phi network: transforms individual elements
        phi_layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            phi_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.phi = nn.Sequential(*phi_layers)

        # rho network: transforms aggregated representation
        rho_layers = [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            rho_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        rho_layers.append(nn.Linear(hidden_dim, output_dim))
        self.rho = nn.Sequential(*rho_layers)

    def forward(
        self,
        chunk_features: torch.Tensor,
        cells: List[Cell],
    ) -> torch.Tensor:
        """
        Compute cell embeddings using DeepSet.

        Args:
            chunk_features: (num_chunks, input_dim) tensor
            cells: List of Cell objects

        Returns:
            (num_cells, output_dim) tensor
        """
        cell_embeddings = []

        for cell in cells:
            indices = list(cell.chunk_indices)
            chunk_embs = chunk_features[indices]  # (k, input_dim)

            # Apply phi to each element
            transformed = self.phi(chunk_embs)  # (k, hidden_dim)

            # Sum aggregation (permutation-invariant)
            aggregated = transformed.sum(dim=0)  # (hidden_dim,)

            # Apply rho
            cell_emb = self.rho(aggregated)  # (output_dim,)
            cell_embeddings.append(cell_emb)

        return torch.stack(cell_embeddings)


class AttentionCellEncoder(nn.Module):
    """
    Attention-based cell encoder.

    Uses self-attention to weight chunks within a cell before aggregation.
    This allows the model to learn which chunks are most important.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        # Projections for attention
        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Linear(input_dim, hidden_dim)
        self.v_proj = nn.Linear(input_dim, hidden_dim)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        chunk_features: torch.Tensor,
        cells: List[Cell],
    ) -> torch.Tensor:
        """
        Compute cell embeddings using attention.

        Args:
            chunk_features: (num_chunks, input_dim) tensor
            cells: List of Cell objects

        Returns:
            (num_cells, output_dim) tensor
        """
        cell_embeddings = []

        for cell in cells:
            indices = list(cell.chunk_indices)
            chunk_embs = chunk_features[indices]  # (k, input_dim)

            # Project to Q, K, V
            q = self.q_proj(chunk_embs).unsqueeze(0)  # (1, k, hidden_dim)
            k = self.k_proj(chunk_embs).unsqueeze(0)
            v = self.v_proj(chunk_embs).unsqueeze(0)

            # Self-attention
            attended, _ = self.attention(q, k, v)  # (1, k, hidden_dim)

            # Mean pooling over attended representations
            cell_emb = attended.squeeze(0).mean(dim=0)  # (hidden_dim,)

            # Project to output
            cell_emb = self.output_proj(cell_emb)  # (output_dim,)
            cell_embeddings.append(cell_emb)

        return torch.stack(cell_embeddings)


class HierarchicalCellEncoder(nn.Module):
    """
    Hierarchical cell encoder that respects boundary relations.

    For cell complexes, this encodes cells by combining:
    1. Aggregated chunk embeddings (boundary)
    2. Embeddings from lower-dimensional cells on the boundary

    This captures the hierarchical structure of the topological domain.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        max_dimension: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_dimension = max_dimension

        # Encoders for each dimension
        self.encoders = nn.ModuleDict()
        for dim in range(max_dimension + 1):
            if dim == 0:
                # 0-cells are just projected chunk features
                self.encoders[str(dim)] = nn.Linear(input_dim, hidden_dim)
            else:
                # Higher-order cells aggregate from lower-dimensional boundary cells
                self.encoders[str(dim)] = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),  # Concat boundary + direct
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )

        # Final projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        complex_: TopologicalComplex,
    ) -> dict:
        """
        Encode all cells in the complex hierarchically.

        Args:
            complex_: TopologicalComplex containing cells at multiple dimensions

        Returns:
            Dict mapping dimension -> (num_cells_at_dim, output_dim) tensor
        """
        cell_embeddings = {}

        for dim in range(self.max_dimension + 1):
            cells = complex_.get_cells(dim)
            if not cells:
                continue

            if dim == 0:
                # 0-cells are chunk embeddings
                embs = self.encoders[str(dim)](complex_.chunk_features)
            else:
                # Higher-order cells
                # Get direct chunk aggregation
                direct_embs = []
                boundary_embs = []

                for cell in cells:
                    # Direct: aggregate chunk embeddings
                    indices = list(cell.chunk_indices)
                    direct = complex_.chunk_features[indices].mean(dim=0)
                    direct_embs.append(direct)

                    # Boundary: aggregate lower-dim cell embeddings
                    if (dim - 1) in cell_embeddings:
                        lower_cells = complex_.get_cells(dim - 1)
                        boundary_indices = [
                            i
                            for i, lc in enumerate(lower_cells)
                            if lc.chunk_indices.issubset(cell.chunk_indices)
                        ]
                        if boundary_indices:
                            boundary = cell_embeddings[dim - 1][boundary_indices].mean(dim=0)
                        else:
                            boundary = torch.zeros(self.hidden_dim, device=complex_.chunk_features.device)
                    else:
                        boundary = torch.zeros(self.hidden_dim, device=complex_.chunk_features.device)

                    boundary_embs.append(boundary)

                # Project direct embeddings
                direct_embs = torch.stack(direct_embs)
                direct_proj = self.encoders["0"](direct_embs)

                boundary_embs = torch.stack(boundary_embs)

                # Combine
                combined = torch.cat([direct_proj, boundary_embs], dim=-1)
                embs = self.encoders[str(dim)](combined)

            cell_embeddings[dim] = self.output_proj(embs)

        return cell_embeddings
