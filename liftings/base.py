"""
Base class for lifting transforms in TopoRAG.

A lifting transform takes a chunk graph (nodes = chunks, edges = similarity)
and produces higher-order structures (cells/hyperedges) that represent
groups of chunks that are jointly related.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Set, Dict, Optional, Tuple
import torch
import numpy as np


@dataclass
class Cell:
    """
    Represents a cell (higher-order structure) in the topological domain.

    A cell is a group of chunk indices that are jointly related.
    For example, in a simplicial complex, a 2-simplex is a triangle
    containing 3 chunks that form a coherent semantic unit.

    Attributes:
        chunk_indices: Set of chunk indices that comprise this cell
        dimension: The dimension of the cell (0=node, 1=edge, 2=face, etc.)
        cell_id: Unique identifier for this cell
        features: Optional feature vector for the cell
    """
    chunk_indices: Set[int]
    dimension: int
    cell_id: int
    features: Optional[torch.Tensor] = None

    def __hash__(self):
        return hash(frozenset(self.chunk_indices))

    def __eq__(self, other):
        if not isinstance(other, Cell):
            return False
        return self.chunk_indices == other.chunk_indices

    def __len__(self):
        return len(self.chunk_indices)

    def to_list(self) -> List[int]:
        """Return chunk indices as a sorted list."""
        return sorted(list(self.chunk_indices))


@dataclass
class TopologicalComplex:
    """
    Represents a topological complex (collection of cells at different dimensions).

    This is the output of a lifting transform. It contains:
    - The original chunk features
    - Cells organized by dimension
    - Incidence relations between cells of adjacent dimensions

    Attributes:
        chunk_features: (num_chunks, feature_dim) tensor of chunk embeddings
        cells_by_dim: Dict mapping dimension -> list of cells at that dimension
        num_chunks: Number of chunks (0-cells)
    """
    chunk_features: torch.Tensor
    cells_by_dim: Dict[int, List[Cell]]
    num_chunks: int

    def __post_init__(self):
        # Add 0-cells (individual chunks) if not present
        if 0 not in self.cells_by_dim:
            self.cells_by_dim[0] = [
                Cell(chunk_indices={i}, dimension=0, cell_id=i)
                for i in range(self.num_chunks)
            ]

    @property
    def max_dimension(self) -> int:
        """Return the maximum dimension of cells in the complex."""
        return max(self.cells_by_dim.keys())

    def get_cells(self, dimension: int) -> List[Cell]:
        """Get all cells of a given dimension."""
        return self.cells_by_dim.get(dimension, [])

    def get_all_higher_order_cells(self) -> List[Cell]:
        """Get all cells with dimension > 0 (i.e., groups of chunks)."""
        cells = []
        for dim in sorted(self.cells_by_dim.keys()):
            if dim > 0:
                cells.extend(self.cells_by_dim[dim])
        return cells

    def compute_cell_features(self, aggregation: str = "mean") -> Dict[int, torch.Tensor]:
        """
        Compute features for each cell by aggregating chunk features.

        Args:
            aggregation: How to aggregate chunk features ("mean", "sum", "max")

        Returns:
            Dict mapping dimension -> (num_cells_at_dim, feature_dim) tensor
        """
        cell_features_by_dim = {}

        for dim, cells in self.cells_by_dim.items():
            if dim == 0:
                # 0-cells are just the chunks themselves
                cell_features_by_dim[0] = self.chunk_features
            else:
                features_list = []
                for cell in cells:
                    indices = list(cell.chunk_indices)
                    chunk_feats = self.chunk_features[indices]  # (k, d)

                    if aggregation == "mean":
                        cell_feat = chunk_feats.mean(dim=0)
                    elif aggregation == "sum":
                        cell_feat = chunk_feats.sum(dim=0)
                    elif aggregation == "max":
                        cell_feat = chunk_feats.max(dim=0).values
                    else:
                        raise ValueError(f"Unknown aggregation: {aggregation}")

                    features_list.append(cell_feat)
                    cell.features = cell_feat

                if features_list:
                    cell_features_by_dim[dim] = torch.stack(features_list)

        return cell_features_by_dim

    def get_incidence_matrix(self, dim_low: int, dim_high: int) -> torch.Tensor:
        """
        Compute the incidence matrix between cells of adjacent dimensions.

        B[i,j] = 1 if cell_i (dim_low) is on the boundary of cell_j (dim_high)

        Args:
            dim_low: Lower dimension
            dim_high: Higher dimension (should be dim_low + 1)

        Returns:
            Binary incidence matrix of shape (num_cells_low, num_cells_high)
        """
        assert dim_high == dim_low + 1, "Dimensions must be adjacent"

        cells_low = self.get_cells(dim_low)
        cells_high = self.get_cells(dim_high)

        B = torch.zeros(len(cells_low), len(cells_high))

        for j, cell_high in enumerate(cells_high):
            for i, cell_low in enumerate(cells_low):
                # cell_low is on boundary of cell_high if cell_low ⊂ cell_high
                if cell_low.chunk_indices.issubset(cell_high.chunk_indices):
                    B[i, j] = 1

        return B


class BaseLiftingTransform(ABC):
    """
    Abstract base class for lifting transforms.

    A lifting transform takes chunk embeddings and a chunk graph
    and produces a TopologicalComplex with higher-order cells.
    """

    def __init__(self, max_cell_size: int = 5):
        """
        Args:
            max_cell_size: Maximum number of chunks in a cell
        """
        self.max_cell_size = max_cell_size

    @abstractmethod
    def lift(
        self,
        chunk_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> TopologicalComplex:
        """
        Lift a chunk graph to a topological complex.

        Args:
            chunk_features: (num_chunks, feature_dim) tensor
            edge_index: (2, num_edges) tensor of edge indices
            edge_attr: Optional (num_edges,) tensor of edge weights

        Returns:
            TopologicalComplex containing the lifted structure
        """
        pass

    def __call__(
        self,
        chunk_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> TopologicalComplex:
        return self.lift(chunk_features, edge_index, edge_attr)
