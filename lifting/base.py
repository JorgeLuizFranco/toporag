"""
Base classes for topological lifting in TopoRAG.

Provides infrastructure for lifting graphs to higher-order structures
(cell complexes, simplicial complexes, hypergraphs) with proper
incidence matrices for TNN message passing.
"""

import torch
import networkx as nx
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from torch_geometric.data import Data


@dataclass
class LiftedTopology:
    """
    Container for lifted topological structure.

    Stores all connectivity information needed for TNN message passing:
    - Node features (x_0)
    - Edge features (x_1)
    - Cell/simplex features (x_2)
    - Incidence matrices
    - Adjacency matrices
    - Laplacians (optional)

    Following TopoModelX conventions.
    """
    # Features at each dimension
    x_0: torch.Tensor  # (num_nodes, dim) node features
    x_1: Optional[torch.Tensor] = None  # (num_edges, dim) edge features
    x_2: Optional[torch.Tensor] = None  # (num_2cells, dim) 2-cell features

    # Incidence matrices (boundary operators)
    incidence_1: Optional[torch.Tensor] = None  # (num_nodes, num_edges) node-edge
    incidence_2: Optional[torch.Tensor] = None  # (num_edges, num_2cells) edge-2cell

    # Adjacency matrices
    adjacency_0: Optional[torch.Tensor] = None  # (num_nodes, num_nodes)
    adjacency_1: Optional[torch.Tensor] = None  # (num_edges, num_edges)
    adjacency_2: Optional[torch.Tensor] = None  # (num_2cells, num_2cells)

    # Hodge Laplacians (optional, can be computed from incidence)
    hodge_laplacian_0: Optional[torch.Tensor] = None
    hodge_laplacian_1: Optional[torch.Tensor] = None
    hodge_laplacian_2: Optional[torch.Tensor] = None

    # Metadata
    num_nodes: int = 0
    num_edges: int = 0
    num_2cells: int = 0

    # Cell information (for retrieval)
    cells: List[Any] = field(default_factory=list)  # List of cell objects
    cell_to_nodes: Dict[int, List[int]] = field(default_factory=dict)  # cell_idx -> node indices

    def to(self, device: str) -> 'LiftedTopology':
        """Move all tensors to device."""
        def _to_device(t):
            return t.to(device) if t is not None else None

        return LiftedTopology(
            x_0=_to_device(self.x_0),
            x_1=_to_device(self.x_1),
            x_2=_to_device(self.x_2),
            incidence_1=_to_device(self.incidence_1),
            incidence_2=_to_device(self.incidence_2),
            adjacency_0=_to_device(self.adjacency_0),
            adjacency_1=_to_device(self.adjacency_1),
            adjacency_2=_to_device(self.adjacency_2),
            hodge_laplacian_0=_to_device(self.hodge_laplacian_0),
            hodge_laplacian_1=_to_device(self.hodge_laplacian_1),
            hodge_laplacian_2=_to_device(self.hodge_laplacian_2),
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            num_2cells=self.num_2cells,
            cells=self.cells,
            cell_to_nodes=self.cell_to_nodes,
        )

    def compute_laplacians(self):
        """Compute Hodge Laplacians from incidence matrices."""
        # L_0 = B_1 @ B_1^T (up Laplacian at nodes)
        if self.incidence_1 is not None:
            B1 = self.incidence_1
            if B1.is_sparse:
                self.hodge_laplacian_0 = torch.sparse.mm(B1, B1.t())
            else:
                self.hodge_laplacian_0 = B1 @ B1.t()

        # L_1 = B_1^T @ B_1 + B_2 @ B_2^T (Hodge Laplacian at edges)
        if self.incidence_1 is not None:
            B1 = self.incidence_1
            L1_down = B1.t() @ B1 if not B1.is_sparse else torch.sparse.mm(B1.t(), B1)

            if self.incidence_2 is not None:
                B2 = self.incidence_2
                L1_up = B2 @ B2.t() if not B2.is_sparse else torch.sparse.mm(B2, B2.t())
                self.hodge_laplacian_1 = L1_down + L1_up
            else:
                self.hodge_laplacian_1 = L1_down

        # L_2 = B_2^T @ B_2 (down Laplacian at 2-cells)
        if self.incidence_2 is not None:
            B2 = self.incidence_2
            if B2.is_sparse:
                self.hodge_laplacian_2 = torch.sparse.mm(B2.t(), B2)
            else:
                self.hodge_laplacian_2 = B2.t() @ B2


class BaseLiftingTransform(ABC):
    """
    Base class for topological lifting transformations.

    Converts a PyG graph to a higher-order topological structure
    with proper incidence matrices for TNN message passing.
    """

    def __init__(
        self,
        complex_dim: int = 2,
        feature_lifting: str = "projection_sum",
    ):
        """
        Args:
            complex_dim: Maximum dimension of the complex (2 for cell/simplicial)
            feature_lifting: How to lift node features to higher cells
                           ('projection_sum', 'mean', 'attention')
        """
        self.complex_dim = complex_dim
        self.feature_lifting = feature_lifting

    @abstractmethod
    def lift(self, data: Data) -> LiftedTopology:
        """
        Lift a graph to a higher-order topological structure.

        Args:
            data: PyG Data object with x (features) and edge_index

        Returns:
            LiftedTopology with all connectivity matrices
        """
        pass

    def _graph_from_data(self, data: Data) -> nx.Graph:
        """Convert PyG Data to NetworkX graph."""
        G = nx.Graph()

        # Add nodes with features
        for i in range(data.x.shape[0]):
            G.add_node(i, features=data.x[i])

        # Add edges
        edge_index = data.edge_index
        for j in range(edge_index.shape[1]):
            src, dst = edge_index[0, j].item(), edge_index[1, j].item()
            if src < dst:  # Avoid duplicates for undirected
                G.add_edge(src, dst)

        return G

    def _build_incidence_1(self, G: nx.Graph, num_nodes: int) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Build node-edge incidence matrix.

        Returns:
            incidence_1: (num_nodes, num_edges) tensor
            edges: List of (src, dst) tuples
        """
        edges = list(G.edges())
        num_edges = len(edges)

        # Build sparse incidence matrix
        rows, cols, vals = [], [], []
        for edge_idx, (src, dst) in enumerate(edges):
            rows.extend([src, dst])
            cols.extend([edge_idx, edge_idx])
            vals.extend([1.0, 1.0])

        if num_edges > 0:
            indices = torch.tensor([rows, cols], dtype=torch.long)
            values = torch.tensor(vals, dtype=torch.float)
            incidence_1 = torch.sparse_coo_tensor(
                indices, values, size=(num_nodes, num_edges)
            ).coalesce()
        else:
            incidence_1 = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long),
                torch.empty(0),
                size=(num_nodes, 0)
            )

        return incidence_1, edges

    def _lift_features(
        self,
        node_features: torch.Tensor,
        incidence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Lift node features to higher-order cells.

        Args:
            node_features: (num_nodes, dim) tensor
            incidence: (num_nodes, num_cells) incidence matrix

        Returns:
            (num_cells, dim) tensor of cell features
        """
        if self.feature_lifting == "projection_sum":
            # Sum of node features in each cell
            if incidence.is_sparse:
                cell_features = torch.sparse.mm(incidence.t(), node_features)
            else:
                cell_features = incidence.t() @ node_features
        elif self.feature_lifting == "mean":
            # Mean of node features in each cell
            if incidence.is_sparse:
                sums = torch.sparse.mm(incidence.t(), node_features)
                counts = torch.sparse.sum(incidence, dim=0).to_dense().unsqueeze(1)
            else:
                sums = incidence.t() @ node_features
                counts = incidence.sum(dim=0, keepdim=True).t()
            cell_features = sums / counts.clamp(min=1)
        else:
            raise ValueError(f"Unknown feature lifting: {self.feature_lifting}")

        return cell_features

    def _build_adjacency_from_incidence(self, incidence: torch.Tensor) -> torch.Tensor:
        """Build adjacency matrix from incidence (A = B @ B^T - diag)."""
        if incidence.is_sparse:
            adj = torch.sparse.mm(incidence, incidence.t())
        else:
            adj = incidence @ incidence.t()

        # Remove diagonal (self-loops)
        if adj.is_sparse:
            adj = adj.to_dense()
        adj.fill_diagonal_(0)

        return adj.to_sparse_coo()
