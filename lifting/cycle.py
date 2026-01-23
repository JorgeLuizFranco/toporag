"""
Cycle Lifting: Graph → Cell Complex

Detects cycles in the graph and creates 2-cells from them.
This is particularly useful for RAG because cycles often represent
semantic relationships that require joint information.

From difflifting:
"The algorithm creates 2-cells by identifying the cycles and
considering them as 2-cells."
"""

import torch
import networkx as nx
from typing import Optional, List, Set
from torch_geometric.data import Data

from .base import BaseLiftingTransform, LiftedTopology


class CycleLifting(BaseLiftingTransform):
    """
    Lift graph to cell complex using cycle detection.

    Finds cycles in the graph using NetworkX's cycle_basis
    and creates 2-cells from them.

    Args:
        max_cycle_length: Maximum length of cycles to consider (None = no limit)
        min_cycle_length: Minimum length of cycles (default 3 = triangles)
        feature_lifting: How to lift features ('projection_sum', 'mean')
    """

    def __init__(
        self,
        max_cycle_length: Optional[int] = None,
        min_cycle_length: int = 3,
        feature_lifting: str = "projection_sum",
    ):
        super().__init__(complex_dim=2, feature_lifting=feature_lifting)
        self.max_cycle_length = max_cycle_length
        self.min_cycle_length = min_cycle_length

    def lift(self, data: Data) -> LiftedTopology:
        """
        Lift graph to cell complex via cycle detection.

        Args:
            data: PyG Data with x and edge_index

        Returns:
            LiftedTopology with incidence matrices and features
        """
        num_nodes = data.x.shape[0]
        node_features = data.x

        # Convert to NetworkX
        G = self._graph_from_data(data)

        # Build incidence_1 (node-edge)
        incidence_1, edges = self._build_incidence_1(G, num_nodes)
        num_edges = len(edges)

        # Create edge index mapping for incidence_2
        edge_to_idx = {(min(e), max(e)): i for i, e in enumerate(edges)}

        # Find cycles
        cycles = nx.cycle_basis(G)

        # Filter cycles by length
        cycles = [c for c in cycles if len(c) >= self.min_cycle_length]
        if self.max_cycle_length is not None:
            cycles = [c for c in cycles if len(c) <= self.max_cycle_length]

        # Build incidence_2 (edge-2cell)
        num_2cells = len(cycles)
        cell_to_nodes = {}

        if num_2cells > 0 and num_edges > 0:
            rows, cols, vals = [], [], []

            for cell_idx, cycle in enumerate(cycles):
                cell_to_nodes[cell_idx] = cycle

                # Find edges in this cycle
                cycle_edges = []
                for i in range(len(cycle)):
                    src = cycle[i]
                    dst = cycle[(i + 1) % len(cycle)]
                    edge_key = (min(src, dst), max(src, dst))
                    if edge_key in edge_to_idx:
                        cycle_edges.append(edge_to_idx[edge_key])

                # Add to incidence matrix
                for edge_idx in cycle_edges:
                    rows.append(edge_idx)
                    cols.append(cell_idx)
                    vals.append(1.0)

            if rows:
                indices = torch.tensor([rows, cols], dtype=torch.long)
                values = torch.tensor(vals, dtype=torch.float)
                incidence_2 = torch.sparse_coo_tensor(
                    indices, values, size=(num_edges, num_2cells)
                ).coalesce()
            else:
                incidence_2 = torch.sparse_coo_tensor(
                    torch.empty((2, 0), dtype=torch.long),
                    torch.empty(0),
                    size=(num_edges, 0)
                )
        else:
            incidence_2 = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long),
                torch.empty(0),
                size=(num_edges, 0)
            )

        # Lift features
        x_1 = self._lift_features(node_features, incidence_1.to_dense())

        if num_2cells > 0:
            # For 2-cells, lift from nodes directly (sum of node features in cycle)
            cell_node_incidence = self._build_cell_node_incidence(
                cycles, num_nodes, num_2cells
            )
            x_2 = self._lift_features(node_features, cell_node_incidence)
        else:
            x_2 = torch.empty((0, node_features.shape[1]))

        # Build adjacency matrices
        adjacency_0 = self._build_adjacency_from_incidence(incidence_1)
        adjacency_1 = self._build_adjacency_from_incidence(incidence_2) if num_2cells > 0 else None

        return LiftedTopology(
            x_0=node_features,
            x_1=x_1,
            x_2=x_2,
            incidence_1=incidence_1,
            incidence_2=incidence_2,
            adjacency_0=adjacency_0,
            adjacency_1=adjacency_1,
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_2cells=num_2cells,
            cells=cycles,
            cell_to_nodes=cell_to_nodes,
        )

    def _build_cell_node_incidence(
        self,
        cycles: List[List[int]],
        num_nodes: int,
        num_cells: int,
    ) -> torch.Tensor:
        """Build direct node-to-cell incidence for feature lifting."""
        rows, cols, vals = [], [], []

        for cell_idx, cycle in enumerate(cycles):
            for node in cycle:
                rows.append(node)
                cols.append(cell_idx)
                vals.append(1.0)

        if rows:
            indices = torch.tensor([rows, cols], dtype=torch.long)
            values = torch.tensor(vals, dtype=torch.float)
            return torch.sparse_coo_tensor(
                indices, values, size=(num_nodes, num_cells)
            ).coalesce().to_dense()
        else:
            return torch.zeros((num_nodes, num_cells))
