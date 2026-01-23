"""
Clique Lifting: Graph → Simplicial Complex

Finds cliques in the graph and creates simplices from them.
This captures tightly connected groups of chunks.

From difflifting/TopoNetX:
"Clique lifting produces a simplicial complex by leveraging
cliques in the input graph."
"""

import torch
import networkx as nx
from typing import Optional, List
from torch_geometric.data import Data
from itertools import combinations

from .base import BaseLiftingTransform, LiftedTopology


class CliqueLifting(BaseLiftingTransform):
    """
    Lift graph to simplicial complex using clique enumeration.

    Finds all cliques up to a maximum size and creates simplices.
    - 3-cliques (triangles) → 2-simplices
    - 4-cliques → 3-simplices (if complex_dim allows)

    Args:
        max_clique_size: Maximum clique size to consider
        min_clique_size: Minimum clique size (3 = triangles)
        feature_lifting: How to lift features
    """

    def __init__(
        self,
        max_clique_size: int = 4,
        min_clique_size: int = 3,
        feature_lifting: str = "projection_sum",
    ):
        super().__init__(complex_dim=2, feature_lifting=feature_lifting)
        self.max_clique_size = max_clique_size
        self.min_clique_size = min_clique_size

    def lift(self, data: Data) -> LiftedTopology:
        """
        Lift graph to simplicial complex via clique detection.

        Args:
            data: PyG Data with x and edge_index

        Returns:
            LiftedTopology with incidence matrices
        """
        num_nodes = data.x.shape[0]
        node_features = data.x

        # Convert to NetworkX
        G = self._graph_from_data(data)

        # Build incidence_1 (node-edge)
        incidence_1, edges = self._build_incidence_1(G, num_nodes)
        num_edges = len(edges)
        edge_to_idx = {(min(e), max(e)): i for i, e in enumerate(edges)}

        # Find cliques
        cliques = list(nx.enumerate_all_cliques(G))

        # Filter by size (only keep cliques that form 2-cells, i.e., 3+ nodes)
        triangles = [
            c for c in cliques
            if self.min_clique_size <= len(c) <= self.max_clique_size
        ]

        # Build incidence_2 (edge-triangle)
        # A triangle has 3 edges, each edge is incident to the triangle
        num_2cells = len(triangles)
        cell_to_nodes = {}

        if num_2cells > 0 and num_edges > 0:
            rows, cols, vals = [], [], []

            for cell_idx, clique in enumerate(triangles):
                cell_to_nodes[cell_idx] = list(clique)

                # For each pair of nodes in clique, find the edge
                for i, j in combinations(clique, 2):
                    edge_key = (min(i, j), max(i, j))
                    if edge_key in edge_to_idx:
                        rows.append(edge_to_idx[edge_key])
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
            cell_node_incidence = self._build_cell_node_incidence(
                triangles, num_nodes, num_2cells
            )
            x_2 = self._lift_features(node_features, cell_node_incidence)
        else:
            x_2 = torch.empty((0, node_features.shape[1]))

        # Build adjacency
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
            cells=triangles,
            cell_to_nodes=cell_to_nodes,
        )

    def _build_cell_node_incidence(
        self,
        cliques: List[List[int]],
        num_nodes: int,
        num_cells: int,
    ) -> torch.Tensor:
        """Build direct node-to-cell incidence for feature lifting."""
        rows, cols, vals = [], [], []

        for cell_idx, clique in enumerate(cliques):
            for node in clique:
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
