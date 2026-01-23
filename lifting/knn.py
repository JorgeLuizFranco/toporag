"""
K-NN Hypergraph Lifting: Graph → Hypergraph

Creates hyperedges by grouping each node with its k nearest neighbors.
This is useful when you want to capture local semantic neighborhoods.

From difflifting:
"k-NN lifting constructs hyperedges by identifying the k nearest neighbors
based on their node features (feature space). For every node, a separate
hyperedge is formed that includes the node itself and its k closest neighbors."
"""

import torch
from typing import Optional
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

from .base import BaseLiftingTransform, LiftedTopology


class KNNHypergraphLifting(BaseLiftingTransform):
    """
    Lift graph to hypergraph using k-NN in feature space.

    Creates hyperedges by grouping each node with its k nearest neighbors.
    Unlike cycle/clique lifting, this operates directly on node features
    rather than graph structure.

    Args:
        k: Number of nearest neighbors per hyperedge
        distance_metric: 'cosine' or 'euclidean'
        deduplicate: Remove duplicate hyperedges
        feature_lifting: How to lift features
    """

    def __init__(
        self,
        k: int = 5,
        distance_metric: str = "cosine",
        deduplicate: bool = True,
        feature_lifting: str = "projection_sum",
    ):
        super().__init__(complex_dim=1, feature_lifting=feature_lifting)  # Hypergraph = dim 1
        self.k = k
        self.distance_metric = distance_metric
        self.deduplicate = deduplicate

    def lift(self, data: Data) -> LiftedTopology:
        """
        Lift to hypergraph via k-NN.

        Args:
            data: PyG Data with x and edge_index

        Returns:
            LiftedTopology with node-hyperedge incidence
        """
        num_nodes = data.x.shape[0]
        node_features = data.x
        features_np = node_features.detach().cpu().numpy()

        # Fit k-NN
        actual_k = min(self.k + 1, num_nodes)  # +1 because includes self
        knn = NearestNeighbors(
            n_neighbors=actual_k,
            metric="cosine" if self.distance_metric == "cosine" else "euclidean",
        )
        knn.fit(features_np)
        distances, indices = knn.kneighbors(features_np)

        # Create hyperedges
        hyperedges = []
        seen = set()

        for anchor in range(num_nodes):
            neighbors = set(indices[anchor].tolist())

            if self.deduplicate:
                key = frozenset(neighbors)
                if key in seen:
                    continue
                seen.add(key)

            hyperedges.append(list(neighbors))

        num_hyperedges = len(hyperedges)
        cell_to_nodes = {i: he for i, he in enumerate(hyperedges)}

        # Build node-hyperedge incidence (incidence_1 in hypergraph sense)
        if num_hyperedges > 0:
            rows, cols, vals = [], [], []
            for he_idx, nodes in enumerate(hyperedges):
                for node in nodes:
                    rows.append(node)
                    cols.append(he_idx)
                    vals.append(1.0)

            indices_tensor = torch.tensor([rows, cols], dtype=torch.long)
            values = torch.tensor(vals, dtype=torch.float)
            incidence_1 = torch.sparse_coo_tensor(
                indices_tensor, values, size=(num_nodes, num_hyperedges)
            ).coalesce()
        else:
            incidence_1 = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long),
                torch.empty(0),
                size=(num_nodes, 0)
            )

        # Lift features to hyperedges
        if num_hyperedges > 0:
            x_1 = self._lift_features(node_features, incidence_1.to_dense())
        else:
            x_1 = torch.empty((0, node_features.shape[1]))

        # Build adjacency (hyperedge co-occurrence)
        adjacency_0 = self._build_adjacency_from_incidence(incidence_1) if num_hyperedges > 0 else None

        return LiftedTopology(
            x_0=node_features,
            x_1=x_1,
            x_2=None,  # No 2-cells in hypergraph
            incidence_1=incidence_1,
            incidence_2=None,
            adjacency_0=adjacency_0,
            adjacency_1=None,
            num_nodes=num_nodes,
            num_edges=num_hyperedges,  # Hyperedges as "edges"
            num_2cells=0,
            cells=hyperedges,
            cell_to_nodes=cell_to_nodes,
        )
