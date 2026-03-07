"""
K-NN Hypergraph Lifting: Graph → Hypergraph (GPU Optimized)

Creates hyperedges by grouping each node with its k nearest neighbors.
Uses pure PyTorch for GPU acceleration.
"""

import torch
from typing import Optional
from torch_geometric.data import Data

from .base import BaseLiftingTransform, LiftedTopology


class KNNHypergraphLifting(BaseLiftingTransform):
    """
    Lift graph to hypergraph using k-NN in feature space.
    GPU Optimized version using torch.topk.
    """

    def __init__(
        self,
        k: int = 5,
        distance_metric: str = "cosine",
        deduplicate: bool = True,
        feature_lifting: str = "projection_sum",
    ):
        super().__init__(complex_dim=1, feature_lifting=feature_lifting)
        self.k = k
        self.distance_metric = distance_metric
        self.deduplicate = deduplicate

    def lift(self, data: Data) -> LiftedTopology:
        """
        Lift to hypergraph via k-NN on GPU.
        """
        device = data.x.device
        num_nodes = data.x.shape[0]
        node_features = data.x

        # 1. GPU-Accelerated KNN
        # Normalize for cosine similarity if needed
        if self.distance_metric == "cosine":
            norm_features = torch.nn.functional.normalize(node_features, p=2, dim=1)
            sims = torch.mm(norm_features, norm_features.t())
        else:
            # Euclidean distance squared
            sims = -torch.cdist(node_features, node_features, p=2)

        actual_k = min(self.k + 1, num_nodes)
        _, indices = torch.topk(sims, k=actual_k, dim=1)

        # 2. Build hyperedges
        from toporag.lifting.base import Cell
        
        hyperedges = []
        rows, cols, vals = [], [], []
        seen = set()

        # Optimize for 8GB GPU: Move to CPU only for the set deduplication logic if N is very large
        indices_np = indices.detach().cpu().numpy()
        
        for anchor in range(num_nodes):
            neighbors = indices_np[anchor].tolist()
            if self.deduplicate:
                key = frozenset(neighbors)
                if key in seen:
                    continue
                seen.add(key)

            he_idx = len(hyperedges)
            hyperedges.append(Cell(
                chunk_indices=set(neighbors),
                dimension=1,
                cell_id=he_idx
            ))
            
            # For incidence matrix
            for node in neighbors:
                rows.append(node)
                cols.append(he_idx)
                vals.append(1.0)

        num_hyperedges = len(hyperedges)
        cell_to_nodes = {i: list(he.chunk_indices) for i, he in enumerate(hyperedges)}

        # 3. Build node-hyperedge incidence (Sparse on GPU)
        if num_hyperedges > 0:
            indices_tensor = torch.tensor([rows, cols], dtype=torch.long, device=device)
            values = torch.tensor(vals, dtype=torch.float, device=device)
            incidence_1 = torch.sparse_coo_tensor(
                indices_tensor, values, size=(num_nodes, num_hyperedges)
            ).coalesce()
            
            # Lift features (using sparse mm to save memory)
            x_1 = torch.sparse.mm(incidence_1.t(), node_features)
            if self.feature_lifting == "mean":
                counts = torch.sparse.sum(incidence_1, dim=0).to_dense().unsqueeze(1)
                x_1 = x_1 / counts.clamp(min=1)
        else:
            incidence_1 = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty(0, device=device),
                size=(num_nodes, 0)
            )
            x_1 = torch.empty((0, node_features.shape[1]), device=device)

        # Adjacency
        adjacency_0 = self._build_adjacency_from_incidence(incidence_1) if num_hyperedges > 0 else None

        return LiftedTopology(
            x_0=node_features,
            x_1=x_1,
            x_2=None,
            incidence_1=incidence_1,
            incidence_2=None,
            adjacency_0=adjacency_0,
            adjacency_1=None,
            num_nodes=num_nodes,
            num_edges=num_hyperedges,
            cells=hyperedges,
            cell_to_nodes=cell_to_nodes,
        )