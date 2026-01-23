"""
k-NN Lifting: Creates hyperedges from k nearest neighbors in embedding space.

This lifting creates higher-order cells by grouping each chunk with its k
nearest neighbors in the embedding space. This is particularly suitable
for TopoRAG since chunk embeddings capture semantic similarity.

From difflifting.tex:
"k-NN lifting constructs hyperedges by identifying the k nearest neighbors
based on their node features (feature space). For every node, a separate
hyperedge is formed that includes the node itself and its k closest neighbors."
"""

from typing import Optional, List, Set
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import BaseLiftingTransform, TopologicalComplex, Cell


class KNNLifting(BaseLiftingTransform):
    """
    k-NN Lifting: Creates cells from k nearest neighbors.

    For each chunk, we create a cell containing the chunk and its k nearest
    neighbors in embedding space. This captures semantic similarity groups.

    Args:
        k: Number of nearest neighbors to include in each cell
        distance_metric: Distance metric for k-NN ('cosine', 'euclidean')
        deduplicate: Whether to remove duplicate cells (cells with same chunks)
        min_cell_size: Minimum number of chunks in a cell (default: 2)
    """

    def __init__(
        self,
        k: int = 3,
        distance_metric: str = "cosine",
        deduplicate: bool = True,
        min_cell_size: int = 2,
    ):
        super().__init__(max_cell_size=k + 1)  # +1 for the anchor chunk
        self.k = k
        self.distance_metric = distance_metric
        self.deduplicate = deduplicate
        self.min_cell_size = min_cell_size

    def lift(
        self,
        chunk_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> TopologicalComplex:
        """
        Lift chunk graph to hypergraph using k-NN.

        Args:
            chunk_features: (num_chunks, feature_dim) tensor of chunk embeddings
            edge_index: (2, num_edges) tensor - used for 1-cells but not for k-NN
            edge_attr: Optional edge weights - not used for k-NN lifting

        Returns:
            TopologicalComplex with:
            - 0-cells: individual chunks
            - 1-cells: edges from edge_index (if provided)
            - Higher-order cells: k-NN hyperedges
        """
        num_chunks = chunk_features.shape[0]
        features_np = chunk_features.detach().cpu().numpy()

        # Fit k-NN model
        # Use k+1 because the nearest neighbor of a point includes itself
        knn = NearestNeighbors(
            n_neighbors=min(self.k + 1, num_chunks),
            metric="cosine" if self.distance_metric == "cosine" else "euclidean",
        )
        knn.fit(features_np)

        # Find k nearest neighbors for each chunk
        distances, indices = knn.kneighbors(features_np)

        # Create cells from k-NN groups
        cells_by_dim = {}

        # Add 1-cells from edge_index if provided
        if edge_index is not None and edge_index.numel() > 0:
            edge_cells = []
            seen_edges = set()
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                edge_set = frozenset({src, dst})
                if edge_set not in seen_edges:
                    seen_edges.add(edge_set)
                    edge_cells.append(
                        Cell(
                            chunk_indices={src, dst},
                            dimension=1,
                            cell_id=len(edge_cells),
                        )
                    )
            cells_by_dim[1] = edge_cells

        # Create higher-order cells from k-NN
        # The dimension depends on the cell size:
        # - 2 chunks = dimension 1 (edge)
        # - 3 chunks = dimension 2 (triangle/hyperedge)
        # - k+1 chunks = dimension k
        higher_order_cells = []
        seen_cells = set()
        cell_id = 0

        for anchor_idx in range(num_chunks):
            # Get k nearest neighbors (including self)
            neighbor_indices = indices[anchor_idx]

            # Create cell with anchor and its neighbors
            cell_indices = set(neighbor_indices.tolist())

            # Skip if cell is too small
            if len(cell_indices) < self.min_cell_size:
                continue

            # Deduplicate if requested
            cell_key = frozenset(cell_indices)
            if self.deduplicate and cell_key in seen_cells:
                continue
            seen_cells.add(cell_key)

            # Determine dimension based on cell size
            # A cell with n elements has dimension n-1 in simplicial terms
            # But for hypergraphs, we treat all hyperedges as dimension 1
            # Here we use dimension = len(cell) - 1 for flexibility
            dim = len(cell_indices) - 1

            cell = Cell(
                chunk_indices=cell_indices,
                dimension=dim,
                cell_id=cell_id,
            )
            higher_order_cells.append(cell)
            cell_id += 1

        # Group cells by dimension
        for cell in higher_order_cells:
            dim = cell.dimension
            if dim not in cells_by_dim:
                cells_by_dim[dim] = []
            cells_by_dim[dim].append(cell)

        # Reassign cell IDs per dimension
        for dim in cells_by_dim:
            for i, cell in enumerate(cells_by_dim[dim]):
                cell.cell_id = i

        return TopologicalComplex(
            chunk_features=chunk_features,
            cells_by_dim=cells_by_dim,
            num_chunks=num_chunks,
        )


class AdaptiveKNNLifting(BaseLiftingTransform):
    """
    Adaptive k-NN Lifting: Creates cells with varying sizes based on local density.

    Instead of fixed k, this uses different k values based on local density
    or a range [k_min, k_max]. This can better capture varying levels of
    semantic granularity in the chunk space.

    Args:
        k_min: Minimum number of neighbors
        k_max: Maximum number of neighbors
        distance_metric: Distance metric for k-NN
        density_based: If True, k is determined by local density
    """

    def __init__(
        self,
        k_min: int = 2,
        k_max: int = 5,
        distance_metric: str = "cosine",
        density_based: bool = False,
    ):
        super().__init__(max_cell_size=k_max + 1)
        self.k_min = k_min
        self.k_max = k_max
        self.distance_metric = distance_metric
        self.density_based = density_based

    def lift(
        self,
        chunk_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> TopologicalComplex:
        """
        Lift with adaptive k values.
        """
        num_chunks = chunk_features.shape[0]
        features_np = chunk_features.detach().cpu().numpy()

        # Fit k-NN with maximum k
        knn = NearestNeighbors(
            n_neighbors=min(self.k_max + 1, num_chunks),
            metric="cosine" if self.distance_metric == "cosine" else "euclidean",
        )
        knn.fit(features_np)
        distances, indices = knn.kneighbors(features_np)

        cells_by_dim = {}
        higher_order_cells = []
        seen_cells = set()
        cell_id = 0

        for anchor_idx in range(num_chunks):
            # Determine k for this anchor
            if self.density_based:
                # Use inverse of average distance to determine density
                avg_dist = np.mean(distances[anchor_idx, 1:])  # Exclude self
                # Higher density (lower avg_dist) -> smaller k
                density_ratio = 1.0 / (avg_dist + 1e-6)
                k = int(
                    self.k_min
                    + (self.k_max - self.k_min)
                    * min(1.0, density_ratio / np.mean(1.0 / (distances[:, 1:].mean(axis=1) + 1e-6)))
                )
                k = max(self.k_min, min(self.k_max, k))
            else:
                # Create cells for each k in range
                for k in range(self.k_min, self.k_max + 1):
                    neighbor_indices = indices[anchor_idx, : k + 1]
                    cell_indices = set(neighbor_indices.tolist())

                    cell_key = frozenset(cell_indices)
                    if cell_key in seen_cells:
                        continue
                    seen_cells.add(cell_key)

                    dim = len(cell_indices) - 1
                    cell = Cell(
                        chunk_indices=cell_indices,
                        dimension=dim,
                        cell_id=cell_id,
                    )
                    higher_order_cells.append(cell)
                    cell_id += 1
                continue

            # Single k case (density_based=True)
            neighbor_indices = indices[anchor_idx, : k + 1]
            cell_indices = set(neighbor_indices.tolist())

            cell_key = frozenset(cell_indices)
            if cell_key not in seen_cells:
                seen_cells.add(cell_key)
                dim = len(cell_indices) - 1
                cell = Cell(
                    chunk_indices=cell_indices,
                    dimension=dim,
                    cell_id=cell_id,
                )
                higher_order_cells.append(cell)
                cell_id += 1

        # Group by dimension
        for cell in higher_order_cells:
            dim = cell.dimension
            if dim not in cells_by_dim:
                cells_by_dim[dim] = []
            cells_by_dim[dim].append(cell)

        for dim in cells_by_dim:
            for i, cell in enumerate(cells_by_dim[dim]):
                cell.cell_id = i

        return TopologicalComplex(
            chunk_features=chunk_features,
            cells_by_dim=cells_by_dim,
            num_chunks=num_chunks,
        )
