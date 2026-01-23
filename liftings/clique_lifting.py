"""
Clique Lifting: Creates simplicial complex from cliques in the chunk graph.

This lifting identifies cliques (complete subgraphs) and uses them to form
a simplicial complex. A k-clique becomes a (k-1)-simplex.

From difflifting.tex:
"The set of cliques in a graph G is given by Cl(G) = {c ⊆ V(G): u ≠ v ∈ c ⟹ {u,v} ∈ E(G)},
i.e., each element of Cl(G) is a complete subgraph of G."
"""

from typing import Optional, List, Set
import torch
import numpy as np
import networkx as nx
from itertools import combinations

from .base import BaseLiftingTransform, TopologicalComplex, Cell


class CliqueLifting(BaseLiftingTransform):
    """
    Clique Lifting: Creates simplicial complex from cliques.

    A simplicial complex is formed where:
    - 0-simplices: individual chunks (vertices)
    - 1-simplices: edges (2-cliques)
    - 2-simplices: triangles (3-cliques)
    - k-simplices: (k+1)-cliques

    The key property is that a simplicial complex is closed under taking
    subsets: if a k-clique is included, all its sub-cliques are also included.

    Args:
        max_clique_size: Maximum size of cliques to find
        include_all_subfaces: If True, include all subfaces (simplicial closure)
    """

    def __init__(
        self,
        max_clique_size: int = 5,
        include_all_subfaces: bool = True,
    ):
        super().__init__(max_cell_size=max_clique_size)
        self.max_clique_size = max_clique_size
        self.include_all_subfaces = include_all_subfaces

    def _edge_index_to_networkx(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> nx.Graph:
        """Convert edge_index to NetworkX graph."""
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edge_list = edge_index.t().tolist()
        G.add_edges_from(edge_list)
        return G

    def _find_cliques(self, G: nx.Graph) -> List[Set[int]]:
        """
        Find all maximal cliques up to max_clique_size.
        """
        cliques = []

        # Use NetworkX to find all maximal cliques
        for clique in nx.find_cliques(G):
            if len(clique) <= self.max_clique_size:
                cliques.append(set(clique))
            elif len(clique) > self.max_clique_size:
                # For large cliques, enumerate all subcliques of max_clique_size
                for sub in combinations(clique, self.max_clique_size):
                    cliques.append(set(sub))

        return cliques

    def _get_all_subfaces(self, clique: Set[int]) -> List[Set[int]]:
        """
        Get all non-empty subsets of a clique (simplicial closure).

        For a k-clique, returns all subsets from size 1 to k.
        """
        subfaces = []
        clique_list = list(clique)
        n = len(clique_list)

        for size in range(1, n + 1):
            for subset in combinations(clique_list, size):
                subfaces.append(set(subset))

        return subfaces

    def lift(
        self,
        chunk_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> TopologicalComplex:
        """
        Lift chunk graph to simplicial complex using cliques.

        Args:
            chunk_features: (num_chunks, feature_dim) tensor
            edge_index: (2, num_edges) tensor of edge indices
            edge_attr: Optional edge weights (not used)

        Returns:
            TopologicalComplex representing a simplicial complex
        """
        num_chunks = chunk_features.shape[0]
        cells_by_dim = {}

        if edge_index is None or edge_index.numel() == 0:
            # No edges, return just 0-cells
            return TopologicalComplex(
                chunk_features=chunk_features,
                cells_by_dim=cells_by_dim,
                num_chunks=num_chunks,
            )

        # Convert to NetworkX
        G = self._edge_index_to_networkx(edge_index, num_chunks)

        # Find all maximal cliques
        maximal_cliques = self._find_cliques(G)

        # Collect all simplices (cells)
        all_simplices = set()

        for clique in maximal_cliques:
            if self.include_all_subfaces:
                # Add all subfaces for simplicial closure
                for subface in self._get_all_subfaces(clique):
                    all_simplices.add(frozenset(subface))
            else:
                # Only add the clique itself
                all_simplices.add(frozenset(clique))

        # Organize by dimension
        simplices_by_dim = {}
        for simplex in all_simplices:
            dim = len(simplex) - 1  # k vertices = (k-1)-simplex
            if dim not in simplices_by_dim:
                simplices_by_dim[dim] = []
            simplices_by_dim[dim].append(set(simplex))

        # Create Cell objects
        for dim, simplices in simplices_by_dim.items():
            if dim == 0:
                continue  # 0-cells handled automatically
            cells_by_dim[dim] = [
                Cell(
                    chunk_indices=simplex,
                    dimension=dim,
                    cell_id=i,
                )
                for i, simplex in enumerate(simplices)
            ]

        return TopologicalComplex(
            chunk_features=chunk_features,
            cells_by_dim=cells_by_dim,
            num_chunks=num_chunks,
        )


class KCliqueLifting(BaseLiftingTransform):
    """
    k-Clique Lifting: Creates cells only from cliques of a specific size.

    Unlike full clique lifting, this only creates k-simplices from
    (k+1)-cliques, without the simplicial closure. This results in
    a hypergraph rather than a simplicial complex.

    Args:
        k: The specific clique size to extract (creates (k-1)-dimensional cells)
    """

    def __init__(self, k: int = 3):
        super().__init__(max_cell_size=k)
        self.k = k

    def lift(
        self,
        chunk_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> TopologicalComplex:
        """Lift using only k-cliques."""
        num_chunks = chunk_features.shape[0]
        cells_by_dim = {}

        if edge_index is None or edge_index.numel() == 0:
            return TopologicalComplex(
                chunk_features=chunk_features,
                cells_by_dim=cells_by_dim,
                num_chunks=num_chunks,
            )

        # Convert to NetworkX
        G = nx.Graph()
        G.add_nodes_from(range(num_chunks))
        G.add_edges_from(edge_index.t().tolist())

        # Add 1-cells (edges)
        edge_cells = []
        seen_edges = set()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            edge_set = frozenset({src, dst})
            if edge_set not in seen_edges and src != dst:
                seen_edges.add(edge_set)
                edge_cells.append(
                    Cell(
                        chunk_indices={src, dst},
                        dimension=1,
                        cell_id=len(edge_cells),
                    )
                )
        if edge_cells:
            cells_by_dim[1] = edge_cells

        # Find all k-cliques
        k_cliques = []
        seen_cliques = set()

        for clique in nx.find_cliques(G):
            if len(clique) >= self.k:
                # Extract all k-subcliques
                for sub in combinations(clique, self.k):
                    sub_set = frozenset(sub)
                    if sub_set not in seen_cliques:
                        seen_cliques.add(sub_set)
                        k_cliques.append(set(sub))

        # Create cells for k-cliques
        dim = self.k - 1
        if k_cliques:
            cells_by_dim[dim] = [
                Cell(
                    chunk_indices=clique,
                    dimension=dim,
                    cell_id=i,
                )
                for i, clique in enumerate(k_cliques)
            ]

        return TopologicalComplex(
            chunk_features=chunk_features,
            cells_by_dim=cells_by_dim,
            num_chunks=num_chunks,
        )
