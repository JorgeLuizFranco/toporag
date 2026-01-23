"""
Cycle Lifting: Creates 2-cells from cycle basis of the chunk graph.

This lifting identifies basic cycles in the chunk graph and creates
2-cells (faces) from them. A cycle represents a group of chunks that
form a closed semantic loop.

From difflifting.tex:
"Cycle lifting is a static (non-learnable) approach that constructs 2-cells
by identifying basic cycles (elements of a cycle basis) or chordless cycles
in input graphs. Specifically, the vertices involved in a basic cycle are
grouped to form a 2-cell in the resulting cell complex."
"""

from typing import Optional, List, Set, Tuple
import torch
import numpy as np
import networkx as nx

from .base import BaseLiftingTransform, TopologicalComplex, Cell


class CycleLifting(BaseLiftingTransform):
    """
    Cycle Lifting: Creates 2-cells from cycles in the chunk graph.

    This lifting produces a cell complex where:
    - 0-cells: individual chunks
    - 1-cells: edges from the chunk graph
    - 2-cells: cycles (closed paths) in the graph

    The 2-cells represent groups of chunks that are interconnected
    in a cyclic pattern, suggesting a coherent semantic unit.

    Args:
        max_cycle_length: Maximum length of cycles to consider
        use_chordless: If True, use chordless cycles; else use cycle basis
        min_cycle_length: Minimum length of cycles (default: 3)
    """

    def __init__(
        self,
        max_cycle_length: int = 6,
        use_chordless: bool = False,
        min_cycle_length: int = 3,
    ):
        super().__init__(max_cell_size=max_cycle_length)
        self.max_cycle_length = max_cycle_length
        self.use_chordless = use_chordless
        self.min_cycle_length = min_cycle_length

    def _edge_index_to_networkx(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> nx.Graph:
        """Convert PyTorch Geometric edge_index to NetworkX graph."""
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        edge_list = edge_index.t().tolist()
        G.add_edges_from(edge_list)

        return G

    def _find_cycle_basis(self, G: nx.Graph) -> List[List[int]]:
        """
        Find a cycle basis of the graph.

        A cycle basis is a minimal set of cycles such that any cycle
        can be expressed as a symmetric difference of cycles in the basis.
        """
        try:
            cycles = nx.cycle_basis(G)
            # Filter by length
            cycles = [
                c
                for c in cycles
                if self.min_cycle_length <= len(c) <= self.max_cycle_length
            ]
            return cycles
        except:
            return []

    def _find_chordless_cycles(self, G: nx.Graph) -> List[List[int]]:
        """
        Find all chordless cycles (induced cycles) in the graph.

        A chordless cycle is a cycle with no chord (an edge connecting
        two non-adjacent vertices of the cycle).
        """
        chordless_cycles = []

        # Use DFS to find all simple cycles up to max_cycle_length
        for start_node in G.nodes():
            # BFS/DFS to find cycles starting from this node
            stack = [(start_node, [start_node])]

            while stack:
                node, path = stack.pop()

                if len(path) > self.max_cycle_length:
                    continue

                for neighbor in G.neighbors(node):
                    if neighbor == start_node and len(path) >= self.min_cycle_length:
                        # Found a cycle
                        cycle = path.copy()
                        # Check if chordless
                        if self._is_chordless(G, cycle):
                            # Normalize cycle to avoid duplicates
                            normalized = self._normalize_cycle(cycle)
                            if normalized not in chordless_cycles:
                                chordless_cycles.append(normalized)

                    elif neighbor not in path:
                        stack.append((neighbor, path + [neighbor]))

        return chordless_cycles

    def _is_chordless(self, G: nx.Graph, cycle: List[int]) -> bool:
        """Check if a cycle has no chords."""
        n = len(cycle)
        for i in range(n):
            for j in range(i + 2, n):
                # Skip adjacent vertices in cycle
                if j == (i + n - 1) % n + i:
                    continue
                # Check for chord
                if G.has_edge(cycle[i], cycle[j]):
                    return False
        return True

    def _normalize_cycle(self, cycle: List[int]) -> List[int]:
        """Normalize cycle representation to avoid duplicates."""
        # Find the minimum element and its index
        min_val = min(cycle)
        min_idx = cycle.index(min_val)

        # Rotate so minimum is first
        rotated = cycle[min_idx:] + cycle[:min_idx]

        # Choose the lexicographically smaller direction
        reversed_cycle = [rotated[0]] + rotated[1:][::-1]
        if reversed_cycle < rotated:
            return reversed_cycle
        return rotated

    def lift(
        self,
        chunk_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> TopologicalComplex:
        """
        Lift chunk graph to cell complex using cycles.

        Args:
            chunk_features: (num_chunks, feature_dim) tensor
            edge_index: (2, num_edges) tensor of edge indices
            edge_attr: Optional edge weights (not used for cycle detection)

        Returns:
            TopologicalComplex with 0-cells, 1-cells, and 2-cells (cycles)
        """
        num_chunks = chunk_features.shape[0]
        cells_by_dim = {}

        # Create 1-cells from edges
        if edge_index is not None and edge_index.numel() > 0:
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
            cells_by_dim[1] = edge_cells

            # Convert to NetworkX graph for cycle finding
            G = self._edge_index_to_networkx(edge_index, num_chunks)

            # Find cycles
            if self.use_chordless:
                cycles = self._find_chordless_cycles(G)
            else:
                cycles = self._find_cycle_basis(G)

            # Create 2-cells from cycles
            cycle_cells = []
            for i, cycle in enumerate(cycles):
                cell = Cell(
                    chunk_indices=set(cycle),
                    dimension=2,
                    cell_id=i,
                )
                cycle_cells.append(cell)

            if cycle_cells:
                cells_by_dim[2] = cycle_cells

        return TopologicalComplex(
            chunk_features=chunk_features,
            cells_by_dim=cells_by_dim,
            num_chunks=num_chunks,
        )


class HierarchicalCycleLifting(BaseLiftingTransform):
    """
    Hierarchical Cycle Lifting: Creates cells at multiple dimensions.

    This extends basic cycle lifting to create a proper cell complex
    with boundary relations between cells.

    For each cycle, we include:
    - All vertices (0-cells)
    - All edges (1-cells)
    - The cycle itself (2-cell)
    """

    def __init__(
        self,
        max_cycle_length: int = 6,
        max_dimension: int = 2,
    ):
        super().__init__(max_cell_size=max_cycle_length)
        self.max_cycle_length = max_cycle_length
        self.max_dimension = max_dimension

    def lift(
        self,
        chunk_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> TopologicalComplex:
        """Lift with hierarchical structure."""
        num_chunks = chunk_features.shape[0]

        # Use basic cycle lifting
        basic_lifter = CycleLifting(
            max_cycle_length=self.max_cycle_length,
            use_chordless=False,
        )
        complex_ = basic_lifter.lift(chunk_features, edge_index, edge_attr)

        # Ensure all boundary cells are included
        # For each 2-cell (cycle), ensure all edges are in 1-cells
        if 2 in complex_.cells_by_dim and 1 in complex_.cells_by_dim:
            existing_edges = {
                frozenset(c.chunk_indices) for c in complex_.cells_by_dim[1]
            }

            new_edges = []
            for cycle_cell in complex_.cells_by_dim[2]:
                cycle_list = sorted(list(cycle_cell.chunk_indices))
                n = len(cycle_list)
                for i in range(n):
                    edge = frozenset({cycle_list[i], cycle_list[(i + 1) % n]})
                    if edge not in existing_edges:
                        existing_edges.add(edge)
                        new_edges.append(
                            Cell(
                                chunk_indices=set(edge),
                                dimension=1,
                                cell_id=len(complex_.cells_by_dim[1]) + len(new_edges),
                            )
                        )

            complex_.cells_by_dim[1].extend(new_edges)

        return complex_
