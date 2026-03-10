"""
Topological Lifting Module for TopoRAG.

Provides static lifting transformations from graphs to higher-order structures:
- CycleLifting: Graph → Cell Complex (via cycle detection)
- CliqueLifting: Graph → Simplicial Complex (via clique enumeration)
- KNNLifting: Graph → Hypergraph (via k-NN neighborhoods)
"""

from .base import BaseLiftingTransform, LiftedTopology, Cell
from .cycle import CycleLifting
from .clique import CliqueLifting
from .knn import KNNHypergraphLifting
from .entity import EntityHypergraphLifting

KNNLifting = KNNHypergraphLifting
EntityLifting = EntityHypergraphLifting

__all__ = [
    "BaseLiftingTransform",
    "LiftedTopology",
    "Cell",
    "CycleLifting",
    "CliqueLifting",
    "KNNHypergraphLifting",
    "KNNLifting",
    "EntityHypergraphLifting",
    "EntityLifting",
]
