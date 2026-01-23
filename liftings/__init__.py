"""
Static lifting procedures for TopoRAG.

Liftings transform a chunk graph into higher-order topological structures.
These are STATIC liftings (not learnable like DiffLift) because the LLM
generates ground truth queries based on the fixed cell structure.

Supported liftings:
- k-NN lifting: Creates hyperedges from k nearest neighbors in embedding space
- Cycle lifting: Creates 2-cells from cycle basis of the chunk graph
- Clique lifting: Creates simplicial complex from cliques
"""

from .base import BaseLiftingTransform
from .knn_lifting import KNNLifting
from .cycle_lifting import CycleLifting
from .clique_lifting import CliqueLifting

__all__ = [
    "BaseLiftingTransform",
    "KNNLifting",
    "CycleLifting",
    "CliqueLifting",
]
