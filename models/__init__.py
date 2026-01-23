"""
Models for TopoRAG.

This module contains:
- TNN: Topological Neural Networks with message passing (CWN, SCN, Hypergraph)
- LP-TNN: Link Prediction Topological Neural Network for query-cell retrieval
- Cell encoders: Various ways to compute cell embeddings
- Link predictors: Score query-cell pairs
"""

from .link_predictor import QueryCellLinkPredictor, NCNLinkPredictor
from .cell_encoder import CellEncoder, DeepSetCellEncoder, AttentionCellEncoder
from .lp_tnn import LPTNN
from .tnn import TNN, CWNLayer, SCNLayer, HypergraphLayer, TNNOutput

__all__ = [
    # TNN with message passing
    "TNN",
    "CWNLayer",
    "SCNLayer",
    "HypergraphLayer",
    "TNNOutput",
    # Link prediction
    "QueryCellLinkPredictor",
    "NCNLinkPredictor",
    "CellEncoder",
    "DeepSetCellEncoder",
    "AttentionCellEncoder",
    "LPTNN",
]
