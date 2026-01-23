"""
TopoRAG: Preferential Higher-Order Networks for Reliable Retrieval

A novel RAG framework that:
- I1: Lifts chunks to higher-order topological structures (cells/hyperedges)
- I2: Generates speculative queries at the cell level as supervision signal
- I3: Casts retrieval as query-cell link prediction using Topological Neural Networks

Main classes:
- TopoRAG: Main pipeline class
- TopoRAGConfig: Configuration dataclass
"""

__version__ = "0.1.0"

from .toporag import TopoRAG, TopoRAGConfig

__all__ = [
    "TopoRAG",
    "TopoRAGConfig",
]
