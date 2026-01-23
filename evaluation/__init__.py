"""
Evaluation module for TopoRAG.

Replicates the evaluation setup from GFM-RAG:
- Retrieval metrics: Recall@k (k=2, 5)
- QA metrics: Exact Match (EM), F1 score

Datasets: HotpotQA, MuSiQue, 2WikiMultiHopQA
"""

from .metrics import (
    normalize_answer,
    exact_match_score,
    f1_score,
    recall_at_k,
)
from .evaluator import (
    RetrievalEvaluator,
    QAEvaluator,
    TopoRAGEvaluator,
)

__all__ = [
    "normalize_answer",
    "exact_match_score",
    "f1_score",
    "recall_at_k",
    "RetrievalEvaluator",
    "QAEvaluator",
    "TopoRAGEvaluator",
]
