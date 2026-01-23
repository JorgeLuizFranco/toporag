"""
Evaluators for TopoRAG.

Replicates the evaluation setup from GFM-RAG paper:
- Table 1: Retrieval performance (R@2, R@5)
- Table 2: QA performance (EM, F1)
- Table 3: Efficiency (query time, R@5)

Datasets:
- HotpotQA
- MuSiQue
- 2WikiMultiHopQA
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from statistics import mean
from typing import List, Dict, Set, Optional, Tuple

from .metrics import (
    compute_retrieval_metrics,
    compute_qa_metrics,
    recall_at_k,
    exact_match_score,
    f1_score,
)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metrics: Dict[str, float]
    num_samples: int
    dataset_name: str
    model_name: str = "TopoRAG"

    def __str__(self) -> str:
        lines = [f"=== {self.model_name} on {self.dataset_name} ==="]
        lines.append(f"Samples: {self.num_samples}")
        for k, v in self.metrics.items():
            lines.append(f"{k}: {v:.4f}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "num_samples": self.num_samples,
            "metrics": self.metrics,
        }


class BaseEvaluator(ABC):
    """Base evaluator class."""

    def __init__(self, prediction_file: Optional[str] = None):
        self.data = []
        if prediction_file:
            self.load_predictions(prediction_file)

    def load_predictions(self, prediction_file: str):
        """Load predictions from JSONL file."""
        with open(prediction_file) as f:
            self.data = [json.loads(line) for line in f]

    def set_predictions(self, predictions: List[dict]):
        """Set predictions directly."""
        self.data = predictions

    @abstractmethod
    def evaluate(self) -> dict:
        pass


class RetrievalEvaluator(BaseEvaluator):
    """
    Evaluator for retrieval performance.

    Computes Recall@k following GFM-RAG Table 1.

    Expected prediction format:
    {
        "question": "...",
        "supporting_facts": ["title1", "title2"],  # ground truth docs
        "retrieved_docs": [
            {"title": "...", "score": 0.95},
            {"title": "...", "score": 0.85},
            ...
        ]
    }
    """

    def __init__(
        self,
        prediction_file: Optional[str] = None,
        k_values: Tuple[int, ...] = (2, 5, 10),
    ):
        super().__init__(prediction_file)
        self.k_values = k_values

    def evaluate(self) -> dict:
        """
        Evaluate retrieval performance.

        Returns:
            Dict with recall@k for each k
        """
        metrics: Dict[str, List[float]] = {f"recall@{k}": [] for k in self.k_values}

        for pred in self.data:
            gold_docs = set(pred["supporting_facts"])

            # Sort retrieved docs by score (descending)
            sorted_docs = sorted(
                pred["retrieved_docs"],
                key=lambda x: x["score"],
                reverse=True,
            )
            retrieved_titles = [doc["title"] for doc in sorted_docs]

            # Compute recall at each k
            for k in self.k_values:
                recall = recall_at_k(retrieved_titles, gold_docs, k)
                metrics[f"recall@{k}"].append(recall)

        # Average over all samples
        return {k: mean(v) for k, v in metrics.items()}


class QAEvaluator(BaseEvaluator):
    """
    Evaluator for question answering performance.

    Computes EM and F1 following GFM-RAG Table 2.

    Expected prediction format:
    {
        "question": "...",
        "answer": "ground truth answer",
        "response": "model generated response"
    }
    """

    def __init__(
        self,
        prediction_file: Optional[str] = None,
        answer_prefix: str = "Answer:",
    ):
        super().__init__(prediction_file)
        self.answer_prefix = answer_prefix

    def _extract_answer(self, response: str) -> str:
        """Extract answer from model response."""
        if self.answer_prefix in response:
            return response.split(self.answer_prefix)[1].strip()
        return response.strip()

    def evaluate(self) -> dict:
        """
        Evaluate QA performance.

        Returns:
            Dict with em, f1, precision, recall
        """
        metrics = {"em": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

        for pred in self.data:
            predicted_answer = self._extract_answer(pred["response"])
            ground_truth = pred["answer"]

            em = exact_match_score(predicted_answer, ground_truth)
            f1, precision, recall = f1_score(predicted_answer, ground_truth)

            metrics["em"] += em
            metrics["f1"] += f1
            metrics["precision"] += precision
            metrics["recall"] += recall

        n = len(self.data)
        return {k: v / n for k, v in metrics.items()}


class TopoRAGEvaluator:
    """
    Complete evaluator for TopoRAG.

    Combines retrieval and QA evaluation to produce
    results comparable to GFM-RAG Tables 1, 2, 3.

    Args:
        dataset_name: Name of the dataset (hotpotqa, musique, 2wiki)
    """

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.retrieval_evaluator = RetrievalEvaluator()
        self.qa_evaluator = QAEvaluator()

    def evaluate_retrieval(
        self,
        predictions: List[dict],
        k_values: Tuple[int, ...] = (2, 5),
    ) -> EvaluationResult:
        """
        Evaluate retrieval performance.

        Args:
            predictions: List of prediction dicts
            k_values: Values of k for Recall@k

        Returns:
            EvaluationResult with retrieval metrics
        """
        self.retrieval_evaluator.set_predictions(predictions)
        self.retrieval_evaluator.k_values = k_values
        metrics = self.retrieval_evaluator.evaluate()

        return EvaluationResult(
            metrics=metrics,
            num_samples=len(predictions),
            dataset_name=self.dataset_name,
        )

    def evaluate_qa(
        self,
        predictions: List[dict],
    ) -> EvaluationResult:
        """
        Evaluate QA performance.

        Args:
            predictions: List of prediction dicts

        Returns:
            EvaluationResult with QA metrics
        """
        self.qa_evaluator.set_predictions(predictions)
        metrics = self.qa_evaluator.evaluate()

        return EvaluationResult(
            metrics=metrics,
            num_samples=len(predictions),
            dataset_name=self.dataset_name,
        )

    def evaluate_efficiency(
        self,
        retriever,
        queries: List[torch.Tensor],
        chunk_features: torch.Tensor,
        cells: List,
    ) -> dict:
        """
        Evaluate retrieval efficiency (Table 3 from GFM-RAG).

        Args:
            retriever: TopoRAG retriever model
            queries: List of query embeddings
            chunk_features: Chunk embeddings
            cells: List of cells

        Returns:
            Dict with query_time_ms and throughput
        """
        import torch

        # Warmup
        for _ in range(10):
            _ = retriever.retrieve(queries[0], chunk_features, cells, top_k=5)

        # Measure
        times = []
        for query in queries:
            start = time.perf_counter()
            _ = retriever.retrieve(query, chunk_features, cells, top_k=5)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        return {
            "query_time_ms": mean(times),
            "throughput_qps": 1000 / mean(times),
            "num_queries": len(queries),
        }

    def full_evaluation(
        self,
        retrieval_predictions: List[dict],
        qa_predictions: List[dict],
    ) -> Dict[str, EvaluationResult]:
        """
        Run full evaluation (retrieval + QA).

        Args:
            retrieval_predictions: Predictions for retrieval eval
            qa_predictions: Predictions for QA eval

        Returns:
            Dict with 'retrieval' and 'qa' EvaluationResults
        """
        return {
            "retrieval": self.evaluate_retrieval(retrieval_predictions),
            "qa": self.evaluate_qa(qa_predictions),
        }


def save_predictions(predictions: List[dict], output_file: str):
    """Save predictions to JSONL file."""
    with open(output_file, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")


def load_predictions(input_file: str) -> List[dict]:
    """Load predictions from JSONL file."""
    with open(input_file) as f:
        return [json.loads(line) for line in f]


# Convenience function for quick evaluation
def evaluate_toporag(
    dataset_name: str,
    retrieval_file: Optional[str] = None,
    qa_file: Optional[str] = None,
) -> dict:
    """
    Quick evaluation function.

    Args:
        dataset_name: Name of the dataset
        retrieval_file: Path to retrieval predictions JSONL
        qa_file: Path to QA predictions JSONL

    Returns:
        Dict with all metrics
    """
    evaluator = TopoRAGEvaluator(dataset_name)
    results = {}

    if retrieval_file:
        preds = load_predictions(retrieval_file)
        ret_result = evaluator.evaluate_retrieval(preds)
        results["retrieval"] = ret_result.metrics
        print(ret_result)

    if qa_file:
        preds = load_predictions(qa_file)
        qa_result = evaluator.evaluate_qa(preds)
        results["qa"] = qa_result.metrics
        print(qa_result)

    return results
