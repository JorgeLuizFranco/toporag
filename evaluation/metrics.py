"""
Evaluation metrics for TopoRAG.

Replicates metrics from GFM-RAG paper:
- Retrieval: Recall@k
- QA: Exact Match (EM), F1 score

References:
- GFM-RAG: Table 1 (Retrieval), Table 2 (QA)
- HotpotQA evaluation: https://hotpotqa.github.io/
"""

import re
import string
from collections import Counter
from typing import List, Set, Tuple


def normalize_answer(s: str) -> str:
    """
    Normalize answer string for evaluation.

    Applies:
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Fix whitespace

    Args:
        s: Answer string

    Returns:
        Normalized answer string
    """
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> int:
    """
    Compute exact match score.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        1 if normalized strings match, 0 otherwise
    """
    return 1 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    """
    Compute F1 score between prediction and ground truth.

    Token-level F1 score following HotpotQA evaluation.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        Tuple of (f1, precision, recall)
    """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    zero_metric = (0.0, 0.0, 0.0)

    # Handle yes/no/noanswer special cases
    special_answers = ["yes", "no", "noanswer"]
    if normalized_prediction in special_answers and normalized_prediction != normalized_ground_truth:
        return zero_metric
    if normalized_ground_truth in special_answers and normalized_prediction != normalized_ground_truth:
        return zero_metric

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return zero_metric

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return zero_metric

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1, precision, recall


def recall_at_k(
    retrieved: List[str],
    ground_truth: Set[str],
    k: int,
) -> float:
    """
    Compute Recall@k for retrieval.

    From GFM-RAG paper (Table 1):
    "Recall@k = |retrieved[:k] ∩ ground_truth| / |ground_truth|"

    Args:
        retrieved: List of retrieved document titles/IDs (ordered by score)
        ground_truth: Set of ground truth document titles/IDs
        k: Number of top documents to consider

    Returns:
        Recall@k score
    """
    if len(ground_truth) == 0:
        return 0.0

    retrieved_at_k = set(retrieved[:k])
    hits = len(retrieved_at_k & ground_truth)

    return hits / len(ground_truth)


def mean_reciprocal_rank(
    retrieved: List[str],
    ground_truth: Set[str],
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    Args:
        retrieved: List of retrieved document titles/IDs (ordered by score)
        ground_truth: Set of ground truth document titles/IDs

    Returns:
        Reciprocal rank (1/rank of first relevant doc, or 0 if none found)
    """
    for i, doc in enumerate(retrieved):
        if doc in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(
    retrieved: List[str],
    ground_truth: Set[str],
    k: int,
) -> float:
    """
    Compute Precision@k for retrieval.

    Args:
        retrieved: List of retrieved document titles/IDs
        ground_truth: Set of ground truth document titles/IDs
        k: Number of top documents to consider

    Returns:
        Precision@k score
    """
    if k == 0:
        return 0.0

    retrieved_at_k = set(retrieved[:k])
    hits = len(retrieved_at_k & ground_truth)

    return hits / k


def compute_retrieval_metrics(
    retrieved: List[str],
    ground_truth: Set[str],
    k_values: Tuple[int, ...] = (2, 5, 10),
) -> dict:
    """
    Compute all retrieval metrics.

    Args:
        retrieved: List of retrieved document titles/IDs
        ground_truth: Set of ground truth document titles/IDs
        k_values: Values of k for Recall@k

    Returns:
        Dict with metrics
    """
    metrics = {}

    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved, ground_truth, k)
        metrics[f"precision@{k}"] = precision_at_k(retrieved, ground_truth, k)

    metrics["mrr"] = mean_reciprocal_rank(retrieved, ground_truth)

    return metrics


def compute_qa_metrics(
    prediction: str,
    ground_truth: str,
) -> dict:
    """
    Compute all QA metrics.

    Args:
        prediction: Predicted answer
        ground_truth: Ground truth answer

    Returns:
        Dict with em, f1, precision, recall
    """
    em = exact_match_score(prediction, ground_truth)
    f1, precision, recall = f1_score(prediction, ground_truth)

    return {
        "em": float(em),
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }
