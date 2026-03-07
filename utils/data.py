"""
Data loading utilities for TopoRAG.

Supports the same benchmarks as GFM-RAG:
- HotpotQA
- MuSiQue
- 2WikiMultiHopQA
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class QASample:
    """A single QA sample."""
    question: str
    answer: str
    supporting_facts: List[str]  # Document titles that support the answer
    context: List[Tuple[str, List[str]]]  # List of (title, sentences)
    sample_id: str = ""
    question_type: str = ""


def load_hotpotqa(
    data_path: str,
    split: str = "dev",
    max_samples: Optional[int] = None,
) -> List[QASample]:
    """
    Load HotpotQA dataset.

    HotpotQA format:
    {
        "_id": "...",
        "question": "...",
        "answer": "...",
        "supporting_facts": [["title", sent_idx], ...],
        "context": [["title", ["sent1", "sent2", ...]], ...],
        "type": "bridge" or "comparison",
        "level": "easy", "medium", or "hard"
    }

    Args:
        data_path: Path to JSON file
        split: Data split (not used, for API consistency)
        max_samples: Maximum samples to load

    Returns:
        List of QASample objects
    """
    with open(data_path) as f:
        data = json.load(f)

    samples = []
    for item in data[:max_samples] if max_samples else data:
        # Extract supporting fact titles
        supporting_titles = list(set(sf[0] for sf in item.get("supporting_facts", [])))

        sample = QASample(
            question=item["question"],
            answer=item["answer"],
            supporting_facts=supporting_titles,
            context=[(ctx[0], ctx[1]) for ctx in item.get("context", [])],
            sample_id=item.get("_id", ""),
            question_type=item.get("type", ""),
        )
        samples.append(sample)

    return samples


def load_musique(
    data_path: str,
    split: str = "dev",
    max_samples: Optional[int] = None,
) -> List[QASample]:
    """
    Load MuSiQue dataset.

    MuSiQue format:
    {
        "id": "...",
        "question": "...",
        "answer": "...",
        "paragraphs": [
            {"idx": 0, "title": "...", "paragraph_text": "...", "is_supporting": true/false},
            ...
        ]
    }

    Args:
        data_path: Path to JSON file
        split: Data split
        max_samples: Maximum samples to load

    Returns:
        List of QASample objects
    """
    with open(data_path) as f:
        data = json.load(f)

    samples = []
    for item in data[:max_samples] if max_samples else data:
        paragraphs = item.get("paragraphs", [])

        # Extract supporting paragraph titles
        supporting_titles = [
            p.get("title", f"para_{p.get('idx', i)}")
            for i, p in enumerate(paragraphs)
            if p.get("is_supporting", False)
        ]

        # Build context
        context = [
            (p.get("title", f"para_{p.get('idx', i)}"), [p.get("paragraph_text", "")])
            for i, p in enumerate(paragraphs)
        ]

        sample = QASample(
            question=item["question"],
            answer=item.get("answer", ""),
            supporting_facts=supporting_titles,
            context=context,
            sample_id=item.get("id", ""),
        )
        samples.append(sample)

    return samples


def load_2wiki(
    data_path: str,
    split: str = "dev",
    max_samples: Optional[int] = None,
) -> List[QASample]:
    """
    Load 2WikiMultiHopQA dataset.

    2Wiki format:
    {
        "_id": "...",
        "question": "...",
        "answer": "...",
        "supporting_facts": [["title", sent_idx], ...],
        "context": [["title", ["sent1", "sent2", ...]], ...],
        "type": "...",
        "evidences": [...]
    }

    Args:
        data_path: Path to JSON file
        split: Data split
        max_samples: Maximum samples to load

    Returns:
        List of QASample objects
    """
    with open(data_path) as f:
        data = json.load(f)

    samples = []
    for item in data[:max_samples] if max_samples else data:
        # Extract supporting fact titles
        supporting_titles = list(set(sf[0] for sf in item.get("supporting_facts", [])))

        sample = QASample(
            question=item["question"],
            answer=item.get("answer", ""),
            supporting_facts=supporting_titles,
            context=[(ctx[0], ctx[1]) for ctx in item.get("context", [])],
            sample_id=item.get("_id", ""),
            question_type=item.get("type", ""),
        )
        samples.append(sample)

    return samples


def extract_chunks_from_documents(
    samples: List[QASample],
    chunk_method: str = "sentence",
) -> Tuple[List[str], Dict[int, List[int]]]:
    """
    Extract chunks from document contexts.

    Args:
        samples: List of QA samples
        chunk_method: How to chunk ("sentence", "paragraph")

    Returns:
        Tuple of:
        - List of chunk texts
        - Dict mapping sample_idx -> list of chunk indices
    """
    all_chunks = []
    sample_to_chunks = {}

    for sample_idx, sample in enumerate(samples):
        chunk_indices = []

        for title, sentences in sample.context:
            if chunk_method == "sentence":
                # Each sentence is a chunk
                for sent in sentences:
                    chunk_indices.append(len(all_chunks))
                    all_chunks.append(f"{title}: {sent}")
            elif chunk_method == "paragraph":
                # All sentences together as one chunk
                chunk_indices.append(len(all_chunks))
                all_chunks.append(f"{title}: {' '.join(sentences)}")

        sample_to_chunks[sample_idx] = chunk_indices

    return all_chunks, sample_to_chunks


def get_ground_truth_chunks(
    sample: QASample,
    all_chunks: List[str],
    sample_chunks: List[int],
) -> List[int]:
    """
    Get indices of ground truth chunks for a sample.

    Args:
        sample: QA sample
        all_chunks: List of all chunk texts
        sample_chunks: Chunk indices for this sample

    Returns:
        List of chunk indices that are supporting facts
    """
    gt_indices = []

    for idx in sample_chunks:
        chunk_text = all_chunks[idx]
        # Check if chunk is from a supporting document
        for title in sample.supporting_facts:
            if chunk_text.startswith(f"{title}:"):
                gt_indices.append(idx)
                break

    return gt_indices


def create_evaluation_data(
    samples: List[QASample],
    all_chunks: List[str],
    sample_to_chunks: Dict[int, List[int]],
) -> List[Dict[str, Any]]:
    """
    Create evaluation data in GFM-RAG format.

    Args:
        samples: QA samples
        all_chunks: All chunk texts
        sample_to_chunks: Mapping from sample to chunk indices

    Returns:
        List of dicts ready for evaluation
    """
    eval_data = []

    for i, sample in enumerate(samples):
        gt_chunks = get_ground_truth_chunks(
            sample, all_chunks, sample_to_chunks[i]
        )

        eval_data.append({
            "question": sample.question,
            "answer": sample.answer,
            "supporting_facts": sample.supporting_facts,
            "ground_truth_chunks": gt_chunks,
            "sample_id": sample.sample_id,
        })

    return eval_data


def load_gfm_dataset(
    data_path: str,
    dataset_name: str,
    max_samples: Optional[int] = None,
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """
    Load dataset in GFM-RAG standard format.

    Returns:
        corpus: Dict[title, text] - The complete document corpus
        test_data: List[Dict] - Test samples with 'supporting_facts' (titles)
    """
    data_path = Path(data_path)
    
    # Check for GFM-RAG 'raw' structure (dataset_corpus.json + test.json)
    # Expected: data_path/raw/dataset_corpus.json
    raw_path = data_path / "raw"
    if (raw_path / "dataset_corpus.json").exists():
        print(f"Loading GFM-RAG format from {raw_path}...")
        
        # Load Corpus
        with open(raw_path / "dataset_corpus.json") as f:
            corpus = json.load(f)
            
        # Load Test Data
        test_file = raw_path / "test.json"
        if not test_file.exists():
             # Fallback to train.json if test doesn't exist (e.g. dev set)
             test_file = raw_path / "train.json"
             
        with open(test_file) as f:
            test_data = json.load(f)
            
        if max_samples:
            test_data = test_data[:max_samples]
            
        return corpus, test_data

    # Fallback: Load from monolithic JSON (LPGNN format) and convert
    print(f"Loading monolithic format from {data_path} and converting...")
    
    # Handle different filenames for different datasets
    filename = f"{dataset_name}.json"
    if dataset_name == "2wiki":
        filename = "2wikimultihopqa.json"
        
    monolithic_path = data_path / filename
    if not monolithic_path.exists():
        # Try finding any .json file
        json_files = list(data_path.glob("*.json"))
        if json_files:
            monolithic_path = json_files[0]
        else:
            raise FileNotFoundError(f"No data found at {data_path}")

    with open(monolithic_path) as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    corpus = {}
    test_data = []

    for item in data:
        # Extract documents for corpus
        # Formats vary slightly between datasets
        paragraphs = item.get("paragraphs", [])
        if not paragraphs and "context" in item:
            # HotpotQA/2Wiki format: context is [[title, [sentences]], ...]
            paragraphs = []
            for ctx in item["context"]:
                paragraphs.append({
                    "title": ctx[0],
                    "paragraph_text": "".join(ctx[1])
                })
        
        supporting_titles = []
        
        # Build Corpus & Extract Ground Truth
        for p in paragraphs:
            title = p.get("title")
            text = p.get("paragraph_text", "")
            if not title: 
                continue
                
            # Add to corpus (deduplicated by title)
            if title not in corpus:
                corpus[title] = text
            
            # Check if supporting (MuSiQue style)
            if p.get("is_supporting", False):
                supporting_titles.append(title)
        
        # HotpotQA/2Wiki separate supporting_facts list: [[title, sent_id], ...]
        if "supporting_facts" in item:
            # Override/Extend with explicit list
            s_facts = item["supporting_facts"]
            # s_facts is list of [title, sent_idx]
            titles = set(f[0] for f in s_facts)
            supporting_titles = list(titles)

        test_data.append({
            "id": item.get("_id") or item.get("id"),
            "question": item["question"],
            "answer": item.get("answer", ""),
            "supporting_facts": supporting_titles
        })

    return corpus, test_data
