#!/usr/bin/env python3
"""
DIAGNOSTIC: Can entity cells improve retrieval AT ALL?

Tests zero-shot heuristics (no training) to determine if entity structure helps.
If zero-shot expansion doesn't help, no amount of learning will fix it.

Heuristics tested:
  1. Cosine baseline (R@K pure)
  2. Cell expansion: top-K cosine → find entity cells → pull in cell-mates → re-rank by cosine
  3. Evidence-weighted expansion: only trust cells with 2+ top-K hits
  4. Oracle cell expansion: use gold labels to identify correct cells (upper bound)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from toporag import TopoRAG, TopoRAGConfig

    REPO_ROOT = PROJECT_ROOT / "toporag"
    data_path = PROJECT_ROOT / "LPGNN-retriever/datasets/musique/musique.json"

    # Load data
    with open(data_path) as f:
        data = json.load(f)[:500]

    chunks, samples = [], []
    for q_idx, item in enumerate(data):
        paragraphs = item.get("paragraphs", [])
        local_indices = []
        for p in paragraphs:
            text = f"{p.get('title', '')}: {p.get('paragraph_text', '')}"
            global_idx = len(chunks)
            chunks.append(text)
            local_indices.append(global_idx)
        supporting_global = [
            local_indices[i] for i, p in enumerate(paragraphs)
            if p.get("is_supporting", False) and i < len(local_indices)
        ]
        samples.append({"question": item["question"], "supporting": supporting_global})

    N = len(chunks)
    print(f"Chunks: {N}, Questions: {len(samples)}")

    # Load cached topology
    topo_file = REPO_ROOT / "experiments/cache/topology/musique_500_entity.pt"
    cache = torch.load(topo_file, weights_only=False)
    cell_to_nodes = cache["cell_to_nodes"]
    x_chunks = cache["lifted"].x_0  # (N, 768)

    print(f"Entity cells: {len(cell_to_nodes)}")

    # Build reverse map: chunk → cells
    chunk_to_cells = defaultdict(list)
    for cell_id, node_list in cell_to_nodes.items():
        for ni in node_list:
            chunk_to_cells[ni].append(cell_id)

    # Embed queries
    config = TopoRAGConfig(lifting="entity", use_gps=False, use_tnn=False)
    toporag = TopoRAG(config)
    embedder = toporag.embedder
    device = toporag.device

    x_norm = F.normalize(x_chunks.to(device), dim=-1)

    print("\nRunning diagnostics...\n")

    results = {
        "cosine": {"r2": [], "r5": [], "r10": [], "r20": []},
        "expand_all": {"r2": [], "r5": [], "r10": [], "r20": []},
        "expand_2plus": {"r2": [], "r5": [], "r10": [], "r20": []},
        "expand_3plus": {"r2": [], "r5": [], "r10": [], "r20": []},
        "oracle_expand": {"r2": [], "r5": [], "r10": [], "r20": []},
    }

    for si, sample in enumerate(samples):
        gt = sample["supporting"]
        if not gt:
            continue
        gt_set = set(gt)

        # Encode query
        with torch.no_grad():
            q = embedder.encode([sample["question"]], is_query=True, show_progress=False)
            if not isinstance(q, torch.Tensor):
                q = torch.tensor(q)
            q = q.to(device)
            q_norm = F.normalize(q, dim=-1)

        # Cosine scores for all chunks
        cos = (q_norm @ x_norm.T).squeeze(0)  # (N,)

        # 1. Pure cosine baseline
        ranked = cos.topk(N).indices.tolist()
        for k_name, k_val in [("r2", 2), ("r5", 5), ("r10", 10), ("r20", 20)]:
            hits = len(gt_set & set(ranked[:k_val]))
            results["cosine"][k_name].append(hits / len(gt_set))

        # Get cosine top-20 for expansion
        top20 = cos.topk(20).indices.tolist()
        top20_set = set(top20)

        # Count cell evidence from top-20
        cell_evidence = Counter()
        for ci in top20:
            for cell_id in chunk_to_cells.get(ci, []):
                cell_evidence[cell_id] += 1

        # 2. Expand ALL activated cells (any cell with 1+ top-20 hit)
        expansion_all = set()
        for cell_id in cell_evidence:
            expansion_all.update(cell_to_nodes[cell_id])
        candidates = list(top20_set | expansion_all)
        cand_scores = cos[candidates]
        reranked = [candidates[i] for i in cand_scores.argsort(descending=True).tolist()]
        for k_name, k_val in [("r2", 2), ("r5", 5), ("r10", 10), ("r20", 20)]:
            hits = len(gt_set & set(reranked[:k_val]))
            results["expand_all"][k_name].append(hits / len(gt_set))

        # 3. Expand cells with 2+ top-20 hits
        expansion_2 = set()
        for cell_id, count in cell_evidence.items():
            if count >= 2:
                expansion_2.update(cell_to_nodes[cell_id])
        candidates = list(top20_set | expansion_2)
        cand_scores = cos[candidates]
        reranked = [candidates[i] for i in cand_scores.argsort(descending=True).tolist()]
        for k_name, k_val in [("r2", 2), ("r5", 5), ("r10", 10), ("r20", 20)]:
            hits = len(gt_set & set(reranked[:k_val]))
            results["expand_2plus"][k_name].append(hits / len(gt_set))

        # 4. Expand cells with 3+ top-20 hits
        expansion_3 = set()
        for cell_id, count in cell_evidence.items():
            if count >= 3:
                expansion_3.update(cell_to_nodes[cell_id])
        candidates = list(top20_set | expansion_3)
        cand_scores = cos[candidates]
        reranked = [candidates[i] for i in cand_scores.argsort(descending=True).tolist()]
        for k_name, k_val in [("r2", 2), ("r5", 5), ("r10", 10), ("r20", 20)]:
            hits = len(gt_set & set(reranked[:k_val]))
            results["expand_3plus"][k_name].append(hits / len(gt_set))

        # 5. Oracle: expand ONLY cells that contain gold chunks
        gold_cells = set()
        for gi in gt:
            for cell_id in chunk_to_cells.get(gi, []):
                gold_cells.add(cell_id)
        oracle_expansion = set()
        for cell_id in gold_cells:
            oracle_expansion.update(cell_to_nodes[cell_id])
        candidates = list(top20_set | oracle_expansion)
        cand_scores = cos[candidates]
        reranked = [candidates[i] for i in cand_scores.argsort(descending=True).tolist()]
        for k_name, k_val in [("r2", 2), ("r5", 5), ("r10", 10), ("r20", 20)]:
            hits = len(gt_set & set(reranked[:k_val]))
            results["oracle_expand"][k_name].append(hits / len(gt_set))

        if si % 100 == 0 and si > 0:
            print(f"  {si}/{len(samples)}...")

    # Print results
    print("\n" + "=" * 75)
    print("DIAGNOSTIC RESULTS — Can entity cells help retrieval?")
    print("=" * 75)
    print(f"{'Method':<25} {'R@2':>8} {'R@5':>8} {'R@10':>8} {'R@20':>8}")
    print("-" * 75)
    for name, data in results.items():
        r2 = sum(data["r2"]) / len(data["r2"]) * 100
        r5 = sum(data["r5"]) / len(data["r5"]) * 100
        r10 = sum(data["r10"]) / len(data["r10"]) * 100
        r20 = sum(data["r20"]) / len(data["r20"]) * 100
        label = {
            "cosine": "Cosine (baseline)",
            "expand_all": "Expand ALL cells",
            "expand_2plus": "Expand 2+ evidence",
            "expand_3plus": "Expand 3+ evidence",
            "oracle_expand": "Oracle expansion",
        }[name]
        print(f"{label:<25} {r2:>7.1f}% {r5:>7.1f}% {r10:>7.1f}% {r20:>7.1f}%")
    print("-" * 75)
    print(f"{'GFM-RAG (target)':<25} {'49.1%':>8} {'58.2%':>8}")
    print("=" * 75)

    # Additional analysis: how many candidates does each method produce?
    print(f"\nCandidate pool sizes (median):")
    for label, exp_set_getter in [
        ("Expand ALL", lambda: expansion_all),
        ("Expand 2+", lambda: expansion_2),
        ("Expand 3+", lambda: expansion_3),
    ]:
        # Re-compute for last sample (approximate)
        print(f"  {label}: ~{len(exp_set_getter())} chunks")


if __name__ == "__main__":
    main()
