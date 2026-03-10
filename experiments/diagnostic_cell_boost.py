#!/usr/bin/env python3
"""
DIAGNOSTIC v2: Can cell-based SCORE BOOSTING help?

The v1 diagnostic showed cell expansion + cosine re-ranking doesn't help.
This tests: what if we BOOST scores of chunks sharing cells with top-K?

score(i) = cos(q, x_i) + boost * max_shared_cells(i, top_K)

This simulates what a learned cell scorer would do.
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

    with open(data_path) as f:
        data = json.load(f)[:500]

    chunks, samples = [], []
    for item in data:
        paragraphs = item.get("paragraphs", [])
        local = []
        for p in paragraphs:
            text = f"{p.get('title', '')}: {p.get('paragraph_text', '')}"
            gi = len(chunks)
            chunks.append(text)
            local.append(gi)
        supp = [local[i] for i, p in enumerate(paragraphs) if p.get("is_supporting", False) and i < len(local)]
        samples.append({"question": item["question"], "supporting": supp})

    N = len(chunks)

    topo_file = REPO_ROOT / "experiments/cache/topology/musique_500_entity.pt"
    cache = torch.load(topo_file, weights_only=False)
    cell_to_nodes = cache["cell_to_nodes"]
    x_chunks = cache["lifted"].x_0

    chunk_to_cells = defaultdict(list)
    for cell_id, nodes in cell_to_nodes.items():
        for ni in nodes:
            chunk_to_cells[ni].append(cell_id)

    config = TopoRAGConfig(lifting="entity", use_gps=False, use_tnn=False)
    toporag = TopoRAG(config)
    device = toporag.device
    embedder = toporag.embedder
    x_norm = F.normalize(x_chunks.to(device), dim=-1)

    print("Testing cell-based score boosting...\n")

    # Test various boost values and K values
    boost_values = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]
    seed_k_values = [5, 10, 20]
    min_evidence = [1, 2]

    results = {}

    for seed_k in seed_k_values:
        for min_ev in min_evidence:
            for boost in boost_values:
                key = f"K={seed_k},ev>={min_ev},b={boost}"
                r2_list, r5_list = [], []

                for sample in samples:
                    gt = sample["supporting"]
                    if not gt:
                        continue
                    gt_set = set(gt)

                    with torch.no_grad():
                        q = embedder.encode([sample["question"]], is_query=True, show_progress=False)
                        if not isinstance(q, torch.Tensor):
                            q = torch.tensor(q)
                        q = q.to(device)
                        q_norm = F.normalize(q, dim=-1)

                    cos = (q_norm @ x_norm.T).squeeze(0)  # (N,)
                    top_k = cos.topk(seed_k).indices.tolist()
                    top_k_set = set(top_k)

                    # Count cell evidence
                    cell_ev = Counter()
                    for ci in top_k:
                        for cid in chunk_to_cells.get(ci, []):
                            cell_ev[cid] += 1

                    # Build boost vector
                    boosted_scores = cos.clone()
                    for cid, count in cell_ev.items():
                        if count >= min_ev:
                            for ni in cell_to_nodes[cid]:
                                if ni not in top_k_set:
                                    boosted_scores[ni] += boost

                    retrieved = boosted_scores.topk(5).indices.tolist()
                    r2_list.append(len(gt_set & set(retrieved[:2])) / len(gt_set))
                    r5_list.append(len(gt_set & set(retrieved[:5])) / len(gt_set))

                results[key] = (sum(r2_list)/len(r2_list), sum(r5_list)/len(r5_list))

    # Also test oracle boost: boost ONLY chunks in gold cells
    for boost in [0.05, 0.1, 0.2, 0.5, 1.0]:
        key = f"ORACLE,b={boost}"
        r2_list, r5_list = [], []
        for sample in samples:
            gt = sample["supporting"]
            if not gt:
                continue
            gt_set = set(gt)
            with torch.no_grad():
                q = embedder.encode([sample["question"]], is_query=True, show_progress=False)
                if not isinstance(q, torch.Tensor):
                    q = torch.tensor(q)
                q = q.to(device)
                q_norm = F.normalize(q, dim=-1)
            cos = (q_norm @ x_norm.T).squeeze(0)

            # Oracle: boost chunks in cells containing gold chunks
            gold_cells = set()
            for gi in gt:
                for cid in chunk_to_cells.get(gi, []):
                    gold_cells.add(cid)
            boosted = cos.clone()
            for cid in gold_cells:
                for ni in cell_to_nodes[cid]:
                    boosted[ni] += boost

            retrieved = boosted.topk(5).indices.tolist()
            r2_list.append(len(gt_set & set(retrieved[:2])) / len(gt_set))
            r5_list.append(len(gt_set & set(retrieved[:5])) / len(gt_set))
        results[key] = (sum(r2_list)/len(r2_list), sum(r5_list)/len(r5_list))

    # Print
    print("=" * 65)
    print("CELL BOOST DIAGNOSTIC")
    print("=" * 65)
    print(f"{'Method':<30} {'R@2':>8} {'R@5':>8} {'Delta R@5':>10}")
    print("-" * 65)
    baseline_r5 = results["K=5,ev>=1,b=0.0"][1]
    for key, (r2, r5) in sorted(results.items()):
        delta = r5 - baseline_r5
        sign = "+" if delta >= 0 else ""
        print(f"{key:<30} {r2*100:>7.1f}% {r5*100:>7.1f}% {sign}{delta*100:>8.1f}%")
    print("-" * 65)
    print(f"{'GFM-RAG (target)':<30} {'49.1%':>8} {'58.2%':>8}")
    print("=" * 65)


if __name__ == "__main__":
    main()
