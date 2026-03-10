#!/usr/bin/env python3
"""
DIAGNOSTIC: Zero-shot cell scoring strategies.

Tests whether cell-level scoring can improve over chunk-level cosine,
WITHOUT any learning (zero-shot). This eliminates distribution shift.

Strategies:
  1. Cosine baseline (chunk-level only)
  2. Cell-cosine: cos(q, cell_emb) where cell_emb = mean(chunk_embs)
  3. Cell-max-cosine: max cos(q, chunk_i) for chunks in cell
  4. Cell-entity-match: query→entity overlap with cell's entity name
  5. Two-stage: cos(q, x_i) + boost * cell_score(q, cells_of_i)

Also measures: cell selection accuracy (what % of top-C cells contain gold chunks)
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
    topo_file = REPO_ROOT / "experiments/cache/topology/musique_500_entity.pt"

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
        supp = [local[i] for i, p in enumerate(paragraphs)
                if p.get("is_supporting", False) and i < len(local)]
        samples.append({"question": item["question"], "supporting": supp})

    N = len(chunks)
    cache = torch.load(topo_file, weights_only=False)
    cell_to_nodes = cache["cell_to_nodes"]
    x_chunks = cache["lifted"].x_0

    # Build maps
    chunk_to_cells = defaultdict(list)
    for cell_id, nodes in cell_to_nodes.items():
        for ni in nodes:
            chunk_to_cells[ni].append(cell_id)

    entity_cell_ids = sorted([ci for ci, nodes in cell_to_nodes.items() if len(nodes) >= 2])
    cell_id_to_pos = {ci: pos for pos, ci in enumerate(entity_cell_ids)}
    M = len(entity_cell_ids)

    # Setup
    config = TopoRAGConfig(lifting="entity", use_gps=False, use_tnn=False)
    toporag = TopoRAG(config)
    device = toporag.device
    embedder = toporag.embedder
    x_norm = F.normalize(x_chunks.to(device), dim=-1)

    # Pre-compute cell embeddings
    cell_embs = []
    for ci in entity_cell_ids:
        nodes = cell_to_nodes[ci]
        cell_embs.append(x_chunks[nodes].mean(dim=0))
    cell_embs = F.normalize(torch.stack(cell_embs).to(device), dim=-1)  # (M, D)

    print(f"Chunks: {N}, Entity cells: {M}, Questions: {len(samples)}")
    print("\nRunning diagnostics...\n")

    # --- Cell selection accuracy ---
    # For each question, what fraction of top-C cells contain gold chunks?
    cell_acc = {c: [] for c in [5, 10, 20, 50, 100]}
    gold_cell_counts = []

    # --- Retrieval results ---
    results = {}
    for method in ["cosine", "cell_mean_b02", "cell_mean_b05", "cell_mean_b10",
                    "cell_max_b02", "cell_max_b05", "cell_max_b10",
                    "oracle_b02", "oracle_b10"]:
        results[method] = {"r2": [], "r5": [], "r10": []}

    for si, sample in enumerate(samples):
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

        # --- Cosine baseline ---
        ranked = cos.topk(10).indices.tolist()
        results["cosine"]["r2"].append(len(gt_set & set(ranked[:2])) / len(gt_set))
        results["cosine"]["r5"].append(len(gt_set & set(ranked[:5])) / len(gt_set))
        results["cosine"]["r10"].append(len(gt_set & set(ranked[:10])) / len(gt_set))

        # --- Cell scoring: cos(q, cell_emb_mean) ---
        cell_cos = (q_norm @ cell_embs.T).squeeze(0)  # (M,)

        # --- Cell scoring: max cos(q, chunk_i) for chunks in cell ---
        cell_max = torch.full((M,), -1.0, device=device)
        for pos, ci in enumerate(entity_cell_ids):
            nodes = cell_to_nodes[ci]
            cell_max[pos] = cos[nodes].max()

        # --- Cell selection accuracy ---
        gold_cells = set()
        for gi in gt:
            for cid in chunk_to_cells.get(gi, []):
                pos = cell_id_to_pos.get(cid)
                if pos is not None:
                    gold_cells.add(pos)
        gold_cell_counts.append(len(gold_cells))

        for C in cell_acc:
            top_c = cell_cos.topk(min(C, M)).indices.tolist()
            hits = len(gold_cells & set(top_c))
            cell_acc[C].append(hits / max(len(gold_cells), 1))

        # --- Boosted retrieval with cell-mean scores ---
        for boost, suffix in [(0.2, "b02"), (0.5, "b05"), (1.0, "b10")]:
            # For each chunk, get max cell-mean-cosine across its cells
            chunk_boost = torch.zeros(N, device=device)
            for ni in range(N):
                for cid in chunk_to_cells.get(ni, []):
                    pos = cell_id_to_pos.get(cid)
                    if pos is not None:
                        val = cell_cos[pos].item()
                        if val > chunk_boost[ni].item():
                            chunk_boost[ni] = val

            # Normalize cell scores to [0,1] via min-max on positive scores
            cmin = chunk_boost[chunk_boost > 0].min() if (chunk_boost > 0).any() else 0
            cmax = chunk_boost.max()
            if cmax > cmin:
                chunk_boost = ((chunk_boost - cmin) / (cmax - cmin)).clamp(0, 1)

            final = cos + boost * chunk_boost
            ranked = final.topk(10).indices.tolist()
            results[f"cell_mean_{suffix}"]["r2"].append(len(gt_set & set(ranked[:2])) / len(gt_set))
            results[f"cell_mean_{suffix}"]["r5"].append(len(gt_set & set(ranked[:5])) / len(gt_set))
            results[f"cell_mean_{suffix}"]["r10"].append(len(gt_set & set(ranked[:10])) / len(gt_set))

        # --- Boosted retrieval with cell-max scores ---
        for boost, suffix in [(0.2, "b02"), (0.5, "b05"), (1.0, "b10")]:
            chunk_boost = torch.zeros(N, device=device)
            for ni in range(N):
                for cid in chunk_to_cells.get(ni, []):
                    pos = cell_id_to_pos.get(cid)
                    if pos is not None:
                        val = cell_max[pos].item()
                        if val > chunk_boost[ni].item():
                            chunk_boost[ni] = val

            cmin = chunk_boost[chunk_boost > 0].min() if (chunk_boost > 0).any() else 0
            cmax = chunk_boost.max()
            if cmax > cmin:
                chunk_boost = ((chunk_boost - cmin) / (cmax - cmin)).clamp(0, 1)

            final = cos + boost * chunk_boost
            ranked = final.topk(10).indices.tolist()
            results[f"cell_max_{suffix}"]["r2"].append(len(gt_set & set(ranked[:2])) / len(gt_set))
            results[f"cell_max_{suffix}"]["r5"].append(len(gt_set & set(ranked[:5])) / len(gt_set))
            results[f"cell_max_{suffix}"]["r10"].append(len(gt_set & set(ranked[:10])) / len(gt_set))

        # --- Oracle ---
        for boost, suffix in [(0.2, "b02"), (1.0, "b10")]:
            boosted = cos.clone()
            for gi in gt:
                for cid in chunk_to_cells.get(gi, []):
                    if cid in cell_to_nodes:
                        for ni in cell_to_nodes[cid]:
                            boosted[ni] += boost
            ranked = boosted.topk(10).indices.tolist()
            results[f"oracle_{suffix}"]["r2"].append(len(gt_set & set(ranked[:2])) / len(gt_set))
            results[f"oracle_{suffix}"]["r5"].append(len(gt_set & set(ranked[:5])) / len(gt_set))
            results[f"oracle_{suffix}"]["r10"].append(len(gt_set & set(ranked[:10])) / len(gt_set))

        if si % 100 == 0 and si > 0:
            print(f"  {si}/{len(samples)}...")

    # Print cell selection accuracy
    print("\n" + "=" * 65)
    print("CELL SELECTION ACCURACY (zero-shot, cos(q, cell_mean))")
    print("=" * 65)
    print(f"Gold cells per question: {sum(gold_cell_counts)/len(gold_cell_counts):.1f} avg "
          f"(max {max(gold_cell_counts)}, min {min(gold_cell_counts)})")
    print(f"{'Top-C':>8}  {'Recall of gold cells':>20}")
    print("-" * 35)
    for C in sorted(cell_acc.keys()):
        acc = sum(cell_acc[C]) / len(cell_acc[C]) * 100
        print(f"{C:>8}  {acc:>19.1f}%")

    # Print retrieval results
    print("\n" + "=" * 65)
    print("RETRIEVAL RESULTS")
    print("=" * 65)
    print(f"{'Method':<25} {'R@2':>8} {'R@5':>8} {'R@10':>8} {'dR@5':>8}")
    print("-" * 65)
    base_r5 = sum(results["cosine"]["r5"]) / len(results["cosine"]["r5"])
    for name in ["cosine",
                 "cell_mean_b02", "cell_mean_b05", "cell_mean_b10",
                 "cell_max_b02", "cell_max_b05", "cell_max_b10",
                 "oracle_b02", "oracle_b10"]:
        d = results[name]
        r2 = sum(d["r2"]) / len(d["r2"]) * 100
        r5 = sum(d["r5"]) / len(d["r5"]) * 100
        r10 = sum(d["r10"]) / len(d["r10"]) * 100
        delta = r5 - base_r5 * 100
        sign = "+" if delta >= 0 else ""
        label = {
            "cosine": "Cosine (baseline)",
            "cell_mean_b02": "Cell-mean boost=0.2",
            "cell_mean_b05": "Cell-mean boost=0.5",
            "cell_mean_b10": "Cell-mean boost=1.0",
            "cell_max_b02": "Cell-max boost=0.2",
            "cell_max_b05": "Cell-max boost=0.5",
            "cell_max_b10": "Cell-max boost=1.0",
            "oracle_b02": "ORACLE boost=0.2",
            "oracle_b10": "ORACLE boost=1.0",
        }[name]
        print(f"{label:<25} {r2:>7.1f}% {r5:>7.1f}% {r10:>7.1f}% {sign}{delta:>7.1f}%")
    print("-" * 65)
    print(f"{'GFM-RAG (target)':<25} {'49.1%':>8} {'58.2%':>8}")
    print("=" * 65)


if __name__ == "__main__":
    main()
