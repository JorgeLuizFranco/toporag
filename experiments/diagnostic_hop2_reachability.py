#!/usr/bin/env python3
"""
DIAGNOSTIC: Are missing gold cells reachable via the cell graph?

Key question: For questions where cell-max fails to find all gold cells,
are the MISSED gold cells within 1-2 hops of the FOUND gold cells in
the cell graph?

If YES → GNN on cell graph should help (information can flow)
If NO  → need fundamentally different approach

Also measures:
- What fraction of gold cells are hop-1 vs hop-2 vs unreachable
- Per-hop-distance recall breakdown
- Whether hop-2 gold cells share entities with hop-1 gold cells
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, deque

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

    # Build cell adjacency (shared chunks)
    adj = defaultdict(set)
    for chunk_idx, cell_ids in chunk_to_cells.items():
        positions = [cell_id_to_pos[c] for c in cell_ids if c in cell_id_to_pos]
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                adj[positions[i]].add(positions[j])
                adj[positions[j]].add(positions[i])

    print(f"Cell graph: {M} nodes, {sum(len(v) for v in adj.values())//2} edges")

    # Embed queries
    config = TopoRAGConfig(lifting="entity", use_gps=False, use_tnn=False)
    toporag = TopoRAG(config)
    device = toporag.device
    embedder = toporag.embedder
    x_norm = F.normalize(x_chunks.to(device), dim=-1)

    # --- BFS distance function ---
    def bfs_distances(start_set, max_dist=5):
        """BFS from start_set, return dict pos -> distance."""
        dist = {}
        queue = deque()
        for s in start_set:
            dist[s] = 0
            queue.append(s)
        while queue:
            u = queue.popleft()
            if dist[u] >= max_dist:
                continue
            for v in adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    # --- Analysis ---
    # For each question:
    #   1. Get gold cells
    #   2. Get top-C cells by cell-max (found cells)
    #   3. Check: are missed gold cells reachable from found cells?

    TOP_C = 100  # same as best cell-max config

    stats = {
        "total_gold": 0,
        "found_by_cellmax": 0,
        "missed": 0,
        "missed_at_dist": defaultdict(int),  # dist -> count
        "missed_unreachable": 0,
        "questions_with_missed": 0,
        "questions_total": 0,
        # Per-question: fraction of missed that are reachable within k hops
        "reachable_within_1": [],
        "reachable_within_2": [],
        "reachable_within_3": [],
        # Gold-to-gold distances
        "gold_pair_distances": defaultdict(int),
    }

    # Also: analyze whether found gold cells reach missed gold cells
    hop2_analysis = {
        "missed_cells_with_found_gold_neighbor": 0,
        "missed_cells_total": 0,
    }

    for si, sample in enumerate(samples):
        gt = sample["supporting"]
        if not gt:
            continue
        gt_set = set(gt)
        stats["questions_total"] += 1

        # Get gold cell positions
        gold_cells = set()
        for gi in gt:
            for cid in chunk_to_cells.get(gi, []):
                pos = cell_id_to_pos.get(cid)
                if pos is not None:
                    gold_cells.add(pos)

        if not gold_cells:
            continue

        # Compute cell-max scores
        with torch.no_grad():
            q = embedder.encode([sample["question"]], is_query=True, show_progress=False)
            if not isinstance(q, torch.Tensor):
                q = torch.tensor(q)
            q = q.to(device)
            q_norm = F.normalize(q, dim=-1)

        cos = (q_norm @ x_norm.T).squeeze(0)  # (N,)

        cell_max_scores = torch.full((M,), -1.0, device=device)
        for pos, ci in enumerate(entity_cell_ids):
            nodes = cell_to_nodes[ci]
            cell_max_scores[pos] = cos[nodes].max()

        top_c_cells = set(cell_max_scores.topk(min(TOP_C, M)).indices.tolist())

        found_gold = gold_cells & top_c_cells
        missed_gold = gold_cells - top_c_cells

        stats["total_gold"] += len(gold_cells)
        stats["found_by_cellmax"] += len(found_gold)
        stats["missed"] += len(missed_gold)

        if missed_gold:
            stats["questions_with_missed"] += 1

            # BFS from found cells (not just found gold — ALL top-C cells)
            distances_from_found = bfs_distances(top_c_cells, max_dist=5)
            # Also BFS from found GOLD cells specifically
            distances_from_found_gold = bfs_distances(found_gold, max_dist=5)

            n_missed = len(missed_gold)
            reachable_1 = sum(1 for m in missed_gold if distances_from_found.get(m, 999) <= 1)
            reachable_2 = sum(1 for m in missed_gold if distances_from_found.get(m, 999) <= 2)
            reachable_3 = sum(1 for m in missed_gold if distances_from_found.get(m, 999) <= 3)

            stats["reachable_within_1"].append(reachable_1 / n_missed)
            stats["reachable_within_2"].append(reachable_2 / n_missed)
            stats["reachable_within_3"].append(reachable_3 / n_missed)

            for m in missed_gold:
                d = distances_from_found.get(m, -1)
                if d == -1:
                    stats["missed_unreachable"] += 1
                else:
                    stats["missed_at_dist"][d] = stats["missed_at_dist"].get(d, 0) + 1

            # Check from found GOLD specifically
            for m in missed_gold:
                hop2_analysis["missed_cells_total"] += 1
                d_from_gold = distances_from_found_gold.get(m, 999)
                if d_from_gold <= 1:
                    hop2_analysis["missed_cells_with_found_gold_neighbor"] += 1

        # Gold-to-gold distances
        gold_list = list(gold_cells)
        for i in range(len(gold_list)):
            dists_i = bfs_distances({gold_list[i]}, max_dist=5)
            for j in range(i + 1, len(gold_list)):
                d = dists_i.get(gold_list[j], -1)
                key = str(d) if d >= 0 else "unreachable"
                stats["gold_pair_distances"][key] += 1

        if si % 100 == 0 and si > 0:
            print(f"  {si}/{len(samples)}...")

    # --- Print results ---
    print("\n" + "=" * 70)
    print("HOP-2 REACHABILITY ANALYSIS")
    print("=" * 70)

    print(f"\nQuestions analyzed: {stats['questions_total']}")
    print(f"Questions with missed gold cells: {stats['questions_with_missed']} "
          f"({stats['questions_with_missed']/stats['questions_total']*100:.1f}%)")
    print(f"\nTotal gold cells: {stats['total_gold']}")
    print(f"Found by cell-max (top-{TOP_C}): {stats['found_by_cellmax']} "
          f"({stats['found_by_cellmax']/stats['total_gold']*100:.1f}%)")
    print(f"Missed: {stats['missed']} "
          f"({stats['missed']/stats['total_gold']*100:.1f}%)")

    print(f"\n--- Distance of missed gold cells from top-{TOP_C} cells ---")
    for d in sorted(stats["missed_at_dist"].keys()):
        cnt = stats["missed_at_dist"][d]
        print(f"  Distance {d}: {cnt} cells ({cnt/stats['missed']*100:.1f}%)")
    print(f"  Unreachable: {stats['missed_unreachable']} "
          f"({stats['missed_unreachable']/stats['missed']*100:.1f}%)")

    if stats["reachable_within_1"]:
        print(f"\n--- Per-question reachability of missed gold cells ---")
        for k, vals in [(1, stats["reachable_within_1"]),
                        (2, stats["reachable_within_2"]),
                        (3, stats["reachable_within_3"])]:
            avg = sum(vals) / len(vals) * 100
            print(f"  Within {k} hop(s): {avg:.1f}% of missed gold cells reachable")

    print(f"\n--- From FOUND GOLD cells specifically ---")
    print(f"  Missed cells adjacent to found gold: "
          f"{hop2_analysis['missed_cells_with_found_gold_neighbor']}/{hop2_analysis['missed_cells_total']} "
          f"({hop2_analysis['missed_cells_with_found_gold_neighbor']/max(hop2_analysis['missed_cells_total'],1)*100:.1f}%)")

    print(f"\n--- Gold-to-gold cell distances ---")
    total_pairs = sum(stats["gold_pair_distances"].values())
    for key in sorted(stats["gold_pair_distances"].keys(), key=lambda x: (x == "unreachable", x)):
        cnt = stats["gold_pair_distances"][key]
        print(f"  Distance {key}: {cnt} pairs ({cnt/total_pairs*100:.1f}%)")

    # --- Key insight ---
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if stats["reachable_within_2"]:
        r2 = sum(stats["reachable_within_2"]) / len(stats["reachable_within_2"]) * 100
        if r2 > 70:
            print(f"✓ {r2:.0f}% of missed gold cells are within 2 hops of top-{TOP_C}.")
            print("  → A GNN on the cell graph CAN reach them. Worth pursuing.")
        elif r2 > 40:
            print(f"~ {r2:.0f}% of missed gold cells are within 2 hops of top-{TOP_C}.")
            print("  → Partial signal. GNN may help but won't close the full gap.")
        else:
            print(f"✗ Only {r2:.0f}% of missed gold cells are within 2 hops.")
            print("  → Cell graph structure insufficient. Need different approach.")


if __name__ == "__main__":
    main()
