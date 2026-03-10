#!/usr/bin/env python3
"""
Remap query cache from old topology cell indices to new topology cell indices.

For each old cell (identified by its chunk set), finds the new cell with highest
Jaccard overlap. If overlap >= 0.5, maps old_idx -> new_idx; otherwise drops.

Per-chunk (zero-cell) queries are remapped using the new zero_cell_offset.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import torch

CACHE_DIR = Path(__file__).resolve().parent / "cache"
TOPO_DIR = CACHE_DIR / "topology"

OLD_TOPO = TOPO_DIR / "musique_500_entity_old.pt"
NEW_TOPO = TOPO_DIR / "musique_500_entity.pt"
QUERY_CACHE = CACHE_DIR / "musique_500_queries.json"
OUTPUT_PATH = CACHE_DIR / "musique_500_queries_remapped.json"


def load_topology(path: Path):
    """Load topology and return cell_to_nodes dict."""
    data = torch.load(path, weights_only=False)
    return data["cell_to_nodes"]


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def build_inverted_index(cell_to_nodes: dict) -> dict:
    """Build node -> set of cell indices for fast lookup."""
    node_to_cells = {}
    for cell_idx, nodes in cell_to_nodes.items():
        for n in nodes:
            node_to_cells.setdefault(n, set()).add(cell_idx)
    return node_to_cells


def main():
    # ---- Load data ----
    print("Loading old topology...")
    old_c2n = load_topology(OLD_TOPO)
    print(f"  {len(old_c2n)} cells, idx range [{min(old_c2n)}, {max(old_c2n)}]")

    print("Loading new topology...")
    new_c2n = load_topology(NEW_TOPO)
    print(f"  {len(new_c2n)} cells, idx range [{min(new_c2n)}, {max(new_c2n)}]")

    print("Loading query cache...")
    with open(QUERY_CACHE) as f:
        queries = json.load(f)
    print(f"  {len(queries)} cell entries, {sum(len(v) for v in queries.values())} total queries")

    # ---- Compute offsets ----
    old_zero_cell_offset = max(old_c2n.keys()) + 1
    new_zero_cell_offset = max(new_c2n.keys()) + 1
    print(f"\nOld zero-cell offset: {old_zero_cell_offset}")
    print(f"New zero-cell offset: {new_zero_cell_offset}")

    # ---- Build inverted index on new topology for fast matching ----
    print("\nBuilding inverted index on new topology...")
    new_node_to_cells = build_inverted_index(new_c2n)

    # Pre-convert new cells to sets for fast Jaccard
    new_c2n_sets = {idx: set(nodes) for idx, nodes in new_c2n.items()}

    # ---- Build old->new cell mapping ----
    print("Computing cell mapping (old -> new) via Jaccard overlap...")
    old_to_new = {}
    overlap_scores = []
    no_match_count = 0

    for old_idx, old_nodes in old_c2n.items():
        old_set = set(old_nodes)

        # Find candidate new cells that share at least one node
        candidate_cells = set()
        for n in old_nodes:
            if n in new_node_to_cells:
                candidate_cells.update(new_node_to_cells[n])

        if not candidate_cells:
            no_match_count += 1
            continue

        # Find best match by Jaccard
        best_new_idx = -1
        best_score = 0.0
        for new_idx in candidate_cells:
            score = jaccard(old_set, new_c2n_sets[new_idx])
            if score > best_score:
                best_score = score
                best_new_idx = new_idx

        if best_score >= 0.5:
            old_to_new[old_idx] = best_new_idx
            overlap_scores.append(best_score)
        else:
            no_match_count += 1
            overlap_scores.append(best_score)

    print(f"  Mapped: {len(old_to_new)} / {len(old_c2n)} cells")
    print(f"  No match (Jaccard < 0.5): {no_match_count}")

    # ---- Remap queries ----
    print("\nRemapping queries...")
    remapped = {}
    stats = Counter()

    for str_key, query_list in queries.items():
        old_cell_idx = int(str_key)

        if old_cell_idx >= old_zero_cell_offset:
            # Per-chunk (zero-cell) query: remap via chunk_idx
            chunk_idx = old_cell_idx - old_zero_cell_offset
            new_cell_idx = new_zero_cell_offset + chunk_idx
            remapped[str(new_cell_idx)] = query_list
            stats["zero_cell_remapped"] += len(query_list)
        elif old_cell_idx in old_to_new:
            # Multi-hop cell query: remap via Jaccard mapping
            new_cell_idx = old_to_new[old_cell_idx]
            new_key = str(new_cell_idx)
            if new_key in remapped:
                remapped[new_key].extend(query_list)
            else:
                remapped[new_key] = list(query_list)
            stats["multi_hop_remapped"] += len(query_list)
        else:
            # No matching cell in new topology — drop
            stats["dropped"] += len(query_list)

    # ---- Save ----
    print(f"\nSaving remapped queries to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(remapped, f, indent=2)

    # ---- Print stats ----
    total_in = sum(len(v) for v in queries.values())
    total_out = sum(len(v) for v in remapped.values())

    print(f"\n{'='*60}")
    print(f"REMAP STATS")
    print(f"{'='*60}")
    print(f"Input:  {len(queries)} cell entries, {total_in} queries")
    print(f"Output: {len(remapped)} cell entries, {total_out} queries")
    print(f"")
    print(f"Multi-hop remapped: {stats['multi_hop_remapped']} queries")
    print(f"Zero-cell remapped: {stats['zero_cell_remapped']} queries")
    print(f"Dropped (no match): {stats['dropped']} queries")
    print(f"")

    # Overlap distribution for matched cells
    if overlap_scores:
        bins = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
        print("Jaccard overlap distribution (all old cells):")
        for lo, hi in bins:
            count = sum(1 for s in overlap_scores if lo <= s < hi)
            label = f"  [{lo:.1f}, {hi:.1f})" if hi < 1.01 else f"  [{lo:.1f}, 1.0]"
            bar = "#" * (count // 20)
            print(f"  {label}: {count:5d}  {bar}")
        print(f"  mean={sum(overlap_scores)/len(overlap_scores):.3f}, "
              f"median={sorted(overlap_scores)[len(overlap_scores)//2]:.3f}")

    print(f"\nDone. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
