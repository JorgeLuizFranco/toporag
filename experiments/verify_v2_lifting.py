#!/usr/bin/env python3
"""Verify enhanced entity lifting v2: max_cell_size=50 + ego-cell subdivision."""

import json
import sys
import torch
from pathlib import Path
from collections import defaultdict
from itertools import combinations

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    # Load dataset
    data_path = PROJECT_ROOT / "LPGNN-retriever" / "datasets" / "musique" / "musique.json"
    with open(data_path) as f:
        data = json.load(f)[:500]

    chunks = []
    gold_pairs = []
    for q_idx, item in enumerate(data):
        paragraphs = item.get("paragraphs", [])
        local_indices = []
        supporting_global = []
        for p in paragraphs:
            text = f"{p.get('title', '')}: {p.get('paragraph_text', '')}"
            global_idx = len(chunks)
            chunks.append(text)
            local_indices.append(global_idx)
            if p.get("is_supporting", False):
                supporting_global.append(global_idx)
        # All pairs of supporting chunks
        for a, b in combinations(supporting_global, 2):
            gold_pairs.append((a, b))

    print(f"Chunks: {len(chunks)}, Gold pairs: {len(gold_pairs)}")

    # Embed
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_tensor=True)

    from torch_geometric.data import Data
    graph_data = Data(x=embeddings.cpu())

    from toporag.lifting.entity import EntityHypergraphLifting

    # Test with new defaults (max_cell_size=50, resolve, demonym, ego-cell subdivision)
    lifter = EntityHypergraphLifting(
        max_cell_size=50,
        resolve_aliases=True,
        normalize_demonyms=True,
        subdivide_large=True,
        subdivision_target_size=8,
        subdivision_max_cells_per_entity=30,
    )
    lifted = lifter.lift(graph_data, chunks=chunks)

    cell_to_nodes = lifted.cell_to_nodes
    print(f"\nCells: {lifted.num_edges}")

    # Build chunk → cells map
    chunk_to_cells = defaultdict(set)
    for cell_idx, node_list in cell_to_nodes.items():
        for ni in node_list:
            chunk_to_cells[ni].add(cell_idx)

    # Measure gold pair connectivity
    connected = 0
    gold_chunks_in_cells = set()
    all_gold_chunks = set()
    for a, b in gold_pairs:
        all_gold_chunks.update([a, b])
        if chunk_to_cells[a] & chunk_to_cells[b]:
            connected += 1

    for gc in all_gold_chunks:
        if chunk_to_cells[gc]:
            gold_chunks_in_cells.add(gc)

    print(f"\n{'='*60}")
    print(f"COMPARISON")
    print(f"{'='*60}")
    print(f"{'Metric':<35} {'OLD':>10} {'NEW':>10} {'Delta':>10}")
    print(f"{'-'*65}")
    print(f"{'max_cell_size':<35} {'10':>10} {'50':>10}")
    print(f"{'Gold pair connectivity':<35} {'21.0%':>10} {f'{100*connected/len(gold_pairs):.1f}%':>10} {f'+{100*connected/len(gold_pairs)-21.0:.1f}%':>10}")
    print(f"{'Gold chunk coverage':<35} {'91.6%':>10} {f'{100*len(gold_chunks_in_cells)/len(all_gold_chunks):.1f}%':>10}")
    print(f"{'Total cells':<35} {'8,570':>10} {f'{lifted.num_edges:,}':>10}")

    # Cell size distribution
    sizes = [len(v) for v in cell_to_nodes.values()]
    print(f"\nCell size distribution:")
    from collections import Counter
    size_counts = Counter()
    for s in sizes:
        if s <= 10:
            size_counts[f"{s}"] += 1
        elif s <= 20:
            size_counts["11-20"] += 1
        elif s <= 50:
            size_counts["21-50"] += 1
        else:
            size_counts[">50"] += 1

    for k in sorted(size_counts.keys(), key=lambda x: int(x.split("-")[0]) if x[0].isdigit() else 999):
        print(f"  Size {k:>5}: {size_counts[k]:>5} cells")

if __name__ == "__main__":
    main()
