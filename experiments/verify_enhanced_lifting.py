#!/usr/bin/env python3
"""
Verify Enhanced Entity Lifting on MuSiQue 500.

Measures gold chunk pair connectivity and coverage with the new enhanced
EntityHypergraphLifting (resolve_aliases, normalize_demonyms, subdivide_large)
and compares against the old baseline numbers (21% pair connectivity, 91.6%
chunk coverage).
"""

import json
import sys
import time
from pathlib import Path
from itertools import combinations
from collections import Counter

import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer

# ── Setup paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from toporag.lifting.entity import EntityHypergraphLifting

DATA_PATH = PROJECT_ROOT / "LPGNN-retriever/datasets/musique/musique.json"
MAX_SAMPLES = 500


# ── Load data ────────────────────────────────────────────────────────────

def load_musique(max_samples: int = MAX_SAMPLES):
    """Load MuSiQue dataset, returning flat chunk list and per-question metadata."""
    print(f"Loading MuSiQue dataset from {DATA_PATH} ...")
    with open(DATA_PATH) as f:
        data = json.load(f)

    chunks = []
    samples = []

    for q_idx, item in enumerate(data[:max_samples]):
        paragraphs = item.get("paragraphs", [])
        local_indices = []

        for p in paragraphs:
            text = f"{p.get('title', '')}: {p.get('paragraph_text', '')}"
            global_idx = len(chunks)
            chunks.append(text)
            local_indices.append(global_idx)

        supporting_global = [
            local_indices[i]
            for i, p in enumerate(paragraphs)
            if p.get("is_supporting", False) and i < len(local_indices)
        ]

        samples.append({
            "question": item["question"],
            "answer": item.get("answer", ""),
            "all_chunks": local_indices,
            "supporting": supporting_global,
        })

    print(f"  {max_samples} questions, {len(chunks)} total chunks")
    return chunks, samples


# ── Metric computation ───────────────────────────────────────────────────

def compute_metrics(samples, cell_to_nodes):
    """
    Compute gold chunk pair connectivity and gold chunk coverage.

    Args:
        samples: list of dicts with 'supporting' key (list of global chunk indices)
        cell_to_nodes: dict mapping cell_idx -> list of chunk indices

    Returns:
        pair_connectivity: fraction of gold pairs sharing at least one cell
        chunk_coverage: fraction of gold chunks appearing in at least one cell
    """
    # Build reverse index: chunk_idx -> set of cell indices
    chunk_to_cells = {}
    for cell_idx, node_list in cell_to_nodes.items():
        for node_idx in node_list:
            if node_idx not in chunk_to_cells:
                chunk_to_cells[node_idx] = set()
            chunk_to_cells[node_idx].add(cell_idx)

    total_pairs = 0
    connected_pairs = 0
    total_gold_chunks = 0
    covered_gold_chunks = 0

    for sample in samples:
        gold = sample["supporting"]
        if len(gold) < 2:
            continue

        # Pair connectivity
        for c1, c2 in combinations(gold, 2):
            total_pairs += 1
            cells_1 = chunk_to_cells.get(c1, set())
            cells_2 = chunk_to_cells.get(c2, set())
            if cells_1 & cells_2:  # intersection
                connected_pairs += 1

        # Chunk coverage
        for c in gold:
            total_gold_chunks += 1
            if c in chunk_to_cells:
                covered_gold_chunks += 1

    pair_conn = 100.0 * connected_pairs / total_pairs if total_pairs > 0 else 0.0
    chunk_cov = 100.0 * covered_gold_chunks / total_gold_chunks if total_gold_chunks > 0 else 0.0

    return pair_conn, chunk_cov, total_pairs, connected_pairs, total_gold_chunks, covered_gold_chunks


def print_cell_size_distribution(cell_to_nodes):
    """Print histogram of cell sizes."""
    sizes = [len(nodes) for nodes in cell_to_nodes.values()]
    if not sizes:
        print("  No cells!")
        return

    counter = Counter(sizes)
    print(f"  Cell size distribution (total {len(sizes)} cells):")
    print(f"    min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/len(sizes):.1f}, median={sorted(sizes)[len(sizes)//2]}")
    print(f"    Size  Count")
    for size in sorted(counter.keys()):
        bar = "#" * min(counter[size] // 10, 60)
        print(f"    {size:>4d}  {counter[size]:>5d}  {bar}")


# ── Run OLD lifting (no enhancements) ───────────────────────────────────

def run_old_lifting(chunks, embeddings):
    """Run old-style lifting: no alias resolution, no demonyms, no subdivision."""
    print("\n" + "=" * 70)
    print("OLD LIFTING (max_cell_size=10, no resolution, no subdivision)")
    print("=" * 70)

    lifter = EntityHypergraphLifting(
        min_cell_size=2,
        max_cell_size=10,
        resolve_aliases=False,
        normalize_demonyms=False,
        subdivide_large=False,
    )

    data = Data(x=embeddings)
    t0 = time.time()
    topology = lifter.lift(data, chunks=chunks)
    elapsed = time.time() - t0
    print(f"  Lifting time: {elapsed:.1f}s")

    return topology


# ── Run NEW lifting (all enhancements) ──────────────────────────────────

def run_new_lifting(chunks, embeddings):
    """Run enhanced lifting: alias resolution + demonyms + subdivision."""
    print("\n" + "=" * 70)
    print("NEW LIFTING (resolve_aliases + normalize_demonyms + subdivide_large)")
    print("=" * 70)

    lifter = EntityHypergraphLifting(
        min_cell_size=2,
        max_cell_size=10,
        resolve_aliases=True,
        normalize_demonyms=True,
        subdivide_large=True,
        subdivision_target_size=5,
        subdivision_max_cells_per_entity=50,
    )

    data = Data(x=embeddings)
    t0 = time.time()
    topology = lifter.lift(data, chunks=chunks)
    elapsed = time.time() - t0
    print(f"  Lifting time: {elapsed:.1f}s")

    return topology


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    # 1. Load data
    chunks, samples = load_musique()

    # 2. Embed chunks
    print("\nEmbedding chunks with all-mpnet-base-v2 ...")
    t0 = time.time()
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_tensor=True)
    print(f"  Embedding shape: {embeddings.shape}  ({time.time()-t0:.1f}s)")

    # 3. Run OLD lifting
    old_topo = run_old_lifting(chunks, embeddings)
    old_pair_conn, old_chunk_cov, old_tp, old_cp, old_tg, old_cg = compute_metrics(
        samples, old_topo.cell_to_nodes
    )
    print(f"\n  Gold pair connectivity: {old_pair_conn:.1f}% ({old_cp}/{old_tp})")
    print(f"  Gold chunk coverage:   {old_chunk_cov:.1f}% ({old_cg}/{old_tg})")
    print_cell_size_distribution(old_topo.cell_to_nodes)

    # 4. Run NEW lifting
    new_topo = run_new_lifting(chunks, embeddings)
    new_pair_conn, new_chunk_cov, new_tp, new_cp, new_tg, new_cg = compute_metrics(
        samples, new_topo.cell_to_nodes
    )
    print(f"\n  Gold pair connectivity: {new_pair_conn:.1f}% ({new_cp}/{new_tp})")
    print(f"  Gold chunk coverage:   {new_chunk_cov:.1f}% ({new_cg}/{new_tg})")
    print_cell_size_distribution(new_topo.cell_to_nodes)

    # 5. Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"""
OLD LIFTING (max_cell_size=10, no resolution, no subdivision):
  Gold pair connectivity: {old_pair_conn:.1f}%  ({old_cp}/{old_tp} pairs)
  Gold chunk coverage:    {old_chunk_cov:.1f}%  ({old_cg}/{old_tg} chunks)
  Total cells:            {old_topo.num_edges:,}

NEW LIFTING (resolve_aliases + normalize_demonyms + subdivide_large):
  Gold pair connectivity: {new_pair_conn:.1f}%  ({new_cp}/{new_tp} pairs)
  Gold chunk coverage:    {new_chunk_cov:.1f}%  ({new_cg}/{new_tg} chunks)
  Total cells:            {new_topo.num_edges:,}

DELTA:
  Pair connectivity:  {old_pair_conn:.1f}% -> {new_pair_conn:.1f}%  ({new_pair_conn - old_pair_conn:+.1f}%)
  Chunk coverage:     {old_chunk_cov:.1f}% -> {new_chunk_cov:.1f}%  ({new_chunk_cov - old_chunk_cov:+.1f}%)
  Cell count:         {old_topo.num_edges:,} -> {new_topo.num_edges:,}  ({new_topo.num_edges - old_topo.num_edges:+,})

BASELINE REFERENCE (from MEMORY.md):
  Old pair connectivity: 21.0%
  Old chunk coverage:    91.6%
  Old total cells:       8,570
""")


if __name__ == "__main__":
    main()
