#!/usr/bin/env python3
"""
TopoRAG Training Script — Modular, Subset-friendly, fully offline.

Usage examples:
  # Quick validation on 50 samples — local Qwen2.5-3B on GPU (default)
  python experiments/train_toporag.py --max_samples 50 --epochs 50

  # Higher quality queries: Qwen2.5-7B on CPU (uses ~14 GB RAM, slow but one-time)
  python experiments/train_toporag.py --max_samples 50 --llm_model Qwen/Qwen2.5-7B-Instruct --llm_device cpu

  # No LLM — baseline cosine only (verify plumbing before generating queries)
  python experiments/train_toporag.py --max_samples 50 --llm none

  # Re-use cached queries from a previous run (no LLM call at all)
  python experiments/train_toporag.py --query_cache experiments/cache/musique_50_queries.json

  # Scale to full dataset once validated
  python experiments/train_toporag.py --max_samples 1000 --query_cache experiments/cache/musique_1000_queries.json

Architecture: LPTNN — TNN backbone + DeepSet cell encoder + MLP link predictor.
Training: BCE with (same-dim / diff-dim / corrupted) negative sampling.
Query generation: one-time, cached to SQLite + JSON. Never re-runs if cache exists.
"""

import os
import sys
import json
import math
import random
import argparse
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# REPO_ROOT = toporag/ (the package itself)
# PROJECT_ROOT = parent of toporag/ (where datasets live)
REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_musique(data_path: Path, max_samples: int) -> Tuple[List[str], List[int], List[dict]]:
    """
    Load MuSiQue and return:
      chunks       : flat list of paragraph texts ("Title: body")
      chunk_to_doc : chunk_idx -> question_idx mapping
      samples      : list of {question, answer, chunks, supporting_global_indices}
    """
    with open(data_path) as f:
        data = json.load(f)

    chunks, chunk_to_doc, samples = [], [], []

    for q_idx, item in enumerate(data[:max_samples]):
        paragraphs = item.get("paragraphs", [])
        local_indices = []

        for p in paragraphs:
            text = f"{p.get('title', '')}: {p.get('paragraph_text', '')}"
            global_idx = len(chunks)
            chunks.append(text)
            chunk_to_doc.append(q_idx)
            local_indices.append(global_idx)

        # Gold supporting paragraph indices in the global chunk list
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

    return chunks, chunk_to_doc, samples


# ---------------------------------------------------------------------------
# Query generation / caching
# ---------------------------------------------------------------------------

MULTIHOP_PROMPT = """\
You are given {n} text passages that are related. \
Write ONE question that can ONLY be answered by combining information from MULTIPLE passages — \
it must be impossible to answer from any single passage alone.

Rules:
- Use exact names and facts from the passages
- The question must require cross-passage reasoning
- Output the question only, no explanation

Passages:
{context}

Question:"""

SINGLECHUNK_PROMPT = """\
Given the following text passage, write ONE factual question that is directly answered by it.
Output only the question, nothing else.

Passage:
{chunk}

Question:"""


def generate_queries(
    chunks: List[str],
    cell_to_nodes: Dict[int, List[int]],
    llm_provider: str,
    cache_path: Optional[Path],
    queries_per_cell: int = 1,
    also_per_chunk: bool = True,
    llm_kwargs: Optional[dict] = None,
) -> Dict[int, List[str]]:
    """
    Generate queries and return {cell_idx: [query_text, ...]}.

    Generates:
      - 1 multi-hop question per cell  (TopoRAG novel signal)
      - 1 simple question per chunk    (LP-RAG style fallback, if also_per_chunk=True)
    Both are mapped into the cell_idx → [queries] dict.
    Single-chunk queries use a synthetic cell_idx offset by len(cell_to_nodes).

    Saves/loads from cache_path (JSON) so you never re-call the API.
    """
    # Try loading from cache first
    if cache_path and cache_path.exists():
        print(f"  Loading cached queries from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
        # Normalize: some caches store list-of-dicts {"query_text": "..."} instead of list-of-str
        def _extract(v):
            if isinstance(v, list):
                out = []
                for item in v:
                    if isinstance(item, dict):
                        out.append(item.get("query_text") or item.get("text") or str(item))
                    else:
                        out.append(str(item))
                return out
            return v
        return {int(k): _extract(v) for k, v in cached.items()}

    if llm_provider == "none":
        print("  --llm none: skipping query generation (baseline only)")
        return {}

    from toporag.llms import get_llm
    llm_kwargs = llm_kwargs or {}
    print(f"  Using {llm_provider} LLM for query generation...")
    llm = get_llm(llm_provider, cache_dir=str(PROJECT_ROOT / ".cache"), **llm_kwargs)

    # After generation the LLM will be freed — training uses only embedder + LPTNN

    result: Dict[int, List[str]] = {}

    # --- Multi-hop queries per cell ---
    cells = [(idx, nodes) for idx, nodes in cell_to_nodes.items() if len(nodes) >= 2]
    print(f"  Generating multi-hop queries for {len(cells)} cells...")
    for i, (cell_idx, node_indices) in enumerate(cells):
        cell_chunks = [chunks[j] for j in node_indices[:5] if j < len(chunks)]
        context = "\n\n".join(f"[Passage {k+1}]: {c[:400]}" for k, c in enumerate(cell_chunks))
        prompt = MULTIHOP_PROMPT.format(n=len(cell_chunks), context=context)
        try:
            response = llm.generate(prompt).strip()
            q = response.split("\n")[0].strip()
            if "?" in q:
                result.setdefault(cell_idx, []).append(q)
                if i < 3:
                    print(f"    [cell {cell_idx}] {q}")
        except Exception as e:
            print(f"    Error at cell {cell_idx}: {e}")

        if i % 50 == 49:
            print(f"    Progress: {i+1}/{len(cells)} cells")

    # --- Single-chunk queries (LP-RAG style fallback) ---
    if also_per_chunk:
        # Map each chunk into a synthetic cell (chunk alone is a 0-cell)
        # We reuse cell indices starting from max_cell_idx + 1
        offset = max(cell_to_nodes.keys(), default=-1) + 1
        single_cells = {(offset + ci): [ci] for ci in range(min(len(chunks), 500))}
        print(f"  Generating per-chunk queries for {len(single_cells)} chunks...")
        for i, (fake_cell_idx, (chunk_idx,)) in enumerate(list(single_cells.items())[:300]):
            chunk_text = chunks[chunk_idx][:500]
            prompt = SINGLECHUNK_PROMPT.format(chunk=chunk_text)
            try:
                response = llm.generate(prompt).strip()
                q = response.split("\n")[0].strip()
                if "?" in q:
                    result.setdefault(fake_cell_idx, []).append(q)
                    # Register this fake cell so training knows it maps to chunk_idx
                    cell_to_nodes[fake_cell_idx] = [chunk_idx]
            except Exception as e:
                print(f"    Error at chunk {chunk_idx}: {e}")

            if i % 100 == 99:
                print(f"    Progress: {i+1} chunks")

    # Free LLM from GPU memory before training starts
    from toporag.llms import CachedLLM, LocalLLM
    inner = llm.llm if isinstance(llm, CachedLLM) else llm
    if isinstance(inner, LocalLLM):
        inner.free()

    print(f"  Total cells with queries: {len(result)}")
    total_q = sum(len(v) for v in result.values())
    print(f"  Total queries: {total_q}")

    # Save cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({str(k): v for k, v in result.items()}, f, indent=2)
        print(f"  Queries cached to {cache_path}")

    return result


def build_gold_queries(
    samples: List[dict],
    cell_to_nodes: Dict[int, List[int]],
) -> Dict[int, List[str]]:
    """Build training pairs from gold labels.

    For each MuSiQue question with supporting chunk ids, find all cells
    that contain at least one gold supporting chunk.  Map question → cell.
    This gives direct supervised training signal aligned with evaluation.
    """
    # Reverse map: chunk_idx -> list of cell indices containing it
    chunk_to_cells: Dict[int, List[int]] = {}
    for cell_idx, nodes in cell_to_nodes.items():
        for node in nodes:
            chunk_to_cells.setdefault(node, []).append(cell_idx)

    gold_queries: Dict[int, List[str]] = {}
    covered = 0
    for sample in samples:
        q = sample["question"]
        for chunk_idx in sample["supporting"]:
            for cell_idx in chunk_to_cells.get(chunk_idx, []):
                gold_queries.setdefault(cell_idx, []).append(q)
                covered += 1

    n_cells = len(gold_queries)
    print(f"  Gold training: {covered} (question,cell) pairs covering {n_cells} cells")
    return gold_queries


# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------

def sample_negatives(
    pos_cell_idx: int,
    cell_to_nodes: Dict[int, List[int]],
    num_same: int = 3,
    num_diff: int = 2,
    num_corrupted: int = 2,
) -> Tuple[List[int], List[List[int]]]:
    """Sample negative cells: same-size, diff-size, and corrupted."""
    cell_list = list(cell_to_nodes.keys())
    pos_nodes = cell_to_nodes[pos_cell_idx]
    pos_size = len(pos_nodes)
    all_node_ids = list({n for nodes in cell_to_nodes.values() for n in nodes})

    same = [c for c in cell_list if c != pos_cell_idx and len(cell_to_nodes[c]) == pos_size]
    diff = [c for c in cell_list if c != pos_cell_idx and len(cell_to_nodes[c]) != pos_size]

    negs = []
    negs += random.sample(same, min(num_same, len(same)))
    negs += random.sample(diff, min(num_diff, len(diff)))

    # Corrupted: replace one node with an unrelated one
    unrelated = [n for n in all_node_ids if n not in pos_nodes]
    corrupted = []
    for _ in range(min(num_corrupted, len(pos_nodes))):
        if unrelated:
            c = list(pos_nodes)
            c[random.randint(0, len(c) - 1)] = random.choice(unrelated)
            corrupted.append(c)

    return negs, corrupted


# ---------------------------------------------------------------------------
# Level 2: Dynamic Cell Construction
# ---------------------------------------------------------------------------

def construct_candidate_cells(
    candidate_indices: List[int],
    r: int = 2,
) -> Tuple[List[List[int]], Dict[int, List[int]]]:
    """Construct all r-subsets from a candidate pool of chunk indices.

    Args:
        candidate_indices: list of K1 chunk indices (from Level 1 top-K1)
        r: cell size (2 = pairs, 3 = triples)

    Returns:
        cells: list of cells, each a list of chunk indices
        chunk_to_cell_positions: chunk_idx → list of cell positions containing it
    """
    cells = [list(combo) for combo in combinations(candidate_indices, r)]
    chunk_to_cell_positions: Dict[int, List[int]] = {}
    for pos, cell in enumerate(cells):
        for ci in cell:
            chunk_to_cell_positions.setdefault(ci, []).append(pos)
    return cells, chunk_to_cell_positions


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def precompute_cell_scatter(
    cell_to_nodes: Dict[int, List[int]],
    device: torch.device,
):
    """
    Build (cell_indices, cells_obj, flat_nodes_t, cell_asgn_t) once.
    Pass flat_nodes_t / cell_asgn_t to DeepSetCellEncoder.forward to avoid
    rebuilding the scatter index arrays on every batch call.
    """
    from toporag.lifting.base import Cell

    cell_indices = list(cell_to_nodes.keys())
    cells_obj = [
        Cell(cell_id=ci, chunk_indices=set(cell_to_nodes[ci]), dimension=len(cell_to_nodes[ci]) - 1)
        for ci in cell_indices
    ]
    flat_nodes, cell_assignments = [], []
    for pos, ci in enumerate(cell_indices):
        nodes = cell_to_nodes[ci]
        flat_nodes.extend(nodes)
        cell_assignments.extend([pos] * len(nodes))
    flat_nodes_t = torch.tensor(flat_nodes, dtype=torch.long, device=device)
    cell_asgn_t = torch.tensor(cell_assignments, dtype=torch.long, device=device)
    idx_map = {ci: pos for pos, ci in enumerate(cell_indices)}
    return idx_map, cells_obj, flat_nodes_t, cell_asgn_t


def precompute_chunk_to_1cells(lifted) -> Dict[int, List[int]]:
    """Build chunk_idx → list of column positions in incidence_1 (1-cells only, not 0-cells)."""
    B1 = lifted.incidence_1
    if B1 is None:
        return {}
    B1_dense = B1.to_dense().cpu()  # (n, m_1)
    chunk_to_1cells: Dict[int, List[int]] = {}
    for col in range(B1_dense.shape[1]):
        node_indices = torch.where(B1_dense[:, col] > 0)[0].tolist()
        for ni in node_indices:
            chunk_to_1cells.setdefault(ni, []).append(col)
    return chunk_to_1cells


def build_query_rows(
    q_raws: torch.Tensor,
    x_0: torch.Tensor,
    m_1: int,
    chunk_to_1cells: Dict[int, List[int]],
    k: int = 10,
    device=None,
) -> torch.Tensor:
    """Build (Q, m_1) incidence rows: query q connects to cell j if any top-k chunk neighbor is in cell j."""
    if device is None:
        device = q_raws.device
    Q = q_raws.shape[0]
    if m_1 == 0:
        return torch.zeros(Q, 0, device=device)
    q_norm = F.normalize(q_raws.to(device), p=2, dim=-1)
    x0_norm = F.normalize(x_0.to(device), p=2, dim=-1)
    sims = q_norm @ x0_norm.T  # (Q, n)
    top_indices = torch.topk(sims, k=min(k, x_0.shape[0]), dim=1).indices.cpu().tolist()
    rows = torch.zeros(Q, m_1, device=device)
    for qi, neighbors in enumerate(top_indices):
        for chunk_idx in neighbors:
            for col in chunk_to_1cells.get(chunk_idx, []):
                rows[qi, col] = 1.0
    return rows


def precompute_cell_mean_raw(
    x_0_raw: torch.Tensor,
    cell_to_nodes: Dict[int, List[int]],
    cell_scatter,   # (idx_map, ...) — same ordering used for cell_embs
    device: torch.device,
) -> torch.Tensor:
    """(M, d) L2-normalised mean raw embedding per cell.
    Ordering matches cell_scatter so cell_mean_raw[pos] corresponds to cell_embs[pos].
    Used as the cosine residual baseline: score += cos(q_raw, cell_mean_raw).
    """
    idx_map = cell_scatter[0]
    M, d = len(idx_map), x_0_raw.shape[1]
    cell_mean = torch.zeros(M, d)
    for cell_id, pos in idx_map.items():
        nodes = cell_to_nodes[cell_id]
        cell_mean[pos] = x_0_raw[nodes].mean(dim=0)
    return F.normalize(cell_mean, p=2, dim=-1).to(device)


def build_cell_embeddings(
    node_embs: torch.Tensor,
    cell_to_nodes: Dict[int, List[int]],
    cell_encoder,
    device: torch.device,
    cells_obj=None,
    flat_nodes_t=None,
    cell_asgn_t=None,
) -> Tuple[Dict[int, int], torch.Tensor]:
    """
    Aggregate node embeddings to cell embeddings using DeepSet.
    Pass precomputed cells_obj/flat_nodes_t/cell_asgn_t (from precompute_cell_scatter)
    to avoid rebuilding index structures every call.
    Returns (cell_idx -> position_in_tensor, cell_embs).
    """
    from toporag.lifting.base import Cell

    if cells_obj is None:
        cell_indices = list(cell_to_nodes.keys())
        cells_obj = [
            Cell(cell_id=ci, chunk_indices=set(cell_to_nodes[ci]), dimension=len(cell_to_nodes[ci]) - 1)
            for ci in cell_indices
        ]
        idx_map = {ci: pos for pos, ci in enumerate(cell_indices)}
    else:
        idx_map = {cell.cell_id: pos for pos, cell in enumerate(cells_obj)}

    from toporag.models.cell_encoder import DeepSetCellEncoder
    if isinstance(cell_encoder, DeepSetCellEncoder):
        cell_embs = cell_encoder(node_embs.to(device), cells_obj,
                                 flat_nodes_t=flat_nodes_t, cell_asgn_t=cell_asgn_t)
    else:
        cell_embs = cell_encoder(node_embs.to(device), cells_obj)
    return idx_map, cell_embs


def train_one_epoch(
    model,
    lifted,
    queries,
    cell_to_nodes,
    embedder,
    optimizer,
    device,
    grad_clip: float = 1.0,
    batch_size: int = 16,
    cell_scatter=None,
    chunk_to_1cells=None,
    B1_dense=None,
    cell_mean_raw=None,
    lambda_l2: float = 0.0,
    k1_candidates: int = 20,
    cell_size: int = 2,
    samples: Optional[List[dict]] = None,
) -> dict:
    """Two-level training epoch.

    Level 1 — Chunk scoring:
      score(q, chunk_i) = cos(q_raw, chunk_raw_i) + tanh(gate) * cos(tnn_q, tnn_chunk_i)
      Loss: InfoNCE over all n chunks.  Positive = random chunk from the target cell.

    Level 2 — Cell scoring (THE NOVELTY):
      1. Get top-K1 candidates from Level 1 scores (detached)
      2. Construct all r-subsets from candidates → dynamic cells
      3. Score cells: ψ(q, σ) = link_predictor([query_proj(q̃); cell_encoder(σ̃)])
      4. Loss: InfoNCE over candidate cells (positive = cells containing gold chunks)

    L_total = L_level1 + λ * L_level2
    """
    model.train()
    n = lifted.x_0.shape[0]
    m_1 = lifted.incidence_1.shape[1] if lifted.incidence_1 is not None else 0
    use_joint = chunk_to_1cells is not None and B1_dense is not None and m_1 > 0
    x_1 = lifted.x_1.to(device) if lifted.x_1 is not None else torch.zeros(m_1, lifted.x_0.shape[1], device=device)

    # Build gold chunk set per question for Level 2 labels
    gold_chunks_per_question: Dict[str, set] = {}
    if lambda_l2 > 0 and samples is not None:
        for sample in samples:
            gold_chunks_per_question[sample["question"]] = set(sample["supporting"])

    # Pre-normalised raw chunk embeddings for cosine baseline term
    raw_chunks_norm = F.normalize(lifted.x_0.to(device), p=2, dim=-1)  # (n, 768)

    # --- Flatten training pairs: keep (cell_idx, q_text) structure ---
    flat_items = [
        (cell_idx, q_text)
        for cell_idx, qs in queries.items()
        for q_text in qs
        if cell_idx in cell_to_nodes
    ]
    random.shuffle(flat_items)

    # --- Pre-embed all queries once ---
    all_q_texts = [q for _, q in flat_items]
    with torch.no_grad():
        q_raws = embedder.encode(all_q_texts, is_query=True, show_progress=False)
        if not isinstance(q_raws, torch.Tensor):
            q_raws = torch.tensor(q_raws, dtype=torch.float32)
        q_raws = q_raws.clone().to(device)  # (N, embed_dim)

    total_loss = 0.0
    total_l1 = 0.0
    total_l2 = 0.0
    n_steps = 0
    l2_steps = 0
    last_diag = {}

    for batch_start in range(0, len(flat_items), batch_size):
        batch = flat_items[batch_start: batch_start + batch_size]
        Q = len(batch)
        q_raws_batch = q_raws[batch_start: batch_start + Q]  # (Q, embed_dim)
        optimizer.zero_grad()

        # Run TNN (joint or standalone)
        if use_joint:
            x_0_aug = torch.cat([lifted.x_0.to(device), q_raws_batch], dim=0)
            query_rows = build_query_rows(q_raws_batch, lifted.x_0, m_1, chunk_to_1cells, k=10, device=device)
            B1_aug = torch.cat([B1_dense.to(device), query_rows], dim=0)
            tnn_out = model.tnn.forward(x_0_aug, x_1, None, B1_aug)
            node_embs = tnn_out.x_0[:n]         # (n, 768) TNN-refined chunks
            q_tnns = tnn_out.x_0[n: n + Q]      # (Q, 768) TNN-refined queries
        else:
            tnn_out = model.tnn.forward_from_lifted(lifted)
            node_embs = tnn_out.x_0.to(device)
            q_tnns = q_raws_batch  # no joint TNN

        # Normalise for cosine similarity
        node_norm = F.normalize(node_embs, p=2, dim=-1)  # (n, 768)

        # ===== Level 1: Chunk-level InfoNCE =====
        batch_losses_l1 = []
        batch_l1_scores = []  # store per-query scores for Level 2 candidate selection
        pos_ranks, score_gaps = [], []

        for i, (cell_idx, _) in enumerate(batch):
            # Pick a random positive chunk from this cell
            pos_chunk = random.choice(cell_to_nodes[cell_idx])

            # Cosine baseline: cos(q_raw, chunk_raw) for all n chunks
            q_raw_norm = F.normalize(q_raws_batch[i].unsqueeze(0), p=2, dim=-1)
            cos_base = (q_raw_norm * raw_chunks_norm).sum(dim=-1)  # (n,)

            # TNN cosine: cos(tnn_q, tnn_chunk) for all n chunks
            q_tnn_norm = F.normalize(q_tnns[i].unsqueeze(0), p=2, dim=-1)
            tnn_cos = (q_tnn_norm * node_norm).sum(dim=-1)  # (n,)

            # Gated combination: starts at pure cosine (gate=0)
            scores = cos_base.detach() + torch.tanh(model.score_gate) * tnn_cos

            batch_losses_l1.append(
                F.cross_entropy(scores.unsqueeze(0),
                                torch.tensor([pos_chunk], device=device))
            )
            batch_l1_scores.append(scores.detach())  # detached for Level 2 candidate selection

            with torch.no_grad():
                rank = (scores > scores[pos_chunk]).sum().item() + 1
                gap = scores[pos_chunk].item() - scores.mean().item()
                pos_ranks.append(rank)
                score_gaps.append(gap)

        loss_l1 = sum(batch_losses_l1) / len(batch_losses_l1) if batch_losses_l1 else torch.tensor(0.0, device=device)

        # ===== Level 2: Cell-level InfoNCE (if enabled) =====
        # IMPORTANT: Detach TNN outputs so L2 gradients don't flow back into TNN.
        # L1 trains TNN + score_gate; L2 only trains cell_encoder + query_proj + link_predictor.
        loss_l2 = torch.tensor(0.0, device=device)
        if lambda_l2 > 0 and batch_l1_scores:
            batch_losses_l2 = []
            node_embs_detached = node_embs.detach()
            q_tnns_detached = q_tnns.detach()

            for i, (cell_idx, q_text) in enumerate(batch):
                # Get gold chunks for this query
                gold_set = gold_chunks_per_question.get(q_text, set())
                if not gold_set:
                    # Fallback: use chunks from the target cell as gold
                    gold_set = set(cell_to_nodes[cell_idx])

                # Level 1 candidate pool: top-K1 chunks by s1 score
                k1 = min(k1_candidates, n)
                top_k1_indices = torch.topk(batch_l1_scores[i], k1).indices.tolist()

                # Construct cells from candidates
                if len(top_k1_indices) < cell_size:
                    continue
                cells, chunk_to_cell_pos = construct_candidate_cells(top_k1_indices, r=cell_size)
                if not cells:
                    continue

                # Label cells: positive if any member is a gold chunk
                cell_labels = []
                for cell in cells:
                    is_pos = any(ci in gold_set for ci in cell)
                    cell_labels.append(is_pos)

                # Need at least one positive and one negative
                if not any(cell_labels) or all(cell_labels):
                    continue

                # Score cells via Level 2: ψ(q, σ) — using DETACHED TNN features
                cell_scores = model.score_cells_from_embeddings(
                    q_tnns_detached[i].unsqueeze(0),  # (1, embed_dim)
                    node_embs_detached,                 # (n, embed_dim)
                    cells,                              # list of cells
                )  # (1, M)
                cell_scores = cell_scores.squeeze(0)  # (M,)

                # Find first positive cell index for cross-entropy
                pos_cell_indices = [j for j, lbl in enumerate(cell_labels) if lbl]
                # Use the highest-scoring positive as target (helps convergence)
                pos_target = max(pos_cell_indices, key=lambda j: cell_scores[j].item())

                batch_losses_l2.append(
                    F.cross_entropy(cell_scores.unsqueeze(0),
                                    torch.tensor([pos_target], device=device))
                )

            if batch_losses_l2:
                loss_l2 = sum(batch_losses_l2) / len(batch_losses_l2)
                total_l2 += loss_l2.item() * len(batch_losses_l2)
                l2_steps += len(batch_losses_l2)

        # Combined loss
        loss = loss_l1 + lambda_l2 * loss_l2

        if batch_losses_l1:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item() * len(batch_losses_l1)
            total_l1 += loss_l1.item() * len(batch_losses_l1)
            n_steps += len(batch_losses_l1)
            if pos_ranks:
                last_diag = {"pos_rank": sum(pos_ranks)/len(pos_ranks),
                             "score_gap": sum(score_gaps)/len(score_gaps)}

    # Gradient norms per module
    grad_norms = {}
    for name, module in [("tnn", model.tnn), ("cell_enc", model.cell_encoder),
                         ("q_proj", model.query_proj), ("lp", model.link_predictor)]:
        g = sum(p.grad.norm().item()**2 for p in module.parameters()
                if p.grad is not None) ** 0.5
        grad_norms[name] = g

    return {
        "loss": total_loss / max(n_steps, 1),
        "loss_l1": total_l1 / max(n_steps, 1),
        "loss_l2": total_l2 / max(l2_steps, 1) if l2_steps > 0 else 0.0,
        "pos_rank": last_diag.get("pos_rank", float("nan")),
        "score_gap": last_diag.get("score_gap", float("nan")),
        "grad_norms": grad_norms,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model,
    lifted,
    samples: List[dict],
    embedder,
    cell_to_nodes: Dict[int, List[int]],
    device: torch.device,
    top_k: int = 5,
    mode: str = "trained",
    cell_scatter=None,
    chunk_to_1cells=None,
    B1_dense=None,
    cell_mean_raw=None,
    k1_candidates: int = 20,
    cell_size: int = 2,
    use_level2: bool = False,
    l2_eval_weight: float = 0.1,
) -> Dict[str, float]:
    """Compute Recall@2 and Recall@5.

    Level 1 — Chunk-level scoring:
      score(q, chunk_i) = cos(q_raw, chunk_i_raw) + tanh(gate) * cos(q_proj, tnn_chunk_i)

    Level 2 (if use_level2=True) — Cell re-ranking:
      1. C_q = top-K1 chunks from Level 1
      2. Construct r-subsets from C_q
      3. Score cells with ψ(q, σ)
      4. Extract top-K chunks from top-scoring cells
    """
    model.eval()
    recalls_2, recalls_5 = [], []

    n = lifted.x_0.shape[0]
    m_1 = lifted.incidence_1.shape[1] if lifted.incidence_1 is not None else 0
    use_joint = (mode == "trained") and chunk_to_1cells is not None and B1_dense is not None and m_1 > 0
    x_1 = lifted.x_1.to(device) if lifted.x_1 is not None else torch.zeros(m_1, lifted.x_0.shape[1], device=device)

    # Pre-normalised raw chunk embeddings for cosine baseline term
    raw_chunks_norm = F.normalize(lifted.x_0.to(device), p=2, dim=-1)  # (n, 768)

    with torch.no_grad():
        # For non-joint trained mode, run TNN once for all queries
        if mode == "trained" and not use_joint:
            tnn_out = model.tnn.forward_from_lifted(lifted)
            tnn_node_embs = tnn_out.x_0.to(device)  # (n, 768)

        for sample in samples:
            gt = sample["supporting"]
            if not gt:
                continue

            q_raw = embedder.encode([sample["question"]], is_query=True, show_progress=False)
            if not isinstance(q_raw, torch.Tensor):
                q_raw = torch.tensor(q_raw)
            q_raw = q_raw.clone().detach().to(device)  # (1, d)

            if mode == "baseline":
                # Pure cosine similarity per chunk
                sims = F.cosine_similarity(q_raw, lifted.x_0.to(device))
                retrieved = torch.topk(sims, min(top_k, len(sims))).indices.tolist()
            else:
                # Chunk-level scoring: cos_base + gate * cos_tnn
                q_norm = F.normalize(q_raw, p=2, dim=-1)  # (1, 768)
                cos_base = (q_norm * raw_chunks_norm).sum(dim=-1)  # (n,)

                if use_joint:
                    # Joint TNN: add query as extra node
                    x_0_aug = torch.cat([lifted.x_0.to(device), q_raw], dim=0)
                    query_row = build_query_rows(q_raw, lifted.x_0, m_1, chunk_to_1cells, k=10, device=device)
                    B1_aug = torch.cat([B1_dense.to(device), query_row], dim=0)
                    tnn_out_q = model.tnn.forward(x_0_aug, x_1, None, B1_aug)
                    node_embs = tnn_out_q.x_0[:n]        # (n, 768) TNN-refined chunks
                    q_tnn = tnn_out_q.x_0[n].unsqueeze(0)  # (1, 768) TNN-refined query
                else:
                    node_embs = tnn_node_embs  # pre-computed
                    q_tnn = q_raw  # no joint TNN, use raw query

                # TNN cosine: similarity in TNN output space (768-dim)
                q_tnn_norm = F.normalize(q_tnn, p=2, dim=-1)   # (1, 768)
                node_norm = F.normalize(node_embs, p=2, dim=-1)  # (n, 768)
                tnn_cos = (q_tnn_norm * node_norm).sum(dim=-1)   # (n,)

                l1_scores = cos_base + torch.tanh(model.score_gate) * tnn_cos

                if use_level2 and mode == "trained":
                    # ===== Level 2: Cell re-ranking =====
                    k1 = min(k1_candidates, n)
                    top_k1_vals, top_k1_idx = torch.topk(l1_scores, k1)
                    candidate_indices = top_k1_idx.tolist()

                    if len(candidate_indices) >= cell_size:
                        cells, chunk_to_cell_pos = construct_candidate_cells(
                            candidate_indices, r=cell_size
                        )

                        if cells:
                            # Score cells: ψ(q, σ)
                            cell_scores = model.score_cells_from_embeddings(
                                q_tnn,       # (1, embed_dim)
                                node_embs,   # (n, embed_dim)
                                cells,       # list of cells
                            ).squeeze(0)   # (M,)

                            # Compute per-chunk cell boost: max cell score for each candidate
                            chunk_cell_boost = torch.zeros(n, device=device)
                            for ci in candidate_indices:
                                if ci in chunk_to_cell_pos:
                                    cell_positions = chunk_to_cell_pos[ci]
                                    chunk_cell_boost[ci] = cell_scores[cell_positions].max()

                            # Normalize cell boost: zero-mean unit-var within candidates
                            boost_vals = chunk_cell_boost[candidate_indices]
                            boost_std = boost_vals.std()
                            if boost_std > 1e-6:
                                chunk_cell_boost = (chunk_cell_boost - boost_vals.mean()) / boost_std

                            # Weighted combination: L1 + weight * normalized_cell_boost
                            combined = l1_scores + l2_eval_weight * chunk_cell_boost
                            retrieved = torch.topk(combined, min(top_k, n)).indices.tolist()
                        else:
                            retrieved = torch.topk(l1_scores, min(top_k, n)).indices.tolist()
                    else:
                        retrieved = torch.topk(l1_scores, min(top_k, n)).indices.tolist()
                else:
                    retrieved = torch.topk(l1_scores, min(top_k, len(l1_scores))).indices.tolist()

            gt_set = set(gt)
            hit2 = len(gt_set & set(retrieved[:2])) / len(gt_set)
            hit5 = len(gt_set & set(retrieved[:5])) / len(gt_set)
            recalls_2.append(hit2)
            recalls_5.append(hit5)

    return {
        "recall_2": sum(recalls_2) / len(recalls_2) if recalls_2 else 0.0,
        "recall_5": sum(recalls_5) / len(recalls_5) if recalls_5 else 0.0,
        "n": len(recalls_2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train TopoRAG on a subset")
    parser.add_argument("--dataset", default="musique", choices=["musique", "2wiki", "hotpotqa"])
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Number of questions to use (50 for quick validation)")
    parser.add_argument("--lifting", default="knn", choices=["knn", "clique", "cycle"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--llm", default="local", choices=["local", "groq", "openai", "none"],
                        help="LLM provider for query generation. 'none' = baseline only, no training.")
    parser.add_argument("--llm_model", type=str, default=None,
                        help="Override LLM model (e.g. Qwen/Qwen2.5-7B-Instruct). "
                             "Default: Qwen/Qwen2.5-3B-Instruct for local.")
    parser.add_argument("--llm_device", type=str, default="auto",
                        help="Device for local LLM: 'auto' (GPU if available), 'cpu', 'cuda'. "
                             "Use 'cpu' for 7B model if GPU VRAM is tight.")
    parser.add_argument("--query_cache", type=str, default=None,
                        help="Path to cached queries JSON. If set and file exists, skips generation entirely.")
    parser.add_argument("--no_per_chunk", action="store_true",
                        help="Disable LP-RAG style per-chunk queries (multi-hop cells only)")
    parser.add_argument("--gold_train", action="store_true",
                        help="Add gold-label training pairs (real MuSiQue questions → gold cells). "
                             "Useful to validate architecture before self-supervised queries are ready.")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size. Reduce if CUDA OOM (InfoNCE scores all cells per item).")
    parser.add_argument("--scoring", default="mlp", choices=["mlp", "dot", "bilinear"],
                        help="Link predictor scoring: 'mlp' (default), 'dot', or 'bilinear'.")
    # Level 2 arguments
    parser.add_argument("--lambda_l2", type=float, default=0.0,
                        help="Weight for Level 2 cell-scoring loss. 0 = Level 1 only (backward compatible).")
    parser.add_argument("--k1_candidates", type=int, default=20,
                        help="Number of top-K1 candidates from Level 1 for cell construction.")
    parser.add_argument("--cell_size", type=int, default=2,
                        help="Size of cells to construct (r-subsets). 2 = pairs, 3 = triples.")
    parser.add_argument("--l2_eval_weight", type=float, default=0.1,
                        help="Weight for L2 cell boost in final score: s = s_L1 + w * cell_boost_normalized.")
    args = parser.parse_args()

    # ---- Setup ----
    from toporag import TopoRAG, TopoRAGConfig
    from toporag.models.lp_tnn import LPTNN

    dataset_paths = {
        "musique": PROJECT_ROOT / "LPGNN-retriever/datasets/musique/musique.json",
        "2wiki":   PROJECT_ROOT / "LPGNN-retriever/datasets/2wiki/2wikimultihopqa.json",
        "hotpotqa": PROJECT_ROOT / "LPGNN-retriever/datasets/hotpotqa/hotpotqa.json",
    }
    data_path = dataset_paths[args.dataset]

    print("=" * 60)
    print(f"TopoRAG Training — {args.dataset} / {args.max_samples} samples")
    print("=" * 60)

    # ---- 1. Load data ----
    print(f"\n[1/5] Loading dataset...")
    chunks, chunk_to_doc, samples = load_musique(data_path, args.max_samples)
    print(f"  {len(chunks)} chunks from {len(samples)} questions")
    supporting_total = sum(len(s["supporting"]) for s in samples)
    print(f"  {supporting_total} gold supporting paragraphs ({supporting_total/len(samples):.1f} avg/question)")

    # ---- 2. Build lifted topology ----
    print(f"\n[2/5] Building k-NN graph + {args.lifting} lifting...")
    config = TopoRAGConfig(
        lifting=args.lifting,
        use_gps=False,   # GPS is separate from LPTNN; keep off for simplicity
        use_tnn=False,   # LPTNN handles TNN internally
        hidden_dim=args.hidden_dim,
    )
    toporag_builder = TopoRAG(config)
    lifted = toporag_builder.build_from_chunks(chunks, chunk_to_doc)
    cell_to_nodes = dict(lifted.cell_to_nodes)
    device = toporag_builder.device
    embedder = toporag_builder.embedder

    print(f"  Nodes: {lifted.num_nodes}, Cells: {lifted.num_edges}, Device: {device}")

    # ---- 3. Generate / load queries ----
    cache_path = Path(args.query_cache) if args.query_cache else (
        REPO_ROOT / f"experiments/cache/{args.dataset}_{args.max_samples}_queries.json"
    )
    # Build LLM kwargs for local provider
    llm_kwargs = {}
    if args.llm == "local":
        llm_kwargs["device"] = args.llm_device
        if args.llm_model:
            llm_kwargs["model_name"] = args.llm_model

    print(f"\n[3/5] Queries (cache: {cache_path})...")
    queries = generate_queries(
        chunks, cell_to_nodes, args.llm, cache_path,
        also_per_chunk=not args.no_per_chunk,
        llm_kwargs=llm_kwargs,
    )

    # Always register 0-cells (one per chunk) in cell_to_nodes.
    # When loading from cache, generate_queries returns early and skips the
    # mutation that adds fake single-chunk cells.  We rebuild them here so:
    #   (a) evaluation can score each chunk directly via its 0-cell (no distractors),
    #   (b) build_gold_queries will create (question, 0-cell) pairs for gold chunks,
    #       enabling direct single-chunk retrieval for each hop of a multi-hop question.
    zero_cell_offset = max(cell_to_nodes.keys(), default=-1) + 1
    for ci in range(len(chunks)):
        cell_to_nodes[zero_cell_offset + ci] = [ci]
    print(f"  Added {len(chunks)} zero-cells (indices {zero_cell_offset}–{zero_cell_offset+len(chunks)-1})")

    # Optionally augment (or replace) with gold label training pairs
    if args.gold_train:
        print("  Adding gold-label training pairs from MuSiQue questions...")
        gold_q = build_gold_queries(samples, cell_to_nodes)
        for cell_idx, qlist in gold_q.items():
            for q in qlist:
                queries.setdefault(cell_idx, []).append(q)
        print(f"  Total after merging gold: {sum(len(v) for v in queries.values())} pairs")

    # ---- 4. Build LPTNN ----
    print(f"\n[4/5] Building LPTNN model...")
    embed_dim = toporag_builder.config.embed_dim
    model = LPTNN(
        embed_dim=embed_dim,
        hidden_dim=args.hidden_dim,
        cell_encoder_type="deepset",
        link_predictor_type=args.scoring,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # ---- Pre-compute all static structures ----
    print("  Pre-computing cell scatter indices...")
    cell_scatter = precompute_cell_scatter(cell_to_nodes, device)
    print(f"  Cell scatter: {len(cell_to_nodes)} cells, "
          f"{cell_scatter[2].shape[0]} total node-cell incidences")
    print("  Pre-computing joint TNN + cosine residual structures...")
    chunk_to_1cells = precompute_chunk_to_1cells(lifted)
    B1_dense = lifted.incidence_1.to_dense().cpu() if lifted.incidence_1 is not None else None
    cell_mean_raw = precompute_cell_mean_raw(lifted.x_0, cell_to_nodes, cell_scatter, device)
    print(f"  Chunk-to-1cell map: {len(chunk_to_1cells)} chunks | "
          f"cell_mean_raw: {cell_mean_raw.shape}")

    # ---- Baseline evaluation ----
    print("\n  [Baseline] Evaluating cosine similarity (no training)...")
    m_base = evaluate(model, lifted, samples, embedder, cell_to_nodes, device,
                      top_k=args.top_k, mode="baseline", cell_scatter=cell_scatter)
    print(f"  Baseline  R@2={m_base['recall_2']*100:.1f}%  R@5={m_base['recall_5']*100:.1f}%")

    if not queries:
        print("\n  No queries — showing baseline only. Re-run with --llm groq to train.")
        return

    # ---- 5. Single-phase training ----
    # No freezing. TNN at 0.1× LR throughout (differential learning rates).
    # Cosine residual anchors scoring to baseline from epoch 1.
    # Warmup for first 10% of epochs to let random weights stabilise before big updates.
    use_l2 = args.lambda_l2 > 0
    n_pairs = sum(len(v) for v in queries.values())
    warmup = max(args.epochs // 10, 3)
    print(f"\n[5/5] Training on {n_pairs} query-cell pairs | {args.epochs} epochs")
    print(f"      Warmup: {warmup} epochs | LR: {args.lr} (TNN: {args.lr*0.1:.0e})")
    print(f"      Scoring: {args.scoring} + cosine residual (gated)")
    if use_l2:
        n_cells_per_query = math.comb(args.k1_candidates, args.cell_size)
        print(f"      Level 2: λ={args.lambda_l2}, K1={args.k1_candidates}, "
              f"r={args.cell_size} → {n_cells_per_query} cells/query")

    if use_l2:
        print(f"{'Ep':>4}  {'L1':>7}  {'L2':>7}  {'R@2':>6}  {'R@5':>6}  {'R@5L2':>6}  "
              f"{'gate':>6}  {'g/tnn':>7}  {'g/enc':>7}  {'g/qp':>7}  {'g/lp':>7}")
    else:
        print(f"{'Ep':>4}  {'loss':>7}  {'R@2':>6}  {'R@5':>6}  {'rank':>5}  "
              f"{'gap':>6}  {'gate':>6}  {'g/tnn':>7}  {'g/enc':>7}  {'g/qp':>7}")

    tnn_params   = list(model.tnn.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("tnn")]
    optimizer = torch.optim.Adam([
        {"params": other_params, "lr": args.lr},
        {"params": tnn_params,   "lr": args.lr * 0.1},
    ])

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        progress = (ep - warmup) / max(args.epochs - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_recall5 = 0.0
    best_state: Optional[dict] = None

    for epoch in range(args.epochs):
        d = train_one_epoch(model, lifted, queries, cell_to_nodes, embedder, optimizer, device,
                            batch_size=args.batch_size, cell_scatter=cell_scatter,
                            chunk_to_1cells=chunk_to_1cells, B1_dense=B1_dense,
                            cell_mean_raw=cell_mean_raw,
                            lambda_l2=args.lambda_l2,
                            k1_candidates=args.k1_candidates,
                            cell_size=args.cell_size,
                            samples=samples)
        scheduler.step()

        # Evaluate Level 1 only
        m = evaluate(model, lifted, samples, embedder, cell_to_nodes, device,
                     top_k=args.top_k, cell_scatter=cell_scatter,
                     chunk_to_1cells=chunk_to_1cells, B1_dense=B1_dense,
                     cell_mean_raw=cell_mean_raw,
                     use_level2=False)

        # Evaluate Level 2 (if enabled) — separate metric
        m_l2 = None
        if use_l2:
            m_l2 = evaluate(model, lifted, samples, embedder, cell_to_nodes, device,
                            top_k=args.top_k, cell_scatter=cell_scatter,
                            chunk_to_1cells=chunk_to_1cells, B1_dense=B1_dense,
                            cell_mean_raw=cell_mean_raw,
                            k1_candidates=args.k1_candidates,
                            cell_size=args.cell_size,
                            use_level2=True,
                            l2_eval_weight=args.l2_eval_weight)

        # Track best by L1 R@5 (L2 is secondary, needs more epochs to converge)
        eval_r5 = m["recall_5"]
        improved = ""
        if eval_r5 > best_recall5:
            best_recall5 = eval_r5
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            improved = " *"

        gn = d["grad_norms"]
        gate_val = torch.tanh(model.score_gate).item()

        if use_l2:
            print(f"{epoch+1:>4}  {d['loss_l1']:>7.4f}  {d['loss_l2']:>7.4f}  "
                  f"{m['recall_2']*100:>5.1f}%  {m['recall_5']*100:>5.1f}%  "
                  f"{m_l2['recall_5']*100:>5.1f}%"
                  f"  {gate_val:>6.3f}"
                  f"  {gn['tnn']:>7.4f}  {gn['cell_enc']:>7.4f}  {gn['q_proj']:>7.4f}"
                  f"  {gn.get('lp', 0):>7.4f}{improved}")
        else:
            print(f"{epoch+1:>4}  {d['loss']:>7.4f}  {m['recall_2']*100:>5.1f}%  {m['recall_5']*100:>5.1f}%"
                  f"  {d['pos_rank']:>5.1f}  {d['score_gap']:>6.3f}"
                  f"  {gate_val:>6.3f}"
                  f"  {gn['tnn']:>7.4f}  {gn['cell_enc']:>7.4f}  {gn['q_proj']:>7.4f}{improved}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  Restored best checkpoint (R@5={best_recall5*100:.1f}%)")

    # ---- Final results ----
    m_trained = evaluate(model, lifted, samples, embedder, cell_to_nodes, device,
                         top_k=args.top_k, cell_scatter=cell_scatter,
                         chunk_to_1cells=chunk_to_1cells, B1_dense=B1_dense,
                         cell_mean_raw=cell_mean_raw,
                         use_level2=False)

    m_trained_l2 = None
    if use_l2:
        m_trained_l2 = evaluate(model, lifted, samples, embedder, cell_to_nodes, device,
                                top_k=args.top_k, cell_scatter=cell_scatter,
                                chunk_to_1cells=chunk_to_1cells, B1_dense=B1_dense,
                                cell_mean_raw=cell_mean_raw,
                                k1_candidates=args.k1_candidates,
                                cell_size=args.cell_size,
                                use_level2=True,
                                l2_eval_weight=args.l2_eval_weight)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Method':<30} {'R@2':>8} {'R@5':>8}")
    print("-" * 50)
    print(f"{'Cosine baseline':<30} {m_base['recall_2']*100:>7.1f}% {m_base['recall_5']*100:>7.1f}%")
    print(f"{'TopoRAG L1 (chunk)':<30} {m_trained['recall_2']*100:>7.1f}% {m_trained['recall_5']*100:>7.1f}%")
    if m_trained_l2:
        print(f"{'TopoRAG L1+L2 (cell)':<30} {m_trained_l2['recall_2']*100:>7.1f}% {m_trained_l2['recall_5']*100:>7.1f}%")
    print("-" * 50)
    print(f"{'LP-RAG (target)':<30} {'~45%':>8} {'~55%':>8}")
    print(f"{'GFM-RAG (SOTA)':<30} {'49.1%':>8} {'58.2%':>8}")
    print("=" * 60)
    print(f"\nNote: these numbers are on {args.max_samples} samples (subset).")
    print("Scale to full dataset once results look promising.")


if __name__ == "__main__":
    main()
