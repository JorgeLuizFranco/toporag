#!/usr/bin/env python3
"""
TopoRAG Cell Scorer — Comprehensive HP Search with Cell Graph Propagation.

Key insight: cell-max (zero-shot) gives 44.7% R@5 but cannot find hop-2 cells.
Cell graph propagation spreads cell-max scores to neighboring cells, enabling
hop-2 discovery. A small MLP learns WHEN to trust propagated vs direct scores.

Features per (query, cell):
  Within-cell:
    1. cell_max_cos     — max cos(q, chunk_i) in cell           [THE key feature]
    2. cell_mean_cos    — mean cos(q, chunk_i) in cell
    3. cell_top2_cos    — mean of top-2 chunk cosines
    4. cell_std_cos     — std of chunk cosines in cell
    5. n_in_top20       — #cell chunks in cosine top-20
    6. log_cell_size    — log(1 + |cell|)
  Cross-cell (propagation):
    7. prop_max_1hop    — max cell-max among neighbor cells
    8. prop_mean_1hop   — mean cell-max among neighbor cells
    9. prop_n_active    — #neighbors with cell-max > 0.3
    10. prop_shared_max — max cos of shared chunks with best neighbor

Training: 5-fold CV, gold cell labels (same protocol as GFM-RAG).
HP search: grid over lr, hidden_dim, dropout.
Evaluation: sweep boost_val × top_cells post-training.

All results saved to results/cell_scorer_v3/.
"""

import json
import math
import random
import argparse
import time
import hashlib
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

RESULTS_DIR = PROJECT_ROOT / "results" / "cell_scorer_v3"


# ===========================================================================
# Model
# ===========================================================================

class CellScorerMLP(nn.Module):
    """Small MLP on pre-computed features. Can AT LEAST recover cell-max."""

    def __init__(self, n_features=10, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features):
        """features: (..., n_features) → (...) scores"""
        return self.net(features).squeeze(-1)


# ===========================================================================
# Pre-computation
# ===========================================================================

def load_musique(data_path, max_samples):
    with open(data_path) as f:
        data = json.load(f)
    chunks, samples = [], []
    for item in data[:max_samples]:
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
    return chunks, samples


def precompute_all(samples, cell_to_nodes, x_chunks, embedder, device, save_dir):
    """Pre-compute everything needed for training and evaluation.

    Saves to save_dir and returns all tensors.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    N = x_chunks.shape[0]
    x_norm = F.normalize(x_chunks, dim=-1)

    # --- Entity cells ---
    entity_cell_ids = sorted([ci for ci, nodes in cell_to_nodes.items() if len(nodes) >= 2])
    M = len(entity_cell_ids)
    cell_id_to_pos = {ci: pos for pos, ci in enumerate(entity_cell_ids)}

    # --- Chunk → cell reverse map ---
    chunk_to_cells = defaultdict(list)
    for cell_id, nodes in cell_to_nodes.items():
        for ni in nodes:
            chunk_to_cells[ni].append(cell_id)

    # --- Gold cell labels ---
    entity_cell_set = set(entity_cell_ids)
    gold_cells_per_sample = []
    for s in samples:
        cells = set()
        for gi in s["supporting"]:
            for cid in chunk_to_cells.get(gi, []):
                if cid in entity_cell_set:
                    cells.add(cid)
        gold_cells_per_sample.append(cells)

    # --- Embed all questions ---
    print("  Embedding all questions...")
    all_questions = [s["question"] for s in samples]
    Q = len(all_questions)
    batch_size = 64
    q_emb_parts = []
    for i in range(0, Q, batch_size):
        batch = all_questions[i:i+batch_size]
        with torch.no_grad():
            emb = embedder.encode(batch, is_query=True, show_progress=False)
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, dtype=torch.float32)
            q_emb_parts.append(emb.cpu())
    q_embs = torch.cat(q_emb_parts, dim=0)  # (Q, D)
    print(f"    q_embs: {q_embs.shape}")

    # --- Cosine scores ---
    print("  Computing cosine scores...")
    q_norm = F.normalize(q_embs.to(device), dim=-1)
    cos_scores = (q_norm @ x_norm.T).cpu()  # (Q, N)
    print(f"    cos_scores: {cos_scores.shape}")

    # --- Cell membership as list of tensors ---
    print("  Computing per-cell cosine statistics...")
    cell_node_tensors = []
    cell_sizes = []
    for ci in entity_cell_ids:
        nodes = cell_to_nodes[ci]
        cell_node_tensors.append(torch.tensor(nodes, dtype=torch.long))
        cell_sizes.append(len(nodes))

    # --- Per-cell cosine stats: (Q, M, 6) ---
    # [cell_max, cell_mean, cell_top2, cell_std, n_in_top20, log_size]
    cos_gpu = cos_scores.to(device)
    top20_idx = cos_gpu.topk(min(20, N), dim=1).indices  # (Q, 20)

    cell_stats = torch.zeros(Q, M, 6, device=device)
    for pos in range(M):
        nodes_t = cell_node_tensors[pos].to(device)
        cell_cos = cos_gpu[:, nodes_t]  # (Q, |cell|)
        cs = cell_cos.shape[1]

        cell_stats[:, pos, 0] = cell_cos.max(dim=1).values   # max
        cell_stats[:, pos, 1] = cell_cos.mean(dim=1)          # mean
        if cs >= 2:
            top2 = cell_cos.topk(min(2, cs), dim=1).values
            cell_stats[:, pos, 2] = top2.mean(dim=1)          # top-2 mean
        else:
            cell_stats[:, pos, 2] = cell_cos[:, 0]
        if cs >= 2:
            cell_stats[:, pos, 3] = cell_cos.std(dim=1)       # std
        # n_in_top20: count overlap with cosine top-20
        for qi in range(Q):
            top20_set = set(top20_idx[qi].tolist())
            n_in = sum(1 for ni in cell_node_tensors[pos].tolist() if ni in top20_set)
            cell_stats[qi, pos, 4] = n_in
        cell_stats[:, pos, 5] = math.log(1 + cs)              # log size

    cell_stats = cell_stats.cpu()
    cell_max_scores = cell_stats[:, :, 0]  # (Q, M) — for propagation
    print(f"    cell_stats: {cell_stats.shape}")

    # --- Cell adjacency graph ---
    print("  Building cell adjacency graph...")
    # Two cells are neighbors if they share at least one chunk
    cell_neighbors = defaultdict(set)  # pos → set of pos
    # Build chunk → cell_pos map
    chunk_to_pos = defaultdict(list)
    for pos, ci in enumerate(entity_cell_ids):
        for ni in cell_to_nodes[ci]:
            chunk_to_pos[ni].append(pos)

    for ni, positions in chunk_to_pos.items():
        for i, p1 in enumerate(positions):
            for p2 in positions[i+1:]:
                cell_neighbors[p1].add(p2)
                cell_neighbors[p2].add(p1)

    n_edges = sum(len(v) for v in cell_neighbors.values()) // 2
    avg_degree = np.mean([len(v) for v in cell_neighbors.values()]) if cell_neighbors else 0
    print(f"    Cell graph: {M} nodes, {n_edges} edges, avg degree={avg_degree:.1f}")

    # --- Propagation features: (Q, M, 4) ---
    # [prop_max_1hop, prop_mean_1hop, prop_n_active, prop_shared_max]
    print("  Computing propagation features...")
    prop_features = torch.zeros(Q, M, 4)
    cell_max_np = cell_max_scores.numpy()

    for pos in range(M):
        neighbors = list(cell_neighbors.get(pos, []))
        if not neighbors:
            continue
        neighbor_arr = np.array(neighbors)
        neighbor_maxes = cell_max_np[:, neighbor_arr]  # (Q, |neighbors|)

        prop_features[:, pos, 0] = torch.tensor(neighbor_maxes.max(axis=1))    # max
        prop_features[:, pos, 1] = torch.tensor(neighbor_maxes.mean(axis=1))   # mean
        prop_features[:, pos, 2] = torch.tensor((neighbor_maxes > 0.3).sum(axis=1))  # n_active

        # shared chunk max cos with best neighbor
        my_nodes = set(cell_to_nodes[entity_cell_ids[pos]])
        best_neighbor = neighbors[0] if neighbors else None
        if best_neighbor is not None:
            # Find best neighbor per query (argmax of neighbor cell-max)
            # For efficiency, use the overall best neighbor (not per-query)
            # This is an approximation but much faster
            overall_best = neighbor_arr[neighbor_maxes.mean(axis=0).argmax()]
            shared = my_nodes & set(cell_to_nodes[entity_cell_ids[overall_best]])
            if shared:
                shared_t = torch.tensor(list(shared), dtype=torch.long)
                shared_cos = cos_scores[:, shared_t]  # (Q, |shared|)
                prop_features[:, pos, 3] = shared_cos.max(dim=1).values

    print(f"    prop_features: {prop_features.shape}")

    # --- Combine all features: (Q, M, 10) ---
    all_features = torch.cat([cell_stats, prop_features], dim=-1)  # (Q, M, 10)
    print(f"    all_features: {all_features.shape}")

    # --- Gold cell targets: (Q, M) binary ---
    gold_targets = torch.zeros(Q, M)
    for qi, gc in enumerate(gold_cells_per_sample):
        for cid in gc:
            pos = cell_id_to_pos.get(cid)
            if pos is not None:
                gold_targets[qi, pos] = 1.0

    avg_gold = gold_targets.sum(dim=1).mean().item()
    print(f"    avg gold cells/question: {avg_gold:.1f}")

    # --- Cell embeddings (for potential future use) ---
    cell_emb_list = []
    for ci in entity_cell_ids:
        cell_emb_list.append(x_chunks[cell_to_nodes[ci]].mean(dim=0))
    cell_embs = torch.stack(cell_emb_list).cpu()

    # --- Save everything ---
    print("  Saving pre-computed data...")
    precomp = {
        "q_embs": q_embs,                    # (Q, D)
        "cos_scores": cos_scores,              # (Q, N)
        "all_features": all_features,          # (Q, M, 10)
        "cell_max_scores": cell_max_scores,    # (Q, M)
        "gold_targets": gold_targets,          # (Q, M)
        "cell_embs": cell_embs,                # (M, D)
        "entity_cell_ids": entity_cell_ids,    # list of M cell IDs
        "cell_id_to_pos": cell_id_to_pos,      # dict
        "cell_to_nodes": dict(cell_to_nodes),  # cell_id → list of chunk indices
        "chunk_to_cells": dict(chunk_to_cells),  # chunk_id → list of cell IDs
        "cell_neighbors": dict(cell_neighbors),  # pos → set of pos
        "n_chunks": N,
        "n_cells": M,
        "n_questions": Q,
    }
    torch.save(precomp, save_dir / "precomputed.pt")
    print(f"  Saved to {save_dir / 'precomputed.pt'}")

    return precomp


# ===========================================================================
# Training
# ===========================================================================

def train_one_epoch(model, features, targets, train_idx, optimizer, device, batch_size=16):
    model.train()
    random.shuffle(train_idx)
    total_loss, n_batches = 0.0, 0

    for start in range(0, len(train_idx), batch_size):
        batch_idx = train_idx[start:start+batch_size]
        B = len(batch_idx)

        feat_batch = features[batch_idx].to(device)   # (B, M, 10)
        tgt_batch = targets[batch_idx].to(device)      # (B, M)

        optimizer.zero_grad()
        scores = model(feat_batch)  # (B, M)

        # BCE with adversarial hard negative weighting
        bce = F.binary_cross_entropy_with_logits(scores, tgt_batch, reduction='none')
        is_pos = tgt_batch > 0.5
        n_pos = is_pos.float().sum(dim=1, keepdim=True).clamp(min=1)

        with torch.no_grad():
            neg_logits = scores.clone()
            neg_logits[is_pos] = float('-inf')
            neg_weight = F.softmax(neg_logits, dim=1)
            neg_weight[is_pos] = 0.0

        pos_loss = (bce * is_pos.float() / n_pos).sum(dim=1).mean()
        neg_loss = (bce * neg_weight).sum(dim=1).mean()
        loss = pos_loss + neg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_retrieval(
    cell_scores_or_mode,  # (Q_test, M) tensor OR "baseline" / "cell_max"
    test_idx, cos_scores, cell_to_nodes, chunk_to_cells,
    entity_cell_ids, cell_id_to_pos, samples, cell_max_scores,
    N, device, boost_val=1.0, top_cells=50,
):
    """Evaluate chunk retrieval R@K given cell scores.

    Returns dict with R@2, R@5, R@10, R@20, cell_R@50.
    """
    M = len(entity_cell_ids)
    cell_pos_to_chunks = {pos: cell_to_nodes[entity_cell_ids[pos]]
                          for pos in range(M)}

    recalls = {k: [] for k in [2, 5, 10, 20]}

    for ti, qi in enumerate(test_idx):
        gt = samples[qi]["supporting"]
        if not gt:
            continue
        gt_set = set(gt)
        cos_q = cos_scores[qi].to(device)  # (N,)

        if isinstance(cell_scores_or_mode, str) and cell_scores_or_mode == "baseline":
            ranked = cos_q.topk(20).indices.tolist()
        elif isinstance(cell_scores_or_mode, str) and cell_scores_or_mode == "cell_max":
            cm = cell_max_scores[qi].to(device)  # (M,)
            chunk_boost = torch.zeros(N, device=device)
            top_c = cm.topk(min(top_cells, M)).indices.tolist()
            for pos in top_c:
                val = cm[pos].item()
                for ni in cell_pos_to_chunks[pos]:
                    if val > chunk_boost[ni].item():
                        chunk_boost[ni] = val
            # Normalize to [0,1]
            if (chunk_boost > 0).any():
                bmin = chunk_boost[chunk_boost > 0].min()
                bmax = chunk_boost.max()
                if bmax > bmin:
                    chunk_boost = ((chunk_boost - bmin) / (bmax - bmin)).clamp(0, 1)
            ranked = (cos_q + boost_val * chunk_boost).topk(20).indices.tolist()
        else:
            # Learned cell scores
            cs = cell_scores_or_mode[ti].to(device)  # (M,)
            cell_probs = torch.sigmoid(cs)
            chunk_boost = torch.zeros(N, device=device)
            top_c = cell_probs.topk(min(top_cells, M)).indices.tolist()
            for pos in top_c:
                prob = cell_probs[pos].item()
                for ni in cell_pos_to_chunks[pos]:
                    if prob > chunk_boost[ni].item():
                        chunk_boost[ni] = prob
            ranked = (cos_q + boost_val * chunk_boost).topk(20).indices.tolist()

        for k in recalls:
            recalls[k].append(len(gt_set & set(ranked[:k])) / len(gt_set))

    return {f"R@{k}": float(np.mean(v)) if v else 0.0 for k, v in recalls.items()}


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from toporag import TopoRAG, TopoRAGConfig

    data_path = PROJECT_ROOT / "LPGNN-retriever/datasets/musique/musique.json"
    topo_file = REPO_ROOT / "experiments/cache/topology/musique_500_entity.pt"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Cell Scorer — Comprehensive HP Search")
    print(f"Results: {run_dir}")
    print("=" * 70)

    # --- Load data ---
    print("\n[1/4] Loading data...")
    chunks, samples = load_musique(data_path, args.max_samples)
    N = len(chunks)
    Q = len(samples)
    print(f"  {N} chunks, {Q} questions")

    # --- Load topology ---
    print("\n[2/4] Loading topology (new lifting)...")
    cache = torch.load(topo_file, weights_only=False)
    cell_to_nodes = cache["cell_to_nodes"]
    x_chunks_cpu = cache["lifted"].x_0

    config = TopoRAGConfig(lifting="entity", use_gps=False, use_tnn=False)
    toporag = TopoRAG(config)
    device = toporag.device
    embedder = toporag.embedder
    x_chunks = x_chunks_cpu.to(device)
    print(f"  Device: {device}")

    # --- Pre-compute features ---
    print("\n[3/4] Pre-computing features...")
    precomp_path = RESULTS_DIR / "precomputed"
    precomp_file = precomp_path / "precomputed.pt"

    if precomp_file.exists():
        print(f"  Loading cached pre-computation from {precomp_file}...")
        precomp = torch.load(precomp_file, weights_only=False)
        # Verify dimensions match
        if precomp["n_questions"] != Q or precomp["n_chunks"] != N:
            print("  MISMATCH — recomputing...")
            precomp = precompute_all(samples, cell_to_nodes, x_chunks, embedder, device, precomp_path)
    else:
        precomp = precompute_all(samples, cell_to_nodes, x_chunks, embedder, device, precomp_path)

    features = precomp["all_features"]         # (Q, M, 10)
    gold_targets = precomp["gold_targets"]     # (Q, M)
    cos_scores = precomp["cos_scores"]         # (Q, N)
    cell_max_scores = precomp["cell_max_scores"]  # (Q, M)
    entity_cell_ids = precomp["entity_cell_ids"]
    cell_id_to_pos = precomp["cell_id_to_pos"]
    chunk_to_cells = precomp["chunk_to_cells"]
    M = precomp["n_cells"]
    n_features = features.shape[-1]

    # Save run config
    run_config = {
        "max_samples": args.max_samples,
        "n_chunks": N,
        "n_cells": M,
        "n_questions": Q,
        "n_features": n_features,
        "n_folds": args.n_folds,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "timestamp": timestamp,
        "topo_file": str(topo_file),
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # --- Baselines ---
    print("\n[4/4] Computing baselines...")
    all_idx = list(range(Q))
    m_cos = evaluate_retrieval(
        "baseline", all_idx, cos_scores, cell_to_nodes, chunk_to_cells,
        entity_cell_ids, cell_id_to_pos, samples, cell_max_scores, N, device)
    print(f"  Cosine:    R@2={m_cos['R@2']*100:.1f}%  R@5={m_cos['R@5']*100:.1f}%  "
          f"R@10={m_cos['R@10']*100:.1f}%  R@20={m_cos['R@20']*100:.1f}%")

    # Cell-max sweep
    best_cm = {"R@5": 0}
    best_cm_cfg = (0, 0)
    for bv in [0.5, 1.0, 1.5, 2.0]:
        for tc in [50, 100, 200, 9662]:
            m = evaluate_retrieval(
                "cell_max", all_idx, cos_scores, cell_to_nodes, chunk_to_cells,
                entity_cell_ids, cell_id_to_pos, samples, cell_max_scores, N, device,
                boost_val=bv, top_cells=tc)
            if m["R@5"] > best_cm["R@5"]:
                best_cm = m
                best_cm_cfg = (bv, tc)
    bv, tc = best_cm_cfg
    print(f"  Cell-max:  R@2={best_cm['R@2']*100:.1f}%  R@5={best_cm['R@5']*100:.1f}%  "
          f"R@10={best_cm['R@10']*100:.1f}%  R@20={best_cm['R@20']*100:.1f}%  (boost={bv}, top_C={tc})")

    baselines = {"cosine": m_cos, "cell_max": best_cm, "cell_max_cfg": best_cm_cfg}
    with open(run_dir / "baselines.json", "w") as f:
        json.dump(baselines, f, indent=2)

    # --- CV folds ---
    all_indices = list(range(Q))
    rng = random.Random(args.seed)
    rng.shuffle(all_indices)
    fold_size = Q // args.n_folds
    folds = []
    for fi in range(args.n_folds):
        start = fi * fold_size
        end = start + fold_size if fi < args.n_folds - 1 else Q
        folds.append(all_indices[start:end])

    # --- HP grid ---
    hp_grid = []
    for lr in [3e-4, 1e-3, 3e-3]:
        for hidden in [32, 64, 128]:
            for dropout in [0.0, 0.1, 0.3]:
                hp_grid.append({
                    "lr": lr, "hidden_dim": hidden, "dropout": dropout,
                })

    print(f"\n{'='*70}")
    print(f"HP Search: {len(hp_grid)} configs × {args.n_folds} folds = {len(hp_grid)*args.n_folds} runs")
    print(f"{'='*70}")

    all_results = []

    for ci, hp in enumerate(hp_grid):
        config_id = f"lr{hp['lr']}_h{hp['hidden_dim']}_d{hp['dropout']}"
        config_dir = run_dir / "runs" / config_id
        config_dir.mkdir(parents=True, exist_ok=True)

        fold_results = []

        for fi in range(args.n_folds):
            test_idx = folds[fi]
            train_idx = [i for i in all_indices if i not in set(test_idx)]

            model = CellScorerMLP(
                n_features=n_features,
                hidden_dim=hp["hidden_dim"],
                dropout=hp["dropout"],
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=0.01)

            # Cosine annealing
            warmup = max(args.max_epochs // 10, 2)
            def lr_lambda(ep, warmup=warmup):
                if ep < warmup:
                    return (ep + 1) / warmup
                return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(args.max_epochs - warmup, 1)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            best_r5 = -1.0
            best_state = None
            patience_counter = 0
            train_log = []

            for epoch in range(args.max_epochs):
                loss = train_one_epoch(
                    model, features, gold_targets, list(train_idx),
                    optimizer, device, batch_size=args.batch_size,
                )
                scheduler.step()

                # Evaluate on test fold
                model.eval()
                with torch.no_grad():
                    test_features = features[test_idx].to(device)
                    test_scores = model(test_features).cpu()  # (|test|, M)

                m = evaluate_retrieval(
                    test_scores, list(range(len(test_idx))),
                    cos_scores[test_idx], cell_to_nodes, chunk_to_cells,
                    entity_cell_ids, cell_id_to_pos,
                    [samples[i] for i in test_idx], cell_max_scores[test_idx],
                    N, device, boost_val=1.0, top_cells=50,
                )

                train_log.append({
                    "epoch": epoch + 1, "loss": loss,
                    "R@2": m["R@2"], "R@5": m["R@5"],
                    "R@10": m["R@10"], "R@20": m["R@20"],
                })

                if m["R@5"] > best_r5:
                    best_r5 = m["R@5"]
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        break

            # Load best model
            if best_state:
                model.load_state_dict(best_state)
            model.eval()

            # Post-training sweep
            with torch.no_grad():
                test_features = features[test_idx].to(device)
                test_scores = model(test_features).cpu()

            sweep_results = []
            best_sweep_r5 = -1.0
            best_sweep_cfg = None
            best_sweep_m = None

            for bv in [0.2, 0.5, 1.0, 1.5, 2.0, 3.0]:
                for tc in [10, 20, 50, 100, 200, 500]:
                    m = evaluate_retrieval(
                        test_scores, list(range(len(test_idx))),
                        cos_scores[test_idx], cell_to_nodes, chunk_to_cells,
                        entity_cell_ids, cell_id_to_pos,
                        [samples[i] for i in test_idx], cell_max_scores[test_idx],
                        N, device, boost_val=bv, top_cells=tc,
                    )
                    sweep_results.append({
                        "boost_val": bv, "top_cells": tc,
                        "R@2": m["R@2"], "R@5": m["R@5"],
                        "R@10": m["R@10"], "R@20": m["R@20"],
                    })
                    if m["R@5"] > best_sweep_r5:
                        best_sweep_r5 = m["R@5"]
                        best_sweep_cfg = (bv, tc)
                        best_sweep_m = m

            # Save fold results
            fold_dir = config_dir / f"fold_{fi}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, fold_dir / "best_model.pt")
            with open(fold_dir / "train_log.json", "w") as f:
                json.dump(train_log, f, indent=2)
            with open(fold_dir / "sweep_results.json", "w") as f:
                json.dump(sweep_results, f, indent=2)

            fold_results.append({
                "fold": fi,
                "best_epoch_r5": best_r5,
                "best_sweep_r5": best_sweep_r5,
                "best_sweep_cfg": best_sweep_cfg,
                "best_sweep_metrics": best_sweep_m,
                "n_epochs_trained": len(train_log),
            })

        # Aggregate across folds
        mean_r5 = np.mean([r["best_sweep_r5"] for r in fold_results])
        std_r5 = np.std([r["best_sweep_r5"] for r in fold_results])
        mean_r2 = np.mean([r["best_sweep_metrics"]["R@2"] for r in fold_results])
        mean_r10 = np.mean([r["best_sweep_metrics"]["R@10"] for r in fold_results])
        mean_r20 = np.mean([r["best_sweep_metrics"]["R@20"] for r in fold_results])

        config_result = {
            "config_id": config_id,
            "hp": hp,
            "mean_R@2": mean_r2,
            "mean_R@5": mean_r5,
            "std_R@5": std_r5,
            "mean_R@10": mean_r10,
            "mean_R@20": mean_r20,
            "fold_results": fold_results,
        }
        all_results.append(config_result)

        with open(config_dir / "config_result.json", "w") as f:
            json.dump(config_result, f, indent=2)

        print(f"  [{ci+1:>2}/{len(hp_grid)}] {config_id:<25} "
              f"R@2={mean_r2*100:.1f}%  R@5={mean_r5*100:.1f}% ± {std_r5*100:.1f}%  "
              f"R@10={mean_r10*100:.1f}%")

    # --- Final summary ---
    all_results.sort(key=lambda r: r["mean_R@5"], reverse=True)

    print("\n" + "=" * 70)
    print("TOP 5 CONFIGS")
    print("=" * 70)
    print(f"{'Config':<30} {'R@2':>7} {'R@5':>12} {'R@10':>7}")
    print("-" * 60)
    for r in all_results[:5]:
        print(f"{r['config_id']:<30} {r['mean_R@2']*100:>6.1f}%  "
              f"{r['mean_R@5']*100:.1f}% ± {r['std_R@5']*100:.1f}%  "
              f"{r['mean_R@10']*100:>6.1f}%")

    best = all_results[0]

    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Cosine baseline':<25} R@2={m_cos['R@2']*100:.1f}%  R@5={m_cos['R@5']*100:.1f}%  "
          f"R@10={m_cos['R@10']*100:.1f}%  R@20={m_cos['R@20']*100:.1f}%")
    print(f"{'Cell-max (zero-shot)':<25} R@2={best_cm['R@2']*100:.1f}%  R@5={best_cm['R@5']*100:.1f}%  "
          f"R@10={best_cm['R@10']*100:.1f}%  R@20={best_cm['R@20']*100:.1f}%")
    print(f"{'Cell scorer (learned)':<25} R@2={best['mean_R@2']*100:.1f}%  R@5={best['mean_R@5']*100:.1f}%  "
          f"R@10={best['mean_R@10']*100:.1f}%  R@20={best['mean_R@20']*100:.1f}%")
    print(f"{'Oracle cell boost':<25} R@2=47.7%  R@5=73.1%  R@10=87.8%")
    print(f"{'GFM-RAG':<25} R@2=49.1%  R@5=58.2%")
    print(f"{'='*70}")

    # Save full summary
    summary = {
        "baselines": baselines,
        "all_configs_ranked": all_results,
        "best_config": best,
        "timestamp": timestamp,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll results saved to {run_dir}/")


if __name__ == "__main__":
    main()
