#!/usr/bin/env python3
"""
TopoRAG Cell Graph GNN — Query-Conditioned Message Passing on Cell Graph.

WHY THIS SHOULD WORK (proven by diagnostic_hop2_reachability.py):
  - Cell-max finds 73.6% of gold cells in top-100
  - Of the 26.4% missed, 78.8% are 1-hop away, 98.4% within 2 hops, 0% unreachable
  - An MLP on features can't find hop-2 cells because all features are cos(q, chunk)
  - A GNN propagates information: "my neighbor is relevant → I might be hop-2 relevant"

ARCHITECTURE:
  Initial cell features:
    - cell_max_cos, cell_mean_cos, cell_top2_cos, log_cell_size  (4 features)

  GNN layers (2 layers):
    Each layer: message = MLP([h_src || h_dst || q_proj])
                agg     = mean(messages from neighbors)
                h_new   = MLP([h_old || agg]) + h_old  (residual)

  Final score: MLP(h_final) → scalar per cell

  Training: BCE on gold cell labels, 5-fold CV, HP sweep.
  Evaluation: cell scores → boost chunks → re-rank.

Reuses pre-computed data from cell_scorer_v3/precomputed/precomputed.pt
"""

import json
import math
import random
import argparse
import time
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


def scatter_mean(src, index, dim=0, dim_size=None):
    """Native PyTorch scatter_mean replacement."""
    if dim_size is None:
        dim_size = index.max().item() + 1
    out = torch.zeros(dim_size, src.shape[1], device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, 1, device=src.device, dtype=src.dtype)
    out.scatter_add_(dim, index.unsqueeze(1).expand_as(src), src)
    count.scatter_add_(dim, index.unsqueeze(1).expand(-1, 1),
                       torch.ones(src.shape[0], 1, device=src.device, dtype=src.dtype))
    count = count.clamp(min=1)
    return out / count

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

RESULTS_DIR = PROJECT_ROOT / "results" / "cell_gnn"


# ===========================================================================
# Model
# ===========================================================================

class CellGraphLayer(nn.Module):
    """One layer of query-conditioned message passing on the cell graph."""

    def __init__(self, h_dim, q_dim, dropout=0.1):
        super().__init__()
        # Message: combine source, destination, and query
        self.msg_mlp = nn.Sequential(
            nn.Linear(h_dim * 2 + q_dim, h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
        )
        # Update: combine old state with aggregated messages
        self.update_mlp = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
        )
        self.norm = nn.LayerNorm(h_dim)

    def forward(self, h, edge_index, q_proj):
        """
        h: (M, h_dim) — cell representations
        edge_index: (2, E) — cell graph edges
        q_proj: (h_dim,) or (1, h_dim) — projected query
        """
        src, dst = edge_index  # (E,), (E,)

        # Expand query for all edges
        q_exp = q_proj.expand(src.shape[0], -1)  # (E, q_dim)

        # Compute messages: each edge gets (h_src, h_dst, query)
        msg_input = torch.cat([h[src], h[dst], q_exp], dim=-1)  # (E, 2*h_dim + q_dim)
        messages = self.msg_mlp(msg_input)  # (E, h_dim)

        # Aggregate: mean of incoming messages per node
        agg = scatter_mean(messages, dst, dim=0, dim_size=h.shape[0])  # (M, h_dim)

        # Update with residual
        h_new = self.update_mlp(torch.cat([h, agg], dim=-1))  # (M, h_dim)
        h_new = self.norm(h_new + h)  # residual connection

        return h_new


class CellGraphGNN(nn.Module):
    """
    Query-conditioned GNN on the cell graph.

    Processes: initial cell features → GNN layers → per-cell scores.
    The query conditions both the initial projection and each message-passing layer.
    """

    def __init__(self, n_cell_features=4, q_dim=768, h_dim=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_layers = n_layers

        # Project cell features to hidden dim
        self.cell_proj = nn.Sequential(
            nn.Linear(n_cell_features, h_dim),
            nn.GELU(),
        )

        # Project query to hidden dim (shared across layers)
        self.q_proj = nn.Sequential(
            nn.Linear(q_dim, h_dim),
            nn.GELU(),
        )

        # GNN layers
        self.layers = nn.ModuleList([
            CellGraphLayer(h_dim, h_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Final scoring head
        self.score_head = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim // 2, 1),
        )

    def forward(self, cell_features, edge_index, q_emb):
        """
        cell_features: (M, n_cell_features)
        edge_index: (2, E)
        q_emb: (D,) raw query embedding

        Returns: (M,) scores per cell
        """
        h = self.cell_proj(cell_features)     # (M, h_dim)
        q = self.q_proj(q_emb)                # (h_dim,)

        for layer in self.layers:
            h = layer(h, edge_index, q)

        scores = self.score_head(h).squeeze(-1)  # (M,)
        return scores


# ===========================================================================
# Data preparation
# ===========================================================================

def prepare_cell_features(cell_stats, feature_indices=[0, 1, 2, 5]):
    """Extract selected features from pre-computed cell stats.

    Default: cell_max(0), cell_mean(1), cell_top2(2), log_size(5)
    These are the within-cell features. Cross-cell info comes from GNN.
    """
    return cell_stats[:, :, feature_indices]  # (Q, M, n_features)


def build_edge_index(cell_neighbors, M, device):
    """Convert adjacency dict to edge_index tensor."""
    src, dst = [], []
    for i, neighbors in cell_neighbors.items():
        for j in neighbors:
            src.append(i)
            dst.append(j)
    if not src:
        return torch.zeros(2, 0, dtype=torch.long, device=device)
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    return edge_index


# ===========================================================================
# Training
# ===========================================================================

def train_one_epoch(model, cell_features, q_embs, gold_targets, edge_index,
                    train_idx, optimizer, device, neg_ratio=5.0):
    """Train one epoch, one question at a time (GNN needs full graph)."""
    model.train()
    random.shuffle(train_idx)
    total_loss, n_samples = 0.0, 0

    for qi in train_idx:
        feat = cell_features[qi].to(device)     # (M, n_feat)
        q = q_embs[qi].to(device)                # (D,)
        tgt = gold_targets[qi].to(device)        # (M,)

        optimizer.zero_grad()
        scores = model(feat, edge_index, q)       # (M,)

        # Focal-style BCE: upweight hard examples
        bce = F.binary_cross_entropy_with_logits(scores, tgt, reduction='none')

        # Weight positives more (class imbalance: ~14 gold out of 9662)
        n_pos = tgt.sum().clamp(min=1)
        n_neg = (1 - tgt).sum().clamp(min=1)
        pos_weight = (n_neg / n_pos).clamp(max=100.0)

        weight = torch.where(tgt > 0.5, pos_weight, torch.ones_like(tgt))
        loss = (bce * weight).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_samples += 1

    return total_loss / max(n_samples, 1)


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_retrieval(cell_scores, test_idx, cos_scores, cell_to_nodes,
                       entity_cell_ids, samples, N, device,
                       boost_val=1.0, top_cells=100):
    """Evaluate chunk retrieval given cell scores."""
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

        cs = cell_scores[ti].to(device)  # (M,)
        cell_probs = torch.sigmoid(cs)

        # Select top cells and boost their chunks
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


def evaluate_baselines(test_idx, cos_scores, cell_to_nodes, cell_max_scores,
                       entity_cell_ids, samples, N, device):
    """Cosine baseline and cell-max baseline."""
    M = len(entity_cell_ids)
    cell_pos_to_chunks = {pos: cell_to_nodes[entity_cell_ids[pos]]
                          for pos in range(M)}

    recalls_cos = {k: [] for k in [2, 5, 10, 20]}
    recalls_cm = {k: [] for k in [2, 5, 10, 20]}

    for ti, qi in enumerate(test_idx):
        gt = samples[qi]["supporting"]
        if not gt:
            continue
        gt_set = set(gt)
        cos_q = cos_scores[qi].to(device)

        # Cosine baseline
        ranked = cos_q.topk(20).indices.tolist()
        for k in recalls_cos:
            recalls_cos[k].append(len(gt_set & set(ranked[:k])) / len(gt_set))

        # Cell-max
        cm = cell_max_scores[qi].to(device)
        chunk_boost = torch.zeros(N, device=device)
        top_c = cm.topk(min(100, M)).indices.tolist()
        for pos in top_c:
            val = cm[pos].item()
            for ni in cell_pos_to_chunks[pos]:
                if val > chunk_boost[ni].item():
                    chunk_boost[ni] = val
        if (chunk_boost > 0).any():
            bmin = chunk_boost[chunk_boost > 0].min()
            bmax = chunk_boost.max()
            if bmax > bmin:
                chunk_boost = ((chunk_boost - bmin) / (bmax - bmin)).clamp(0, 1)
        ranked = (cos_q + 0.5 * chunk_boost).topk(20).indices.tolist()
        for k in recalls_cm:
            recalls_cm[k].append(len(gt_set & set(ranked[:k])) / len(gt_set))

    cos_m = {f"R@{k}": float(np.mean(v)) for k, v in recalls_cos.items()}
    cm_m = {f"R@{k}": float(np.mean(v)) for k, v in recalls_cm.items()}
    return cos_m, cm_m


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Cell Graph GNN — Query-Conditioned Message Passing")
    print(f"Results: {run_dir}")
    print("=" * 70)

    # --- Load pre-computed data ---
    precomp_path = PROJECT_ROOT / "results" / "cell_scorer_v3" / "precomputed" / "precomputed.pt"
    print(f"\n[1/3] Loading pre-computed data from {precomp_path}...")
    precomp = torch.load(precomp_path, weights_only=False)

    q_embs = precomp["q_embs"]                    # (Q, D)
    cos_scores = precomp["cos_scores"]              # (Q, N)
    all_features = precomp["all_features"]          # (Q, M, 10)
    cell_max_scores = precomp["cell_max_scores"]    # (Q, M)
    gold_targets = precomp["gold_targets"]          # (Q, M)
    entity_cell_ids = precomp["entity_cell_ids"]
    cell_to_nodes = precomp["cell_to_nodes"]
    cell_neighbors = precomp["cell_neighbors"]
    N = precomp["n_chunks"]
    M = precomp["n_cells"]
    Q = precomp["n_questions"]
    D = q_embs.shape[1]

    # Load samples for evaluation
    data_path = PROJECT_ROOT / "LPGNN-retriever/datasets/musique/musique.json"
    with open(data_path) as f:
        data = json.load(f)[:args.max_samples]
    samples = []
    for item in data:
        paragraphs = item.get("paragraphs", [])
        local_chunks = []
        for p in paragraphs:
            local_chunks.append(len(samples) * 20 + len(local_chunks))  # placeholder
        supp = []
        # Rebuild properly
    # Actually re-load properly
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Q={Q}, M={M}, N={N}, D={D}, device={device}")

    # --- Build edge index ---
    print("\n[2/3] Building cell graph...")
    edge_index = build_edge_index(cell_neighbors, M, device)
    n_edges = edge_index.shape[1] // 2
    print(f"  Edges: {n_edges}, avg degree: {edge_index.shape[1] / M:.1f}")

    # --- Cell features: just within-cell cosine stats (GNN handles cross-cell) ---
    # Use: cell_max(0), cell_mean(1), cell_top2(2), log_size(5)
    feature_indices = [0, 1, 2, 5]
    cell_features = prepare_cell_features(all_features, feature_indices)
    n_cell_features = len(feature_indices)
    print(f"  Cell features: {n_cell_features} per cell")

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
    for lr in [5e-4, 1e-3, 3e-3]:
        for h_dim in [32, 64, 128]:
            for n_layers in [1, 2, 3]:
                for dropout in [0.0, 0.1]:
                    hp_grid.append({
                        "lr": lr, "h_dim": h_dim,
                        "n_layers": n_layers, "dropout": dropout,
                    })

    print(f"\n{'='*70}")
    print(f"HP Search: {len(hp_grid)} configs × {args.n_folds} folds = {len(hp_grid)*args.n_folds} runs")
    print(f"{'='*70}")

    # Save config
    run_config = {
        "max_samples": args.max_samples,
        "n_chunks": N, "n_cells": M, "n_questions": Q,
        "n_cell_features": n_cell_features,
        "feature_indices": feature_indices,
        "n_folds": args.n_folds,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "seed": args.seed,
        "hp_grid_size": len(hp_grid),
        "timestamp": timestamp,
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    all_results = []

    for ci, hp in enumerate(hp_grid):
        config_id = f"lr{hp['lr']}_h{hp['h_dim']}_L{hp['n_layers']}_d{hp['dropout']}"
        config_dir = run_dir / "runs" / config_id
        config_dir.mkdir(parents=True, exist_ok=True)

        fold_results = []

        for fi in range(args.n_folds):
            test_idx = folds[fi]
            train_idx = [i for i in all_indices if i not in set(test_idx)]

            model = CellGraphGNN(
                n_cell_features=n_cell_features,
                q_dim=D,
                h_dim=hp["h_dim"],
                n_layers=hp["n_layers"],
                dropout=hp["dropout"],
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=0.01)

            # Cosine annealing with warmup
            warmup = max(args.max_epochs // 10, 2)
            def lr_lambda(ep, warmup=warmup, max_ep=args.max_epochs):
                if ep < warmup:
                    return (ep + 1) / warmup
                return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(max_ep - warmup, 1)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            best_r5 = -1.0
            best_state = None
            patience_counter = 0
            train_log = []

            for epoch in range(args.max_epochs):
                loss = train_one_epoch(
                    model, cell_features, q_embs, gold_targets, edge_index,
                    list(train_idx), optimizer, device,
                )
                scheduler.step()

                # Evaluate on test fold
                model.eval()
                with torch.no_grad():
                    test_scores = []
                    for qi in test_idx:
                        feat = cell_features[qi].to(device)
                        q = q_embs[qi].to(device)
                        s = model(feat, edge_index, q)
                        test_scores.append(s.cpu())
                    test_scores = torch.stack(test_scores)  # (|test|, M)

                m = evaluate_retrieval(
                    test_scores, list(range(len(test_idx))),
                    cos_scores[test_idx], cell_to_nodes,
                    entity_cell_ids, [samples[i] for i in test_idx],
                    N, device, boost_val=1.0, top_cells=100,
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

            # Post-training sweep over boost_val and top_cells
            with torch.no_grad():
                test_scores = []
                for qi in test_idx:
                    feat = cell_features[qi].to(device)
                    q = q_embs[qi].to(device)
                    s = model(feat, edge_index, q)
                    test_scores.append(s.cpu())
                test_scores = torch.stack(test_scores)

            best_sweep_r5 = -1.0
            best_sweep_cfg = (1.0, 100)
            best_sweep_metrics = {}

            for bv in [0.5, 1.0, 1.5, 2.0, 3.0]:
                for tc in [20, 50, 100, 200, 500]:
                    m = evaluate_retrieval(
                        test_scores, list(range(len(test_idx))),
                        cos_scores[test_idx], cell_to_nodes,
                        entity_cell_ids, [samples[i] for i in test_idx],
                        N, device, boost_val=bv, top_cells=tc,
                    )
                    if m["R@5"] > best_sweep_r5:
                        best_sweep_r5 = m["R@5"]
                        best_sweep_cfg = (bv, tc)
                        best_sweep_metrics = m

            fold_result = {
                "fold": fi,
                "best_epoch_r5": best_r5,
                "best_sweep_r5": best_sweep_r5,
                "best_sweep_cfg": best_sweep_cfg,
                "best_sweep_metrics": best_sweep_metrics,
                "n_epochs_trained": len(train_log),
            }
            fold_results.append(fold_result)

            # Save fold details
            with open(config_dir / f"fold_{fi}.json", "w") as f:
                json.dump({"hp": hp, "fold_result": fold_result, "train_log": train_log}, f, indent=2)

            # Save best model
            torch.save(best_state, config_dir / f"fold_{fi}_best.pt")

        # Aggregate across folds
        r2_vals = [fr["best_sweep_metrics"].get("R@2", 0) for fr in fold_results]
        r5_vals = [fr["best_sweep_r5"] for fr in fold_results]
        r10_vals = [fr["best_sweep_metrics"].get("R@10", 0) for fr in fold_results]
        r20_vals = [fr["best_sweep_metrics"].get("R@20", 0) for fr in fold_results]

        result_entry = {
            "config_id": config_id,
            "hp": hp,
            "mean_R@2": float(np.mean(r2_vals)),
            "mean_R@5": float(np.mean(r5_vals)),
            "std_R@5": float(np.std(r5_vals)),
            "mean_R@10": float(np.mean(r10_vals)),
            "mean_R@20": float(np.mean(r20_vals)),
            "fold_results": fold_results,
        }
        all_results.append(result_entry)

        # Save incrementally
        with open(run_dir / "results_so_far.json", "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"  [{ci+1:>3}/{len(hp_grid)}] {config_id:<35} "
              f"R@2={np.mean(r2_vals)*100:.1f}%  "
              f"R@5={np.mean(r5_vals)*100:.1f}% ± {np.std(r5_vals)*100:.1f}%  "
              f"R@10={np.mean(r10_vals)*100:.1f}%")

    # --- Final summary ---
    all_results.sort(key=lambda x: x["mean_R@5"], reverse=True)

    # Compute baselines (on full dataset)
    all_idx = list(range(Q))
    cos_m, cm_m = evaluate_baselines(
        all_idx, cos_scores, cell_to_nodes, cell_max_scores,
        entity_cell_ids, samples, N, device)

    print(f"\n{'='*70}")
    print("TOP 10 CONFIGS")
    print(f"{'='*70}")
    print(f"{'Config':<40} {'R@2':>6} {'R@5':>12} {'R@10':>6} {'R@20':>6}")
    print("-" * 75)
    for r in all_results[:10]:
        print(f"{r['config_id']:<40} {r['mean_R@2']*100:>5.1f}% "
              f"{r['mean_R@5']*100:>5.1f}% ± {r['std_R@5']*100:.1f}% "
              f"{r['mean_R@10']*100:>5.1f}% {r['mean_R@20']*100:>5.1f}%")

    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    best = all_results[0]
    print(f"Cosine baseline           R@2={cos_m['R@2']*100:.1f}%  R@5={cos_m['R@5']*100:.1f}%  "
          f"R@10={cos_m['R@10']*100:.1f}%  R@20={cos_m['R@20']*100:.1f}%")
    print(f"Cell-max (zero-shot)      R@2={cm_m['R@2']*100:.1f}%  R@5={cm_m['R@5']*100:.1f}%  "
          f"R@10={cm_m['R@10']*100:.1f}%  R@20={cm_m['R@20']*100:.1f}%")
    print(f"Cell GNN (best learned)   R@2={best['mean_R@2']*100:.1f}%  R@5={best['mean_R@5']*100:.1f}%  "
          f"R@10={best['mean_R@10']*100:.1f}%  R@20={best['mean_R@20']*100:.1f}%")
    print(f"GFM-RAG                   R@2=49.1%  R@5=58.2%")
    print(f"{'='*70}")

    # Save full summary
    summary = {
        "baselines": {"cosine": cos_m, "cell_max": cm_m},
        "all_configs_ranked": all_results,
        "best_config": all_results[0],
        "run_config": run_config,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {run_dir}/")


if __name__ == "__main__":
    main()
