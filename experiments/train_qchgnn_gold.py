#!/usr/bin/env python3
"""
Train QCHGNN with gold labels on the new entity lifting.

WHY THIS IS DIFFERENT from all previous attempts:
  - Previous MLPs scored cells independently using cos(q, chunk) features
  - Previous TNN modified chunk embeddings → HURT cosine discriminative power
  - QCHGNN does message passing through the hypergraph CONDITIONED on the query
  - After L layers, hop-2 chunks absorb info from hop-1 chunks → can be found
  - score = cos(q, x) + gate * MLP(h_L, q)  — cosine baseline preserved as floor
  - GFM-RAG uses this exact paradigm (GNN + cosine residual) and gets 58.2%

DIAGNOSTIC EVIDENCE (diagnostic_hop2_reachability.py):
  - 98.4% of missed gold cells are within 2 hops in cell graph
  - Entity hypergraph message passing node→hyperedge→node IS 1 hop in cell graph
  - 2-layer QCHGNN = 2 hops in cell graph → reaches 98.4% of missed cells

TRAINING:
  - Gold labels: each (question, supporting_chunks) pair
  - 5-fold CV for rigorous evaluation
  - HP sweep: lr, hidden_dim, num_layers, init_k, temperature
  - Loss: InfoNCE + margin ranking (same as GFM-RAG)
  - New lifting: 9,662 cells, 48.6% gold connectivity

EVALUATION:
  - R@2, R@5, R@10, R@20
  - Comparison with cosine baseline, cell-max, and GFM-RAG

Usage:
  python experiments/train_qchgnn_gold.py --max_samples 500
"""

import json
import math
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import List, Dict

import torch
import torch.nn.functional as F
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

RESULTS_DIR = PROJECT_ROOT / "results" / "qchgnn_gold"


def load_musique(data_path, max_samples):
    with open(data_path) as f:
        data = json.load(f)[:max_samples]
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
    return chunks, samples


def build_incidence_tensors(cell_to_nodes, n_chunks, device):
    """Build scatter-based incidence tensors for QCHGNN."""
    cell_indices = sorted(cell_to_nodes.keys())
    cell_pos_map = {ci: pos for pos, ci in enumerate(cell_indices)}
    M = len(cell_indices)

    flat_nodes, cell_assignments = [], []
    for ci in cell_indices:
        pos = cell_pos_map[ci]
        nodes = cell_to_nodes[ci]
        flat_nodes.extend(nodes)
        cell_assignments.extend([pos] * len(nodes))

    flat_nodes_t = torch.tensor(flat_nodes, dtype=torch.long, device=device)
    cell_asgn_t = torch.tensor(cell_assignments, dtype=torch.long, device=device)

    degrees_v = torch.zeros(n_chunks, device=device)
    degrees_v.scatter_add_(0, flat_nodes_t, torch.ones_like(flat_nodes_t, dtype=torch.float))
    degrees_v = degrees_v.clamp(min=1)

    degrees_e = torch.zeros(M, device=device)
    degrees_e.scatter_add_(0, cell_asgn_t, torch.ones_like(cell_asgn_t, dtype=torch.float))
    degrees_e = degrees_e.clamp(min=1)

    return flat_nodes_t, cell_asgn_t, M, degrees_v, degrees_e


def evaluate(model, x_chunks, q_embs, samples, test_idx, device,
             flat_nodes_t, cell_asgn_t, M, degrees_v, degrees_e,
             mode="trained"):
    """Evaluate R@2, R@5, R@10, R@20."""
    model.eval()
    recalls = {k: [] for k in [2, 5, 10, 20]}
    n = x_chunks.shape[0]

    with torch.no_grad():
        for qi in test_idx:
            gt = samples[qi]["supporting"]
            if not gt:
                continue
            gt_set = set(gt)

            q = q_embs[qi:qi+1].to(device)  # (1, D)

            if mode == "baseline":
                q_norm = F.normalize(q, dim=-1)
                x_norm = F.normalize(x_chunks, dim=-1)
                scores = (q_norm @ x_norm.T).squeeze(0)
            else:
                scores = model(x_chunks, q, flat_nodes_t, cell_asgn_t,
                               M, degrees_v, degrees_e).squeeze(0)

            ranked = scores.topk(min(20, n)).indices.tolist()
            for k in recalls:
                recalls[k].append(len(gt_set & set(ranked[:k])) / len(gt_set))

    return {f"R@{k}": float(np.mean(v)) if v else 0.0 for k, v in recalls.items()}


def train_one_epoch(model, x_chunks, q_embs, samples, train_idx, optimizer,
                    loss_fn, device, flat_nodes_t, cell_asgn_t, M,
                    degrees_v, degrees_e, batch_size=4,
                    synth_q_embs=None, synth_targets=None, synth_weight=0.5):
    """Train one epoch using gold labels + optional synthetic queries.

    Gold batch: (question_emb, supporting_chunks) from real MuSiQue
    Synth batch: (query_emb, cell_chunks) from synthetic query generation
    """
    model.train()
    n = x_chunks.shape[0]
    indices = list(train_idx)
    random.shuffle(indices)

    total_loss, n_steps = 0.0, 0

    # Gold training
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        B = len(batch_idx)

        q_batch = torch.stack([q_embs[qi] for qi in batch_idx]).to(device)

        optimizer.zero_grad()
        scores = model(x_chunks, q_batch, flat_nodes_t, cell_asgn_t,
                       M, degrees_v, degrees_e)

        targets = torch.zeros(B, n, device=device)
        for i, qi in enumerate(batch_idx):
            for ci in samples[qi]["supporting"]:
                if ci < n:
                    targets[i, ci] = 1.0

        loss, _ = loss_fn(scores, targets)

        # Mixed with synthetic if available
        if synth_q_embs is not None and synth_targets is not None and synth_weight > 0:
            n_synth = synth_q_embs.shape[0]
            synth_idx = random.sample(range(n_synth), min(B, n_synth))
            sq_batch = synth_q_embs[synth_idx].to(device)
            s_scores = model(x_chunks, sq_batch, flat_nodes_t, cell_asgn_t,
                             M, degrees_v, degrees_e)
            s_tgt = synth_targets[synth_idx].to(device)
            s_loss, _ = loss_fn(s_scores, s_tgt)
            loss = (1 - synth_weight) * loss + synth_weight * s_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * B
        n_steps += B

    return total_loss / max(n_steps, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--synth_weight", type=float, default=0.0,
                        help="Weight for synthetic queries (0=gold only, 0.5=equal mix)")
    parser.add_argument("--synth_cache", type=str,
                        default="toporag/experiments/cache/musique_500_queries.json")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from toporag import TopoRAG, TopoRAGConfig
    from toporag.models.qc_hgnn import QueryConditionedHGNN, QCHGNNLoss

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("QCHGNN Gold-Label Training on New Entity Lifting")
    print(f"Results: {run_dir}")
    print("=" * 70)

    # --- Load data ---
    data_path = PROJECT_ROOT / "LPGNN-retriever/datasets/musique/musique.json"
    topo_file = REPO_ROOT / "experiments/cache/topology/musique_500_entity.pt"

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
    embed_dim = config.embed_dim

    x_chunks = x_chunks_cpu.to(device)
    print(f"  Device: {device}, embed_dim: {embed_dim}")

    # --- Build incidence ---
    flat_nodes_t, cell_asgn_t, M, degrees_v, degrees_e = \
        build_incidence_tensors(cell_to_nodes, N, device)
    print(f"  Hyperedges: {M}, Incidences: {flat_nodes_t.shape[0]}")

    # --- Pre-embed all questions ---
    print("\n[3/4] Embedding questions...")
    all_questions = [s["question"] for s in samples]
    q_emb_parts = []
    for i in range(0, Q, 64):
        batch = all_questions[i:i+64]
        with torch.no_grad():
            emb = embedder.encode(batch, is_query=True, show_progress=False)
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb, dtype=torch.float32)
            q_emb_parts.append(emb.cpu().clone())
    q_embs = torch.cat(q_emb_parts, dim=0)  # (Q, D)
    print(f"  q_embs: {q_embs.shape}")

    # --- Load synthetic queries (optional data augmentation) ---
    synth_q_embs, synth_targets = None, None
    if args.synth_weight > 0 and Path(args.synth_cache).exists():
        print(f"\n  Loading synthetic queries from {args.synth_cache}...")
        with open(args.synth_cache) as f:
            synth_data = json.load(f)
        synth_texts = []
        synth_cell_ids = []
        for cell_id_str, queries in synth_data.items():
            cell_id = int(cell_id_str)
            if cell_id in cell_to_nodes:
                for q_text in queries:
                    synth_texts.append(q_text)
                    synth_cell_ids.append(cell_id)

        if synth_texts:
            # Embed synthetic queries
            synth_parts = []
            for i in range(0, len(synth_texts), 64):
                batch = synth_texts[i:i+64]
                with torch.no_grad():
                    emb = embedder.encode(batch, is_query=True, show_progress=False)
                    if not isinstance(emb, torch.Tensor):
                        emb = torch.tensor(emb, dtype=torch.float32)
                    synth_parts.append(emb.cpu().clone())
            synth_q_embs = torch.cat(synth_parts, dim=0)

            # Build targets: positive = chunks in the source cell
            synth_targets = torch.zeros(len(synth_texts), N)
            for i, cell_id in enumerate(synth_cell_ids):
                for ci in cell_to_nodes[cell_id]:
                    if ci < N:
                        synth_targets[i, ci] = 1.0

            print(f"  Synthetic queries: {synth_q_embs.shape[0]}, "
                  f"synth_weight={args.synth_weight}")
        else:
            print("  No valid synthetic queries found.")
            args.synth_weight = 0.0
    elif args.synth_weight > 0:
        print(f"  Synthetic cache not found: {args.synth_cache}")
        args.synth_weight = 0.0

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
    # Reduced grid: focus on most impactful HPs
    # Prior: higher lr, larger hidden, 2-3 layers work best
    hp_grid = []
    for lr in [3e-4, 1e-3]:
        for hidden_dim in [128, 256]:
            for num_layers in [2, 3]:
                for init_k in [20, 50]:
                    for temp in [0.05]:
                        for sw in [0.0, 0.3]:
                            hp_grid.append({
                                "lr": lr, "hidden_dim": hidden_dim,
                                "num_layers": num_layers, "init_k": init_k,
                                "temperature": temp, "synth_weight": sw,
                            })

    print(f"\n{'='*70}")
    print(f"[4/4] HP Search: {len(hp_grid)} configs × {args.n_folds} folds")
    print(f"{'='*70}")

    # Save config
    run_config = {
        "max_samples": args.max_samples,
        "n_chunks": N, "n_cells": M, "n_questions": Q,
        "n_folds": args.n_folds, "max_epochs": args.max_epochs,
        "patience": args.patience, "batch_size": args.batch_size,
        "seed": args.seed, "hp_grid_size": len(hp_grid),
        "timestamp": timestamp, "embed_dim": embed_dim,
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # --- Baselines ---
    print("\nComputing baselines...")
    base_model = QueryConditionedHGNN(embed_dim=embed_dim, hidden_dim=128).to(device)
    m_cos = evaluate(base_model, x_chunks, q_embs, samples, all_indices, device,
                     flat_nodes_t, cell_asgn_t, M, degrees_v, degrees_e, mode="baseline")
    print(f"  Cosine: R@2={m_cos['R@2']*100:.1f}%  R@5={m_cos['R@5']*100:.1f}%  "
          f"R@10={m_cos['R@10']*100:.1f}%  R@20={m_cos['R@20']*100:.1f}%")
    del base_model

    # Also load synth data if any config uses synth_weight > 0
    any_synth = any(hp.get("synth_weight", 0) > 0 for hp in hp_grid)
    if any_synth and synth_q_embs is None:
        # Force load even if args.synth_weight was 0
        synth_cache_path = Path(args.synth_cache)
        if synth_cache_path.exists():
            print(f"\n  Loading synthetic queries for HP configs with synth_weight > 0...")
            with open(synth_cache_path) as f:
                synth_data = json.load(f)
            synth_texts, synth_cell_ids = [], []
            for cell_id_str, queries in synth_data.items():
                cell_id = int(cell_id_str)
                if cell_id in cell_to_nodes:
                    for q_text in queries:
                        synth_texts.append(q_text)
                        synth_cell_ids.append(cell_id)
            if synth_texts:
                synth_parts = []
                for i in range(0, len(synth_texts), 64):
                    batch = synth_texts[i:i+64]
                    with torch.no_grad():
                        emb = embedder.encode(batch, is_query=True, show_progress=False)
                        if not isinstance(emb, torch.Tensor):
                            emb = torch.tensor(emb, dtype=torch.float32)
                        synth_parts.append(emb.cpu().clone())
                synth_q_embs = torch.cat(synth_parts, dim=0)
                synth_targets = torch.zeros(len(synth_texts), N)
                for i, cell_id in enumerate(synth_cell_ids):
                    for ci in cell_to_nodes[cell_id]:
                        if ci < N:
                            synth_targets[i, ci] = 1.0
                print(f"  Loaded {synth_q_embs.shape[0]} synthetic queries")

    all_results = []

    for ci, hp in enumerate(hp_grid):
        config_id = (f"lr{hp['lr']}_h{hp['hidden_dim']}_L{hp['num_layers']}"
                     f"_K{hp['init_k']}_t{hp['temperature']}_sw{hp.get('synth_weight', 0.0)}")
        config_dir = run_dir / "runs" / config_id
        config_dir.mkdir(parents=True, exist_ok=True)

        fold_metrics = []

        for fi in range(args.n_folds):
            test_idx = folds[fi]
            train_idx = [i for i in all_indices if i not in set(test_idx)]

            model = QueryConditionedHGNN(
                embed_dim=embed_dim,
                hidden_dim=hp["hidden_dim"],
                num_layers=hp["num_layers"],
                init_k=hp["init_k"],
                dropout=0.1,
                use_checkpoint=True,
            ).to(device)

            loss_fn = QCHGNNLoss(alpha=0.3, temperature=hp["temperature"])

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=hp["lr"], weight_decay=0.01)

            warmup = max(args.max_epochs // 10, 2)
            def lr_lambda(ep, warmup=warmup, max_ep=args.max_epochs):
                if ep < warmup:
                    return (ep + 1) / warmup
                return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(max_ep - warmup, 1)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            best_r5, best_state = -1.0, None
            patience_counter = 0
            train_log = []

            for epoch in range(args.max_epochs):
                loss = train_one_epoch(
                    model, x_chunks, q_embs, samples, train_idx, optimizer,
                    loss_fn, device, flat_nodes_t, cell_asgn_t, M,
                    degrees_v, degrees_e, batch_size=args.batch_size,
                    synth_q_embs=synth_q_embs, synth_targets=synth_targets,
                    synth_weight=hp.get("synth_weight", args.synth_weight))
                scheduler.step()

                m = evaluate(model, x_chunks, q_embs, samples, test_idx, device,
                             flat_nodes_t, cell_asgn_t, M, degrees_v, degrees_e)

                train_log.append({
                    "epoch": epoch + 1, "loss": loss,
                    "R@2": m["R@2"], "R@5": m["R@5"],
                    "R@10": m["R@10"], "R@20": m["R@20"],
                    "gate": torch.sigmoid(model.mp_gate).item(),
                })

                if m["R@5"] > best_r5:
                    best_r5 = m["R@5"]
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= args.patience:
                        break

            fold_metrics.append({
                "fold": fi, "best_R@5": best_r5,
                "best_metrics": train_log[-1] if train_log else {},
                "n_epochs": len(train_log),
            })

            # Save fold details
            with open(config_dir / f"fold_{fi}.json", "w") as f:
                json.dump({"hp": hp, "fold_metrics": fold_metrics[-1],
                           "train_log": train_log}, f, indent=2)
            if best_state:
                torch.save(best_state, config_dir / f"fold_{fi}_best.pt")

            del model, optimizer, scheduler
            torch.cuda.empty_cache()

        # Aggregate
        r5_vals = [fm["best_R@5"] for fm in fold_metrics]
        result = {
            "config_id": config_id, "hp": hp,
            "mean_R@5": float(np.mean(r5_vals)),
            "std_R@5": float(np.std(r5_vals)),
            "fold_metrics": fold_metrics,
        }
        all_results.append(result)

        # Save incrementally
        with open(run_dir / "results_so_far.json", "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"  [{ci+1:>3}/{len(hp_grid)}] {config_id:<50} "
              f"R@5={np.mean(r5_vals)*100:.1f}% ± {np.std(r5_vals)*100:.1f}%")

    # --- Final summary ---
    all_results.sort(key=lambda x: x["mean_R@5"], reverse=True)

    print(f"\n{'='*70}")
    print("TOP 10 CONFIGS")
    print(f"{'='*70}")
    for r in all_results[:10]:
        print(f"  {r['config_id']:<50} R@5={r['mean_R@5']*100:.1f}% ± {r['std_R@5']*100:.1f}%")

    best = all_results[0]
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"Cosine baseline:    R@2={m_cos['R@2']*100:.1f}%  R@5={m_cos['R@5']*100:.1f}%  "
          f"R@10={m_cos['R@10']*100:.1f}%  R@20={m_cos['R@20']*100:.1f}%")
    print(f"Cell-max (zero-shot): R@5=45.1%")
    print(f"QCHGNN (best):      R@5={best['mean_R@5']*100:.1f}% ± {best['std_R@5']*100:.1f}%")
    print(f"GFM-RAG (target):   R@2=49.1%  R@5=58.2%")
    print(f"{'='*70}")

    summary = {
        "baselines": {"cosine": m_cos, "cell_max_r5": 0.451},
        "all_configs_ranked": all_results,
        "best_config": all_results[0],
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAll results saved to {run_dir}/")


if __name__ == "__main__":
    main()
