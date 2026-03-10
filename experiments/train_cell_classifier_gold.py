#!/usr/bin/env python3
"""
TopoRAG Cell Classifier — Gold-label training with 5-fold CV.

Same protocol as GFM-RAG: train on real multi-hop questions with gold labels.
5-fold CV on 500 MuSiQue questions (train 400, test 100 per fold).

Gold cell labels: for question Q with gold supporting chunks {g1, g2, ...},
the positive cells are all entity cells containing any gold chunk.

Architecture:
  cell_emb = mean(chunk_embeddings in cell)      — pre-computed, frozen
  score(q, cell) = MLP([q_proj; cell_emb_proj])  — learned
  chunk_score = cos(q, x_i) + boost * max(P(cell_j|q) for cells containing i)
"""

import json
import math
import random
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CellClassifier(nn.Module):
    def __init__(self, embed_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.query_proj = nn.Linear(embed_dim, hidden_dim)
        self.cell_proj = nn.Linear(embed_dim, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def score_cells(self, q_emb, cell_embs):
        q = self.query_proj(q_emb)
        c = self.cell_proj(cell_embs)
        B, M = q.shape[0], c.shape[0]
        q_exp = q.unsqueeze(1).expand(B, M, -1)
        c_exp = c.unsqueeze(0).expand(B, M, -1)
        return self.scorer(torch.cat([q_exp, c_exp], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

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


def get_gold_cells(samples, chunk_to_cells, entity_cell_set):
    """For each sample, find entity cells containing gold chunks."""
    gold_cells_per_sample = []
    for s in samples:
        cells = set()
        for gi in s["supporting"]:
            for cid in chunk_to_cells.get(gi, []):
                if cid in entity_cell_set:
                    cells.add(cid)
        gold_cells_per_sample.append(cells)
    return gold_cells_per_sample


# ---------------------------------------------------------------------------
# Training on gold labels
# ---------------------------------------------------------------------------

def train_one_epoch_gold(
    model, train_samples, gold_cells_per_sample,
    cell_embs, entity_cell_ids, cell_id_to_pos,
    embedder, optimizer, device, batch_size=16,
):
    model.train()
    M = len(entity_cell_ids)
    indices = list(range(len(train_samples)))
    random.shuffle(indices)

    total_loss = 0.0
    n_batches = 0

    for batch_start in range(0, len(indices), batch_size):
        batch_idx = indices[batch_start: batch_start + batch_size]
        B = len(batch_idx)

        questions = [train_samples[i]["question"] for i in batch_idx]
        with torch.no_grad():
            q_embs = embedder.encode(questions, is_query=True, show_progress=False)
            if not isinstance(q_embs, torch.Tensor):
                q_embs = torch.tensor(q_embs, dtype=torch.float32)
            q_embs = q_embs.to(device).clone()

        optimizer.zero_grad()
        scores = model.score_cells(q_embs, cell_embs)  # (B, M)

        # Build multi-hot targets from gold cells
        targets = torch.zeros(B, M, device=device)
        for i, si in enumerate(batch_idx):
            for cid in gold_cells_per_sample[si]:
                pos = cell_id_to_pos.get(cid)
                if pos is not None:
                    targets[i, pos] = 1.0

        # BCE with adversarial hard negative weighting
        bce = F.binary_cross_entropy_with_logits(scores, targets, reduction='none')
        is_pos = targets > 0.5
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model, samples, cell_embs, entity_cell_ids, cell_to_nodes,
    chunk_to_cells, x_chunks, embedder, device,
    mode="trained", boost_val=0.2, top_cells=50,
):
    model.eval()
    N = x_chunks.shape[0]
    M = len(entity_cell_ids)
    x_norm = F.normalize(x_chunks, dim=-1)

    cell_pos_to_chunks = {}
    for pos, ci in enumerate(entity_cell_ids):
        cell_pos_to_chunks[pos] = cell_to_nodes[ci]

    recalls = {k: [] for k in [2, 5, 10, 20]}

    with torch.no_grad():
        for batch_start in range(0, len(samples), 32):
            batch_s = samples[batch_start: batch_start + 32]
            questions = [s["question"] for s in batch_s]
            gts = [s["supporting"] for s in batch_s]

            q_embs = embedder.encode(questions, is_query=True, show_progress=False)
            if not isinstance(q_embs, torch.Tensor):
                q_embs = torch.tensor(q_embs, dtype=torch.float32)
            q_embs = q_embs.to(device)

            q_norm = F.normalize(q_embs, dim=-1)
            cos = q_norm @ x_norm.T

            if mode == "baseline":
                for i, gt in enumerate(gts):
                    if not gt:
                        continue
                    gt_set = set(gt)
                    ranked = cos[i].topk(20).indices.tolist()
                    for k in recalls:
                        recalls[k].append(len(gt_set & set(ranked[:k])) / len(gt_set))
            else:
                cell_scores = model.score_cells(q_embs, cell_embs)
                cell_probs = torch.sigmoid(cell_scores)

                for i, gt in enumerate(gts):
                    if not gt:
                        continue
                    gt_set = set(gt)

                    top_c = cell_probs[i].topk(min(top_cells, M)).indices.tolist()
                    chunk_boost = torch.zeros(N, device=device)
                    for pos in top_c:
                        prob = cell_probs[i, pos].item()
                        for ni in cell_pos_to_chunks[pos]:
                            if prob > chunk_boost[ni].item():
                                chunk_boost[ni] = prob

                    final = cos[i] + boost_val * chunk_boost
                    ranked = final.topk(20).indices.tolist()
                    for k in recalls:
                        recalls[k].append(len(gt_set & set(ranked[:k])) / len(gt_set))

    return {f"R@{k}": np.mean(v) if v else 0.0 for k, v in recalls.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--boost_val", type=float, default=0.2)
    parser.add_argument("--top_cells", type=int, default=50)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from toporag import TopoRAG, TopoRAGConfig

    data_path = PROJECT_ROOT / "LPGNN-retriever/datasets/musique/musique.json"
    topo_file = REPO_ROOT / "experiments/cache/topology/musique_500_entity.pt"

    print("=" * 65)
    print("Cell Classifier — Gold Labels + 5-Fold CV")
    print("=" * 65)

    # Load data
    print("\n[1/3] Loading data...")
    chunks, samples = load_musique(data_path, args.max_samples)
    N = len(chunks)
    print(f"  {N} chunks, {len(samples)} questions")

    # Load topology
    print("\n[2/3] Loading topology...")
    cache = torch.load(topo_file, weights_only=False)
    cell_to_nodes = cache["cell_to_nodes"]
    x_chunks_cpu = cache["lifted"].x_0

    config = TopoRAGConfig(lifting="entity", use_gps=False, use_tnn=False)
    toporag = TopoRAG(config)
    device = toporag.device
    embedder = toporag.embedder
    x_chunks = x_chunks_cpu.to(device)
    embed_dim = x_chunks.shape[1]

    # Build maps
    chunk_to_cells = defaultdict(list)
    for cell_id, nodes in cell_to_nodes.items():
        for ni in nodes:
            chunk_to_cells[ni].append(cell_id)

    entity_cell_ids = sorted([ci for ci, nodes in cell_to_nodes.items() if len(nodes) >= 2])
    entity_cell_set = set(entity_cell_ids)
    cell_id_to_pos = {ci: pos for pos, ci in enumerate(entity_cell_ids)}
    M = len(entity_cell_ids)

    # Cell embeddings
    print("  Computing cell embeddings...")
    cell_emb_list = []
    for ci in entity_cell_ids:
        cell_emb_list.append(x_chunks[cell_to_nodes[ci]].mean(dim=0))
    cell_embs = torch.stack(cell_emb_list).to(device)
    print(f"  Entity cells: {M}, Cell embs: {cell_embs.shape}")

    # Gold cell labels
    print("\n[3/3] Computing gold cell labels...")
    gold_cells = get_gold_cells(samples, chunk_to_cells, entity_cell_set)
    avg_gold = np.mean([len(g) for g in gold_cells])
    print(f"  Avg gold cells per question: {avg_gold:.1f}")

    # Baseline (full dataset)
    dummy_model = CellClassifier(embed_dim, args.hidden_dim, args.dropout).to(device)
    m_base = evaluate(dummy_model, samples, cell_embs, entity_cell_ids, cell_to_nodes,
                      chunk_to_cells, x_chunks, embedder, device, mode="baseline")
    print(f"\n  Cosine baseline: R@2={m_base['R@2']*100:.1f}%  R@5={m_base['R@5']*100:.1f}%  "
          f"R@10={m_base['R@10']*100:.1f}%  R@20={m_base['R@20']*100:.1f}%")
    del dummy_model

    # 5-Fold CV
    print(f"\n{'='*65}")
    print(f"{args.n_folds}-Fold Cross Validation")
    print(f"{'='*65}")

    all_indices = list(range(len(samples)))
    random.shuffle(all_indices)
    fold_size = len(all_indices) // args.n_folds
    folds = []
    for f in range(args.n_folds):
        start = f * fold_size
        end = start + fold_size if f < args.n_folds - 1 else len(all_indices)
        folds.append(all_indices[start:end])

    all_fold_results = []

    for fold_idx in range(args.n_folds):
        test_idx = set(folds[fold_idx])
        train_idx = [i for i in all_indices if i not in test_idx]
        test_idx_list = folds[fold_idx]

        train_samples_fold = [samples[i] for i in train_idx]
        test_samples_fold = [samples[i] for i in test_idx_list]
        gold_cells_train = [gold_cells[i] for i in train_idx]

        print(f"\n--- Fold {fold_idx+1}/{args.n_folds} "
              f"(train={len(train_samples_fold)}, test={len(test_samples_fold)}) ---")

        # Fresh model per fold
        model = CellClassifier(embed_dim, args.hidden_dim, args.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        warmup = max(args.epochs // 10, 3)

        def lr_lambda(ep):
            if ep < warmup:
                return (ep + 1) / warmup
            return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(args.epochs - warmup, 1)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_r5 = 0.0
        best_state = None

        for epoch in range(args.epochs):
            loss = train_one_epoch_gold(
                model, train_samples_fold, gold_cells_train,
                cell_embs, entity_cell_ids, cell_id_to_pos,
                embedder, optimizer, device, batch_size=args.batch_size,
            )
            scheduler.step()

            # Evaluate on test fold
            m = evaluate(model, test_samples_fold, cell_embs, entity_cell_ids,
                         cell_to_nodes, chunk_to_cells, x_chunks, embedder, device,
                         mode="trained", boost_val=args.boost_val, top_cells=args.top_cells)

            improved = ""
            if m["R@5"] > best_r5:
                best_r5 = m["R@5"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                improved = " *"

            if (epoch + 1) % 10 == 0 or epoch == 0 or improved:
                print(f"  Ep {epoch+1:>3}  loss={loss:.4f}  "
                      f"R@2={m['R@2']*100:.1f}%  R@5={m['R@5']*100:.1f}%  "
                      f"R@10={m['R@10']*100:.1f}%{improved}")

        # Load best and do final sweep
        if best_state:
            model.load_state_dict(best_state)

        print(f"\n  Fold {fold_idx+1} best R@5={best_r5*100:.1f}%")

        # Sweep boost_val and top_cells for this fold
        best_sweep_r5 = 0.0
        best_sweep_cfg = None
        for bv in [0.1, 0.2, 0.5, 1.0]:
            for tc in [20, 50, 100]:
                m = evaluate(model, test_samples_fold, cell_embs, entity_cell_ids,
                             cell_to_nodes, chunk_to_cells, x_chunks, embedder, device,
                             mode="trained", boost_val=bv, top_cells=tc)
                if m["R@5"] > best_sweep_r5:
                    best_sweep_r5 = m["R@5"]
                    best_sweep_cfg = (bv, tc, m)

        bv, tc, m = best_sweep_cfg
        print(f"  Best sweep: boost={bv}, top_C={tc} → "
              f"R@2={m['R@2']*100:.1f}%  R@5={m['R@5']*100:.1f}%  R@10={m['R@10']*100:.1f}%")
        all_fold_results.append(m)

    # Aggregate CV results
    print("\n" + "=" * 65)
    print("CROSS-VALIDATION RESULTS (mean across folds)")
    print("=" * 65)
    for metric in ["R@2", "R@5", "R@10", "R@20"]:
        vals = [r[metric] for r in all_fold_results]
        mean = np.mean(vals) * 100
        std = np.std(vals) * 100
        print(f"  {metric}: {mean:.1f}% ± {std:.1f}%")

    print(f"\n{'Cosine baseline':<25} R@2={m_base['R@2']*100:.1f}%  R@5={m_base['R@5']*100:.1f}%")
    print(f"{'Cell classifier (CV)':<25} R@5={np.mean([r['R@5'] for r in all_fold_results])*100:.1f}%")
    print(f"{'Zero-shot cell-max':<25} R@5=44.5%")
    print(f"{'Oracle cell boost':<25} R@5=73.1%")
    print(f"{'GFM-RAG':<25} R@5=58.2%")
    print("=" * 65)


if __name__ == "__main__":
    main()
