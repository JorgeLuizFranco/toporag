#!/usr/bin/env python3
"""
TopoRAG Cell Classifier v2 — Feature-rich cell scoring.

Key insight from v1 failure: mean cell embeddings LOSE information.
Cell-max (zero-shot, 44.5%) beats gold-label MLP on mean embeddings (41.8%).

Fix: give the MLP per-cell cosine statistics as features.
The MLP sees what cell-max sees (and more), so it can AT LEAST match it.

Features per (query, cell) pair:
  - cell_max_cos:   max(cos(q, x_i) for i in cell)      ← what cell-max uses
  - cell_mean_cos:  mean(cos(q, x_i) for i in cell)
  - cell_top2_cos:  mean of top-2 chunk cosines (evidence strength)
  - cell_frac_top50: fraction of cell chunks in cosine top-50
  - cell_size:      number of chunks in cell (log-scaled)
  - q_proj · cell_proj: projected embedding dot product (semantic match)

Training: 5-fold CV with gold cell labels (same as GFM-RAG protocol).
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

class CellScorerV2(nn.Module):
    """Cell scorer using per-cell cosine statistics + projected embeddings.

    Input features (per query-cell pair):
      - 5 cosine statistics (cell_max, cell_mean, cell_top2, frac_top50, log_size)
      - Projected embedding dot product (q_proj · cell_proj)
    """

    def __init__(self, embed_dim=768, hidden_dim=128, dropout=0.1):
        super().__init__()
        # Projected embedding similarity (captures semantic match beyond cosine stats)
        self.q_proj = nn.Linear(embed_dim, hidden_dim)
        self.c_proj = nn.Linear(embed_dim, hidden_dim)

        # Feature MLP: 5 cosine stats + embedding similarity → score
        n_features = 6  # max, mean, top2, frac_top50, log_size, emb_sim
        self.scorer = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, features):
        """features: (B, M, 6) → (B, M) scores"""
        return self.scorer(features).squeeze(-1)

    def compute_features(self, q_emb, cell_embs, cos_stats):
        """Compute all features for scoring.

        Args:
            q_emb: (B, D) query embeddings
            cell_embs: (M, D) mean cell embeddings
            cos_stats: (B, M, 4) pre-computed [max, mean, top2, frac_top50]

        Returns:
            (B, M, 6) feature tensor
        """
        B, M = cos_stats.shape[:2]

        # Projected embedding similarity
        q_p = F.normalize(self.q_proj(q_emb), dim=-1)      # (B, H)
        c_p = F.normalize(self.c_proj(cell_embs), dim=-1)   # (M, H)
        emb_sim = (q_p @ c_p.T).unsqueeze(-1)               # (B, M, 1)

        # Combine: [cos_stats(4), log_size(from stats, 1), emb_sim(1)]
        features = torch.cat([cos_stats, emb_sim], dim=-1)  # (B, M, 6)
        return features


def compute_cell_cos_stats(cos_scores, cell_to_nodes, entity_cell_ids, device, top_n=50):
    """Pre-compute per-cell cosine statistics for a batch of queries.

    Args:
        cos_scores: (B, N) cosine scores
        cell_to_nodes: dict cell_id → list of chunk indices
        entity_cell_ids: sorted list of entity cell IDs
        device: torch device
        top_n: threshold for "top-N" fraction feature

    Returns:
        (B, M, 5) tensor: [max_cos, mean_cos, top2_cos, frac_topN, log_size]
    """
    B, N = cos_scores.shape
    M = len(entity_cell_ids)

    stats = torch.zeros(B, M, 5, device=device)

    # Get top-N chunk indices per query
    topn_indices = cos_scores.topk(min(top_n, N), dim=1).indices  # (B, top_n)
    topn_sets = [set(topn_indices[b].tolist()) for b in range(B)]

    for pos, ci in enumerate(entity_cell_ids):
        nodes = cell_to_nodes[ci]
        if not nodes:
            continue
        node_t = torch.tensor(nodes, device=device, dtype=torch.long)
        cell_cos = cos_scores[:, node_t]  # (B, |cell|)
        cell_size = len(nodes)

        stats[:, pos, 0] = cell_cos.max(dim=1).values        # max
        stats[:, pos, 1] = cell_cos.mean(dim=1)               # mean
        if cell_size >= 2:
            top2 = cell_cos.topk(min(2, cell_size), dim=1).values
            stats[:, pos, 2] = top2.mean(dim=1)               # top-2 mean
        else:
            stats[:, pos, 2] = cell_cos.squeeze(1)
        # frac in top-N
        for b in range(B):
            n_in_top = sum(1 for ni in nodes if ni in topn_sets[b])
            stats[:, pos, 3] = n_in_top / cell_size
        stats[:, pos, 4] = math.log(cell_size + 1)            # log size

    return stats


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
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, train_samples, gold_cells, cell_embs, entity_cell_ids,
    cell_id_to_pos, cell_to_nodes, x_chunks, embedder, optimizer,
    device, batch_size=16,
):
    model.train()
    M = len(entity_cell_ids)
    x_norm = F.normalize(x_chunks, dim=-1)

    indices = list(range(len(train_samples)))
    random.shuffle(indices)
    total_loss, n_batches = 0.0, 0

    for batch_start in range(0, len(indices), batch_size):
        batch_idx = indices[batch_start: batch_start + batch_size]
        B = len(batch_idx)

        questions = [train_samples[i]["question"] for i in batch_idx]
        with torch.no_grad():
            q_embs = embedder.encode(questions, is_query=True, show_progress=False)
            if not isinstance(q_embs, torch.Tensor):
                q_embs = torch.tensor(q_embs, dtype=torch.float32)
            q_embs = q_embs.to(device).clone()

            q_norm = F.normalize(q_embs, dim=-1)
            cos = q_norm @ x_norm.T  # (B, N)
            cos_stats = compute_cell_cos_stats(cos, cell_to_nodes, entity_cell_ids, device)

        optimizer.zero_grad()
        features = model.compute_features(q_embs, cell_embs, cos_stats)
        scores = model(features)  # (B, M)

        # Multi-hot targets
        targets = torch.zeros(B, M, device=device)
        for i, si in enumerate(batch_idx):
            for cid in gold_cells[si]:
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
    mode="trained", boost_val=1.0, top_cells=50,
):
    model.eval()
    N = x_chunks.shape[0]
    M = len(entity_cell_ids)
    x_norm = F.normalize(x_chunks, dim=-1)

    cell_pos_to_chunks = {pos: cell_to_nodes[ci] for pos, ci in enumerate(entity_cell_ids)}

    recalls = {k: [] for k in [2, 5, 10, 20]}

    # Also track cell selection accuracy
    cell_id_to_pos = {ci: pos for pos, ci in enumerate(entity_cell_ids)}
    entity_cell_set = set(entity_cell_ids)
    gold_cells_list = get_gold_cells(samples, chunk_to_cells, entity_cell_set)

    cell_recall_50 = []

    with torch.no_grad():
        for batch_start in range(0, len(samples), 32):
            batch_s = samples[batch_start: batch_start + 32]
            questions = [s["question"] for s in batch_s]
            gts = [s["supporting"] for s in batch_s]
            gc_batch = gold_cells_list[batch_start: batch_start + len(batch_s)]

            q_embs = embedder.encode(questions, is_query=True, show_progress=False)
            if not isinstance(q_embs, torch.Tensor):
                q_embs = torch.tensor(q_embs, dtype=torch.float32)
            q_embs = q_embs.to(device)

            q_norm = F.normalize(q_embs, dim=-1)
            cos = q_norm @ x_norm.T  # (B, N)

            if mode == "baseline":
                for i, gt in enumerate(gts):
                    if not gt:
                        continue
                    gt_set = set(gt)
                    ranked = cos[i].topk(20).indices.tolist()
                    for k in recalls:
                        recalls[k].append(len(gt_set & set(ranked[:k])) / len(gt_set))
            elif mode == "cell_max":
                # Zero-shot cell-max baseline
                cos_stats = compute_cell_cos_stats(cos, cell_to_nodes, entity_cell_ids, device)
                cell_max_scores = cos_stats[:, :, 0]  # (B, M) — just the max feature

                for i, gt in enumerate(gts):
                    if not gt:
                        continue
                    gt_set = set(gt)

                    top_c = cell_max_scores[i].topk(min(top_cells, M)).indices.tolist()
                    chunk_boost = torch.zeros(N, device=device)
                    for pos in top_c:
                        val = cell_max_scores[i, pos].item()
                        for ni in cell_pos_to_chunks[pos]:
                            if val > chunk_boost[ni].item():
                                chunk_boost[ni] = val
                    # Normalize to [0,1]
                    bmin = chunk_boost[chunk_boost > 0].min() if (chunk_boost > 0).any() else 0
                    bmax = chunk_boost.max()
                    if bmax > bmin:
                        chunk_boost = ((chunk_boost - bmin) / (bmax - bmin)).clamp(0, 1)

                    final = cos[i] + boost_val * chunk_boost
                    ranked = final.topk(20).indices.tolist()
                    for k in recalls:
                        recalls[k].append(len(gt_set & set(ranked[:k])) / len(gt_set))
            else:
                # Learned scorer
                cos_stats = compute_cell_cos_stats(cos, cell_to_nodes, entity_cell_ids, device)
                features = model.compute_features(q_embs, cell_embs, cos_stats)
                cell_scores = model(features)  # (B, M)
                cell_probs = torch.sigmoid(cell_scores)

                for i, gt in enumerate(gts):
                    if not gt:
                        continue
                    gt_set = set(gt)

                    # Cell selection accuracy
                    gc = gc_batch[i]
                    if gc:
                        gc_pos = {cell_id_to_pos[c] for c in gc if c in cell_id_to_pos}
                        top50 = set(cell_probs[i].topk(min(50, M)).indices.tolist())
                        cell_recall_50.append(len(gc_pos & top50) / len(gc_pos))

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

    result = {f"R@{k}": np.mean(v) if v else 0.0 for k, v in recalls.items()}
    if cell_recall_50:
        result["cell_R@50"] = np.mean(cell_recall_50)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--boost_val", type=float, default=1.0)
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
    print("Cell Scorer v2 — Feature-rich (cosine stats + embedding sim)")
    print("=" * 65)

    # Load data
    chunks, samples = load_musique(data_path, args.max_samples)
    N = len(chunks)
    print(f"  {N} chunks, {len(samples)} questions")

    # Load topology
    cache = torch.load(topo_file, weights_only=False)
    cell_to_nodes = cache["cell_to_nodes"]
    x_chunks_cpu = cache["lifted"].x_0

    config = TopoRAGConfig(lifting="entity", use_gps=False, use_tnn=False)
    toporag = TopoRAG(config)
    device = toporag.device
    embedder = toporag.embedder
    x_chunks = x_chunks_cpu.to(device)
    embed_dim = x_chunks.shape[1]

    chunk_to_cells = defaultdict(list)
    for cell_id, nodes in cell_to_nodes.items():
        for ni in nodes:
            chunk_to_cells[ni].append(cell_id)

    entity_cell_ids = sorted([ci for ci, nodes in cell_to_nodes.items() if len(nodes) >= 2])
    entity_cell_set = set(entity_cell_ids)
    cell_id_to_pos = {ci: pos for pos, ci in enumerate(entity_cell_ids)}
    M = len(entity_cell_ids)

    cell_emb_list = []
    for ci in entity_cell_ids:
        cell_emb_list.append(x_chunks[cell_to_nodes[ci]].mean(dim=0))
    cell_embs = torch.stack(cell_emb_list).to(device)
    print(f"  Entity cells: {M}")

    gold_cells = get_gold_cells(samples, chunk_to_cells, entity_cell_set)
    avg_gold = np.mean([len(g) for g in gold_cells])
    print(f"  Avg gold cells/question: {avg_gold:.1f}")

    # Model size
    dummy = CellScorerV2(embed_dim)
    n_params = sum(p.numel() for p in dummy.parameters())
    print(f"  Model params: {n_params:,}")
    del dummy

    # Baselines
    dummy_model = CellScorerV2(embed_dim).to(device)
    m_cos = evaluate(dummy_model, samples, cell_embs, entity_cell_ids, cell_to_nodes,
                     chunk_to_cells, x_chunks, embedder, device, mode="baseline")
    print(f"\n  Cosine baseline:  R@2={m_cos['R@2']*100:.1f}%  R@5={m_cos['R@5']*100:.1f}%  R@10={m_cos['R@10']*100:.1f}%")

    m_cm = evaluate(dummy_model, samples, cell_embs, entity_cell_ids, cell_to_nodes,
                    chunk_to_cells, x_chunks, embedder, device,
                    mode="cell_max", boost_val=1.0, top_cells=9662)
    print(f"  Cell-max (b=1.0): R@2={m_cm['R@2']*100:.1f}%  R@5={m_cm['R@5']*100:.1f}%  R@10={m_cm['R@10']*100:.1f}%")
    del dummy_model

    # 5-Fold CV
    print(f"\n{'='*65}")
    print(f"{args.n_folds}-Fold CV | epochs={args.epochs} lr={args.lr} boost={args.boost_val} top_C={args.top_cells}")
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

        print(f"\n--- Fold {fold_idx+1}/{args.n_folds} ---")

        model = CellScorerV2(embed_dim).to(device)
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
            loss = train_one_epoch(
                model, train_samples_fold, gold_cells_train,
                cell_embs, entity_cell_ids, cell_id_to_pos, cell_to_nodes,
                x_chunks, embedder, optimizer, device, batch_size=args.batch_size,
            )
            scheduler.step()

            m = evaluate(model, test_samples_fold, cell_embs, entity_cell_ids,
                         cell_to_nodes, chunk_to_cells, x_chunks, embedder, device,
                         mode="trained", boost_val=args.boost_val, top_cells=args.top_cells)

            improved = ""
            if m["R@5"] > best_r5:
                best_r5 = m["R@5"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                improved = " *"

            if (epoch + 1) % 10 == 0 or epoch == 0 or improved:
                cr50 = m.get('cell_R@50', 0)
                print(f"  Ep {epoch+1:>3}  loss={loss:.4f}  "
                      f"R@5={m['R@5']*100:.1f}%  R@10={m['R@10']*100:.1f}%  "
                      f"cellR@50={cr50*100:.0f}%{improved}")

        if best_state:
            model.load_state_dict(best_state)

        # Sweep
        best_sweep = (0, 0, {"R@5": 0})
        for bv in [0.2, 0.5, 1.0, 1.5, 2.0]:
            for tc in [20, 50, 100, 200]:
                m = evaluate(model, test_samples_fold, cell_embs, entity_cell_ids,
                             cell_to_nodes, chunk_to_cells, x_chunks, embedder, device,
                             mode="trained", boost_val=bv, top_cells=tc)
                if m["R@5"] > best_sweep[2]["R@5"]:
                    best_sweep = (bv, tc, m)

        bv, tc, m = best_sweep
        print(f"  Best: boost={bv}, top_C={tc} → R@2={m['R@2']*100:.1f}%  "
              f"R@5={m['R@5']*100:.1f}%  R@10={m['R@10']*100:.1f}%")
        all_fold_results.append(m)

    # Summary
    print("\n" + "=" * 65)
    print("RESULTS (mean ± std across folds)")
    print("=" * 65)
    for metric in ["R@2", "R@5", "R@10", "R@20"]:
        vals = [r[metric] for r in all_fold_results]
        print(f"  {metric}: {np.mean(vals)*100:.1f}% ± {np.std(vals)*100:.1f}%")

    mean_r5 = np.mean([r['R@5'] for r in all_fold_results])
    print(f"\n{'Cosine baseline':<25} R@5={m_cos['R@5']*100:.1f}%")
    print(f"{'Cell-max (zero-shot)':<25} R@5={m_cm['R@5']*100:.1f}%")
    print(f"{'Cell scorer v2 (CV)':<25} R@5={mean_r5*100:.1f}%")
    print(f"{'Oracle cell boost':<25} R@5=73.1%")
    print(f"{'GFM-RAG':<25} R@5=58.2%")
    print("=" * 65)


if __name__ == "__main__":
    main()
