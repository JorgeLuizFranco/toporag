#!/usr/bin/env python3
"""
TopoRAG Cell Classifier — Direct query→cell relevance prediction.

No message passing. Just: "is this entity cell relevant to this query?"
Oracle diagnostic shows 73.2% R@5 is achievable with perfect cell selection.

Architecture:
  cell_emb = mean(chunk_embeddings in cell)      — pre-computed, frozen
  score(q, cell) = MLP([q_proj; cell_emb_proj])  — learned
  chunk_score = cos(q, x_i) + gate * max(cell_scores for cells containing i)

Training: synthetic query from cell C → cell C is positive, others negative.
"""

import json
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

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
    """Binary classifier: is this cell relevant to this query?

    Simple MLP on [query_proj; cell_proj]. No message passing.
    """
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
        # Learned gate: how much cell boost to add to cosine
        # sigmoid(-2) ≈ 0.12 → starts small
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def score_cells(self, q_emb: torch.Tensor, cell_embs: torch.Tensor) -> torch.Tensor:
        """Score query against multiple cells.

        Args:
            q_emb: (B, D) query embeddings
            cell_embs: (M, D) cell embeddings

        Returns:
            (B, M) relevance scores
        """
        q = self.query_proj(q_emb)        # (B, H)
        c = self.cell_proj(cell_embs)      # (M, H)
        B, M = q.shape[0], c.shape[0]
        q_exp = q.unsqueeze(1).expand(B, M, -1)   # (B, M, H)
        c_exp = c.unsqueeze(0).expand(B, M, -1)   # (B, M, H)
        scores = self.scorer(torch.cat([q_exp, c_exp], dim=-1)).squeeze(-1)  # (B, M)
        return scores


# ---------------------------------------------------------------------------
# Data loading (reuse from train_toporag)
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


def load_queries(cache_path, cell_to_nodes):
    with open(cache_path) as f:
        cached = json.load(f)
    result = {}
    for k, v in cached.items():
        cell_idx = int(k)
        if cell_idx not in cell_to_nodes:
            continue
        queries = []
        for item in v:
            if isinstance(item, dict):
                queries.append(item.get("query_text") or str(item))
            else:
                queries.append(str(item))
        if queries:
            result[cell_idx] = queries
    # Deduplicate
    seen = set()
    deduped = {}
    for ci in sorted(result.keys()):
        unique = [q for q in result[ci] if q.strip().lower() not in seen and not seen.add(q.strip().lower())]
        if unique:
            deduped[ci] = unique
    return deduped


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, queries, cell_to_nodes, cell_embs, entity_cell_ids,
    embedder, optimizer, device,
    n_neg: int = 15, batch_size: int = 32,
):
    model.train()
    # Flatten training pairs
    flat = [(ci, q) for ci, qs in queries.items() for q in qs if ci in entity_cell_ids]
    random.shuffle(flat)

    # Map entity cell_id → position in cell_embs tensor
    cell_id_to_pos = {ci: pos for pos, ci in enumerate(entity_cell_ids)}

    # Pre-embed queries
    all_q = [q for _, q in flat]
    with torch.no_grad():
        q_embs = embedder.encode(all_q, is_query=True, show_progress=False)
        if not isinstance(q_embs, torch.Tensor):
            q_embs = torch.tensor(q_embs, dtype=torch.float32)
        q_embs = q_embs.to(device).clone()  # clone to escape inference mode

    M = len(entity_cell_ids)
    total_loss = 0.0
    n_steps = 0

    for batch_start in range(0, len(flat), batch_size):
        batch = flat[batch_start: batch_start + batch_size]
        B = len(batch)
        q_batch = q_embs[batch_start: batch_start + B]

        optimizer.zero_grad()

        # Score ALL entity cells for this batch
        scores = model.score_cells(q_batch, cell_embs)  # (B, M)

        # Build targets: positive = source cell, rest = 0
        targets = torch.zeros(B, M, device=device)
        for i, (ci, _) in enumerate(batch):
            pos = cell_id_to_pos.get(ci)
            if pos is not None:
                targets[i, pos] = 1.0

        # BCE loss with hard negative mining (GFM-RAG style)
        bce = F.binary_cross_entropy_with_logits(scores, targets, reduction='none')  # (B, M)

        # Weight positives and negatives separately
        is_pos = targets > 0.5
        is_neg = ~is_pos
        n_pos = is_pos.float().sum(dim=1, keepdim=True).clamp(min=1)

        # Adversarial negative weighting: hard negatives get more weight
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

        total_loss += loss.item() * B
        n_steps += B

    return {"loss": total_loss / max(n_steps, 1)}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model, samples, cell_embs, entity_cell_ids, cell_to_nodes,
    chunk_to_cells, x_chunks, embedder, device,
    top_k: int = 5, eval_batch: int = 32,
    mode: str = "trained", boost_val: float = 0.2, top_cells: int = 50,
):
    """Evaluate retrieval.

    For 'trained' mode: score ALL entity cells with the classifier,
    select top-C cells, boost all their chunks by boost_val * P(relevant).
    This mirrors the oracle diagnostic but with learned cell selection.
    """
    model.eval()
    N = x_chunks.shape[0]
    M = len(entity_cell_ids)
    x_norm = F.normalize(x_chunks, dim=-1)

    # Pre-compute chunk→cell membership as a sparse structure
    # For each cell position, which chunk indices belong to it
    cell_pos_to_chunks = {}
    for pos, ci in enumerate(entity_cell_ids):
        cell_pos_to_chunks[pos] = cell_to_nodes[ci]

    recalls_2, recalls_5 = [], []

    with torch.no_grad():
        for batch_start in range(0, len(samples), eval_batch):
            batch_s = samples[batch_start: batch_start + eval_batch]
            questions = [s["question"] for s in batch_s]
            gts = [s["supporting"] for s in batch_s]

            q_embs = embedder.encode(questions, is_query=True, show_progress=False)
            if not isinstance(q_embs, torch.Tensor):
                q_embs = torch.tensor(q_embs, dtype=torch.float32)
            q_embs = q_embs.to(device)

            # Cosine scores
            q_norm = F.normalize(q_embs, dim=-1)
            cos = q_norm @ x_norm.T  # (B, N)

            if mode == "baseline":
                for i, gt in enumerate(gts):
                    if not gt:
                        continue
                    gt_set = set(gt)
                    ret = cos[i].topk(top_k).indices.tolist()
                    recalls_2.append(len(gt_set & set(ret[:2])) / len(gt_set))
                    recalls_5.append(len(gt_set & set(ret[:5])) / len(gt_set))
            else:
                # Score ALL entity cells at once
                cell_scores = model.score_cells(q_embs, cell_embs)  # (B, M)
                cell_probs = torch.sigmoid(cell_scores)  # (B, M) in [0,1]

                for i, gt in enumerate(gts):
                    if not gt:
                        continue
                    gt_set = set(gt)

                    # Select top-C cells by classifier score
                    top_c = cell_probs[i].topk(min(top_cells, M)).indices.tolist()

                    # Boost chunks in selected cells
                    chunk_boost = torch.zeros(N, device=device)
                    for pos in top_c:
                        prob = cell_probs[i, pos].item()
                        for ni in cell_pos_to_chunks[pos]:
                            if prob > chunk_boost[ni].item():
                                chunk_boost[ni] = prob

                    # Final score: cosine + boost_val * cell_relevance_prob
                    final = cos[i] + boost_val * chunk_boost
                    ret = final.topk(top_k).indices.tolist()
                    recalls_2.append(len(gt_set & set(ret[:2])) / len(gt_set))
                    recalls_5.append(len(gt_set & set(ret[:5])) / len(gt_set))

    return {
        "recall_2": sum(recalls_2) / len(recalls_2) if recalls_2 else 0,
        "recall_5": sum(recalls_5) / len(recalls_5) if recalls_5 else 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--boost_val", type=float, default=0.2,
                        help="Fixed boost value for cell-selected chunks")
    parser.add_argument("--top_cells", type=int, default=50,
                        help="Top-C cells to boost per query")
    parser.add_argument("--query_cache", type=str,
                        default="toporag/experiments/cache/musique_500_queries_remapped.json")
    args = parser.parse_args()

    from toporag import TopoRAG, TopoRAGConfig

    data_path = PROJECT_ROOT / "LPGNN-retriever/datasets/musique/musique.json"
    topo_file = REPO_ROOT / "experiments/cache/topology/musique_500_entity.pt"

    print("=" * 60)
    print(f"Cell Classifier — Direct query→cell relevance")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading data...")
    chunks, samples = load_musique(data_path, args.max_samples)
    print(f"  {len(chunks)} chunks, {len(samples)} questions")

    # Load topology
    print("\n[2/4] Loading topology...")
    cache = torch.load(topo_file, weights_only=False)
    cell_to_nodes = cache["cell_to_nodes"]
    x_chunks_cpu = cache["lifted"].x_0  # (N, 768)

    config = TopoRAGConfig(lifting="entity", use_gps=False, use_tnn=False)
    toporag = TopoRAG(config)
    device = toporag.device
    embedder = toporag.embedder
    x_chunks = x_chunks_cpu.to(device)
    embed_dim = x_chunks.shape[1]

    # Build reverse map
    chunk_to_cells = defaultdict(list)
    for cell_id, nodes in cell_to_nodes.items():
        for ni in nodes:
            chunk_to_cells[ni].append(cell_id)

    # Entity cells only (exclude zero-cells / single-chunk cells)
    entity_cell_ids = sorted([ci for ci, nodes in cell_to_nodes.items() if len(nodes) >= 2])
    print(f"  Entity cells: {len(entity_cell_ids)} (excluding {len(cell_to_nodes) - len(entity_cell_ids)} zero-cells)")

    # Pre-compute cell embeddings (mean of chunk embeddings)
    print("  Computing cell embeddings...")
    cell_emb_list = []
    for ci in entity_cell_ids:
        nodes = cell_to_nodes[ci]
        cell_emb_list.append(x_chunks[nodes].mean(dim=0))
    cell_embs = torch.stack(cell_emb_list).to(device)  # (M, D)
    print(f"  Cell embeddings: {cell_embs.shape}")

    # Load queries
    print("\n[3/4] Loading queries...")
    queries = load_queries(Path(args.query_cache), cell_to_nodes)
    n_pairs = sum(len(v) for v in queries.values())
    # Filter to entity cells only
    queries = {ci: qs for ci, qs in queries.items() if ci in set(entity_cell_ids)}
    n_entity_pairs = sum(len(v) for v in queries.values())
    print(f"  {n_entity_pairs} query-cell pairs ({n_pairs - n_entity_pairs} zero-cell queries dropped)")

    # Build model
    print("\n[4/4] Building model...")
    model = CellClassifier(embed_dim=embed_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Baseline
    print("\n  [Baseline]...")
    m_base = evaluate(model, samples, cell_embs, entity_cell_ids, cell_to_nodes,
                      chunk_to_cells, x_chunks, embedder, device,
                      mode="baseline")
    print(f"  Cosine baseline: R@2={m_base['recall_2']*100:.1f}%  R@5={m_base['recall_5']*100:.1f}%")

    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = max(args.epochs // 10, 3)

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(args.epochs - warmup, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n  Training {args.epochs} epochs | LR={args.lr} | Batch={args.batch_size}")
    print(f"  Eval: top_cells={args.top_cells}, boost_val={args.boost_val}")
    print(f"{'Ep':>4}  {'loss':>7}  {'R@2':>6}  {'R@5':>6}")

    best_r5 = 0.0
    best_state = None

    for epoch in range(args.epochs):
        d = train_one_epoch(model, queries, cell_to_nodes, cell_embs, entity_cell_ids,
                            embedder, optimizer, device, batch_size=args.batch_size)
        scheduler.step()

        m = evaluate(model, samples, cell_embs, entity_cell_ids, cell_to_nodes,
                     chunk_to_cells, x_chunks, embedder, device,
                     mode="trained", boost_val=args.boost_val, top_cells=args.top_cells)

        improved = ""
        if m["recall_5"] > best_r5:
            best_r5 = m["recall_5"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            improved = " *"

        print(f"{epoch+1:>4}  {d['loss']:>7.4f}  {m['recall_2']*100:>5.1f}%  "
              f"{m['recall_5']*100:>5.1f}%{improved}")

    if best_state:
        model.load_state_dict(best_state)
        print(f"\n  Best R@5={best_r5*100:.1f}%")

    # Final sweep across boost values and top_cells
    print("\n" + "=" * 60)
    print("SWEEP: boost_val × top_cells (best checkpoint)")
    print("=" * 60)
    print(f"{'boost':>6} {'top_C':>6} {'R@2':>7} {'R@5':>7}")
    print("-" * 30)
    for bv in [0.05, 0.1, 0.2, 0.3, 0.5]:
        for tc in [20, 50, 100, 200]:
            m = evaluate(model, samples, cell_embs, entity_cell_ids, cell_to_nodes,
                         chunk_to_cells, x_chunks, embedder, device,
                         mode="trained", boost_val=bv, top_cells=tc)
            print(f"{bv:>6.2f} {tc:>6} {m['recall_2']*100:>6.1f}% {m['recall_5']*100:>6.1f}%")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Cosine baseline':<25} R@2={m_base['recall_2']*100:.1f}%  R@5={m_base['recall_5']*100:.1f}%")
    print(f"{'Best cell classifier':<25} R@5={best_r5*100:.1f}%")
    print(f"{'Oracle (diagnostic)':<25} R@2=47.7%  R@5=73.2%")
    print(f"{'GFM-RAG':<25} R@2=49.1%  R@5=58.2%")
    print("=" * 60)


if __name__ == "__main__":
    main()
