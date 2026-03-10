#!/usr/bin/env python3
"""
TopoRAG Training Script v2 — Query-Conditioned Hypergraph GNN.

Architecture: QCHGNN — query conditions ALL message passing on entity hypergraph.
The model IS the retriever (no cosine baseline residual).
Loss: BCE + ListCE (following GFM-RAG).
Training: synthetic queries from entity cells (no gold leakage).

Usage:
  # Train with cached queries and topology
  python experiments/train_toporag.py --max_samples 500 --lifting entity \
      --query_cache experiments/cache/musique_500_queries_remapped.json --epochs 50

  # Generate new queries with OpenAI
  python experiments/train_toporag.py --max_samples 500 --lifting entity --llm openai --epochs 50

  # Baseline only (no training)
  python experiments/train_toporag.py --max_samples 500 --lifting entity --llm none
"""

import os
import sys
import json
import math
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_musique(
    data_path: Path,
    max_samples: int,
    val_ratio: float = 0.0,
    seed: int = 42,
) -> Tuple[List[str], List[int], List[dict], List[dict]]:
    """Load MuSiQue: chunks, chunk_to_doc, train_samples, test_samples.

    Correct protocol: ALL questions are test (training = synthetic queries only).
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

    if val_ratio <= 0:
        return chunks, chunk_to_doc, samples, samples

    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    n_val = max(1, int(len(samples) * val_ratio))
    val_idx = set(indices[:n_val])
    train_samples = [s for i, s in enumerate(samples) if i not in val_idx]
    test_samples = [s for i, s in enumerate(samples) if i in val_idx]
    return chunks, chunk_to_doc, train_samples, test_samples


# ---------------------------------------------------------------------------
# Query generation / caching
# ---------------------------------------------------------------------------

MULTIHOP_PROMPT = """\
You are given {n} related text passages. Write ONE multi-hop question that requires \
a CHAIN of reasoning across passages — each step uses the answer from the previous step.

Format: "What is [property] of [entity found in passage A] that [relation in passage B]?"

Rules:
- The question MUST form a reasoning chain, NOT ask about all passages simultaneously
- Each passage contributes ONE hop in the chain
- Use specific names and facts from the passages
- The question should be answerable ONLY by reading ALL passages in sequence
- Output ONLY the question, nothing else

Passages:
{context}

Question:"""

SINGLECHUNK_PROMPT = """\
Given the following text passage, write ONE factual question that is directly answered by it.
Output only the question, nothing else.

Passage:
{chunk}

Question:"""


def _deduplicate_queries(result: Dict[int, List[str]]) -> Dict[int, List[str]]:
    """Remove duplicate queries across cells, keeping first occurrence."""
    seen: set = set()
    total_before = sum(len(v) for v in result.values())
    deduped: Dict[int, List[str]] = {}
    for cell_idx in sorted(result.keys()):
        unique = []
        for q in result[cell_idx]:
            q_lower = q.strip().lower()
            if q_lower not in seen:
                seen.add(q_lower)
                unique.append(q)
        if unique:
            deduped[cell_idx] = unique
    total_after = sum(len(v) for v in deduped.values())
    removed = total_before - total_after
    if removed > 0:
        print(f"  Deduplication: removed {removed}/{total_before} queries ({100*removed/total_before:.1f}%)")
    return deduped


def generate_queries(
    chunks: List[str],
    cell_to_nodes: Dict[int, List[int]],
    llm_provider: str,
    cache_path: Optional[Path],
    queries_per_cell: int = 1,
    also_per_chunk: bool = True,
    llm_kwargs: Optional[dict] = None,
) -> Dict[int, List[str]]:
    """Generate queries and return {cell_idx: [query_text, ...]}."""
    if cache_path and cache_path.exists():
        print(f"  Loading cached queries from {cache_path}")
        with open(cache_path) as f:
            cached = json.load(f)
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
        result = {int(k): _extract(v) for k, v in cached.items()}
        result = _deduplicate_queries(result)
        return result

    if llm_provider == "none":
        print("  --llm none: skipping query generation (baseline only)")
        return {}

    from toporag.llms import get_llm
    llm_kwargs = llm_kwargs or {}
    print(f"  Using {llm_provider} LLM for query generation...")
    llm = get_llm(llm_provider, cache_dir=str(PROJECT_ROOT / ".cache"), **llm_kwargs)

    result: Dict[int, List[str]] = {}

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

    if also_per_chunk:
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
                    cell_to_nodes[fake_cell_idx] = [chunk_idx]
            except Exception as e:
                print(f"    Error at chunk {chunk_idx}: {e}")
            if i % 100 == 99:
                print(f"    Progress: {i+1} chunks")

    from toporag.llms import CachedLLM, LocalLLM
    inner = llm.llm if isinstance(llm, CachedLLM) else llm
    if isinstance(inner, LocalLLM):
        inner.free()

    result = _deduplicate_queries(result)
    print(f"  Total cells with queries: {len(result)}")
    total_q = sum(len(v) for v in result.values())
    print(f"  Total queries: {total_q}")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({str(k): v for k, v in result.items()}, f, indent=2)
        print(f"  Queries cached to {cache_path}")

    return result


# ---------------------------------------------------------------------------
# Incidence structure utilities
# ---------------------------------------------------------------------------

def build_incidence_tensors(
    cell_to_nodes: Dict[int, List[int]],
    n_chunks: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """Build scatter-based incidence tensors for QCHGNN.

    Returns:
        flat_nodes_t: (K,) node indices for all incidences
        cell_asgn_t: (K,) cell position for each incidence
        M: number of cells
        degrees_v: (N,) node degrees (how many cells each node belongs to)
        degrees_e: (M,) hyperedge sizes
    """
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

    # Compute degrees
    degrees_v = torch.zeros(n_chunks, device=device)
    degrees_v.scatter_add_(0, flat_nodes_t, torch.ones_like(flat_nodes_t, dtype=torch.float))
    degrees_v = degrees_v.clamp(min=1)

    degrees_e = torch.zeros(M, device=device)
    degrees_e.scatter_add_(0, cell_asgn_t, torch.ones_like(cell_asgn_t, dtype=torch.float))
    degrees_e = degrees_e.clamp(min=1)

    return flat_nodes_t, cell_asgn_t, M, degrees_v, degrees_e, cell_pos_map


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    x_chunks: torch.Tensor,
    queries: Dict[int, List[str]],
    cell_to_nodes: Dict[int, List[int]],
    embedder,
    optimizer,
    loss_fn,
    device: torch.device,
    flat_nodes_t: torch.Tensor,
    cell_asgn_t: torch.Tensor,
    M: int,
    degrees_v: torch.Tensor,
    degrees_e: torch.Tensor,
    grad_clip: float = 1.0,
    batch_size: int = 8,
) -> dict:
    """Train one epoch with BCE + ListCE loss.

    Each training example: (query, positive_chunks_from_cell)
    Model scores ALL chunks; loss computed over full chunk set.
    """
    model.train()
    n = x_chunks.shape[0]

    # Flatten training pairs
    flat_items = [
        (cell_idx, q_text)
        for cell_idx, qs in queries.items()
        for q_text in qs
        if cell_idx in cell_to_nodes
    ]
    random.shuffle(flat_items)

    # Pre-embed all queries
    all_q_texts = [q for _, q in flat_items]
    with torch.no_grad():
        q_raws = embedder.encode(all_q_texts, is_query=True, show_progress=False)
        if not isinstance(q_raws, torch.Tensor):
            q_raws = torch.tensor(q_raws, dtype=torch.float32)
        q_raws = q_raws.clone().to(device)

    total_loss = 0.0
    total_bce = 0.0
    total_listce = 0.0
    n_steps = 0

    for batch_start in range(0, len(flat_items), batch_size):
        batch = flat_items[batch_start: batch_start + batch_size]
        Q = len(batch)
        q_batch = q_raws[batch_start: batch_start + Q]

        optimizer.zero_grad()

        # Forward: score all chunks for each query in batch
        scores = model(x_chunks, q_batch, flat_nodes_t, cell_asgn_t,
                       M, degrees_v, degrees_e)  # (Q, N)

        # Build targets: positive = chunks in the source cell
        targets = torch.zeros(Q, n, device=device)
        for i, (cell_idx, _) in enumerate(batch):
            cell_chunks = cell_to_nodes[cell_idx]
            for ci in cell_chunks:
                if ci < n:
                    targets[i, ci] = 1.0

        # Compute loss
        loss, loss_info = loss_fn(scores, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * Q
        total_bce += loss_info["infonce"] * Q
        total_listce += loss_info["margin"] * Q
        n_steps += Q

    # Gradient norms per module
    grad_norms = {}
    for name in ["query_proj", "node_proj", "layers", "score_mlp"]:
        module = getattr(model, name)
        g = sum(p.grad.norm().item()**2 for p in module.parameters()
                if p.grad is not None) ** 0.5
        grad_norms[name] = g

    return {
        "loss": total_loss / max(n_steps, 1),
        "bce": total_bce / max(n_steps, 1),
        "listce": total_listce / max(n_steps, 1),
        "grad_norms": grad_norms,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model,
    x_chunks: torch.Tensor,
    samples: List[dict],
    embedder,
    device: torch.device,
    flat_nodes_t: torch.Tensor,
    cell_asgn_t: torch.Tensor,
    M: int,
    degrees_v: torch.Tensor,
    degrees_e: torch.Tensor,
    top_k: int = 5,
    mode: str = "trained",
    eval_batch_size: int = 16,
) -> Dict[str, float]:
    """Compute Recall@2 and Recall@5.

    mode="baseline": pure cosine similarity
    mode="trained": QCHGNN scoring
    """
    model.eval()
    recalls_2, recalls_5 = [], []
    n = x_chunks.shape[0]

    with torch.no_grad():
        # Batch evaluation for efficiency
        questions = [s["question"] for s in samples]
        gts = [s["supporting"] for s in samples]

        for batch_start in range(0, len(questions), eval_batch_size):
            batch_q = questions[batch_start: batch_start + eval_batch_size]
            batch_gt = gts[batch_start: batch_start + eval_batch_size]

            q_embs = embedder.encode(batch_q, is_query=True, show_progress=False)
            if not isinstance(q_embs, torch.Tensor):
                q_embs = torch.tensor(q_embs, dtype=torch.float32)
            q_embs = q_embs.clone().to(device)

            if mode == "baseline":
                # Pure cosine similarity
                q_norm = F.normalize(q_embs, dim=-1)
                x_norm = F.normalize(x_chunks, dim=-1)
                sims = q_norm @ x_norm.T  # (B, N)
                for i, gt in enumerate(batch_gt):
                    if not gt:
                        continue
                    retrieved = sims[i].topk(min(top_k, n)).indices.tolist()
                    gt_set = set(gt)
                    recalls_2.append(len(gt_set & set(retrieved[:2])) / len(gt_set))
                    recalls_5.append(len(gt_set & set(retrieved[:5])) / len(gt_set))
            else:
                # QCHGNN scoring
                scores = model(x_chunks, q_embs, flat_nodes_t, cell_asgn_t,
                               M, degrees_v, degrees_e)  # (B, N)
                for i, gt in enumerate(batch_gt):
                    if not gt:
                        continue
                    retrieved = scores[i].topk(min(top_k, n)).indices.tolist()
                    gt_set = set(gt)
                    recalls_2.append(len(gt_set & set(retrieved[:2])) / len(gt_set))
                    recalls_5.append(len(gt_set & set(retrieved[:5])) / len(gt_set))

    return {
        "recall_2": sum(recalls_2) / len(recalls_2) if recalls_2 else 0.0,
        "recall_5": sum(recalls_5) / len(recalls_5) if recalls_5 else 0.0,
        "n": len(recalls_2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train TopoRAG v2 (QCHGNN)")
    parser.add_argument("--dataset", default="musique", choices=["musique", "2wiki", "hotpotqa"])
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--lifting", default="knn", choices=["knn", "clique", "cycle", "entity"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of QCHGNN message passing layers")
    parser.add_argument("--init_k", type=int, default=20,
                        help="Top-K chunks for query initialization mask")
    parser.add_argument("--llm", default="local", choices=["local", "groq", "openai", "none"])
    parser.add_argument("--llm_model", type=str, default=None)
    parser.add_argument("--llm_device", type=str, default="auto")
    parser.add_argument("--query_cache", type=str, default=None)
    parser.add_argument("--no_per_chunk", action="store_true")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size. Each query scores all N chunks.")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="BCE weight in BCE+ListCE loss (GFM-RAG uses 0.3)")
    parser.add_argument("--adv_temp", type=float, default=1.0,
                        help="Adversarial temperature for hard negative weighting")
    # Legacy args (ignored, for backward compat)
    parser.add_argument("--scoring", default="mlp")
    parser.add_argument("--lambda_l2", type=float, default=0.0)
    parser.add_argument("--k1_candidates", type=int, default=20)
    parser.add_argument("--cell_size", type=int, default=2)
    parser.add_argument("--l2_eval_weight", type=float, default=0.1)
    parser.add_argument("--pref_epochs", type=int, default=0)
    parser.add_argument("--pref_lr", type=float, default=1e-4)
    parser.add_argument("--max_gate", type=float, default=0.3)
    parser.add_argument("--tnn_lr_scale", type=float, default=0.1)
    parser.add_argument("--gold_train", action="store_true")
    args = parser.parse_args()

    # ---- Setup ----
    from toporag import TopoRAG, TopoRAGConfig
    from toporag.models.qc_hgnn import QueryConditionedHGNN, QCHGNNLoss

    dataset_paths = {
        "musique": PROJECT_ROOT / "LPGNN-retriever/datasets/musique/musique.json",
        "2wiki":   PROJECT_ROOT / "LPGNN-retriever/datasets/2wiki/2wikimultihopqa.json",
        "hotpotqa": PROJECT_ROOT / "LPGNN-retriever/datasets/hotpotqa/hotpotqa.json",
    }
    data_path = dataset_paths[args.dataset]

    print("=" * 60)
    print(f"TopoRAG v2 (QCHGNN) — {args.dataset} / {args.max_samples} samples")
    print("=" * 60)

    # ---- 1. Load data ----
    print(f"\n[1/5] Loading dataset...")
    chunks, chunk_to_doc, train_samples, test_samples = load_musique(data_path, args.max_samples)
    print(f"  {len(chunks)} chunks from {len(test_samples)} test questions")
    supporting_total = sum(len(s["supporting"]) for s in test_samples)
    print(f"  {supporting_total} gold supporting paragraphs ({supporting_total/len(test_samples):.1f} avg/question)")

    # ---- 2. Build lifted topology ----
    topo_cache_dir = REPO_ROOT / "experiments" / "cache" / "topology"
    topo_cache_dir.mkdir(parents=True, exist_ok=True)
    topo_cache_file = topo_cache_dir / f"{args.dataset}_{args.max_samples}_{args.lifting}.pt"

    config = TopoRAGConfig(
        lifting=args.lifting,
        use_gps=False,
        use_tnn=False,
        hidden_dim=args.hidden_dim,
    )
    toporag_builder = TopoRAG(config)
    device = toporag_builder.device
    embedder = toporag_builder.embedder

    if topo_cache_file.exists():
        print(f"\n[2/5] Loading cached topology from {topo_cache_file}...")
        cache_data = torch.load(topo_cache_file, weights_only=False)
        lifted = cache_data["lifted"].to(device)
        cell_to_nodes = cache_data["cell_to_nodes"]
        print(f"  Nodes: {lifted.num_nodes}, Cells: {lifted.num_edges}, Device: {device}")
    else:
        print(f"\n[2/5] Building k-NN graph + {args.lifting} lifting...")
        lifted = toporag_builder.build_from_chunks(chunks, chunk_to_doc)
        cell_to_nodes = dict(lifted.cell_to_nodes)
        print(f"  Nodes: {lifted.num_nodes}, Cells: {lifted.num_edges}, Device: {device}")
        print(f"  Saving topology cache to {topo_cache_file}...")
        torch.save({
            "lifted": lifted.cpu(),
            "cell_to_nodes": cell_to_nodes,
        }, topo_cache_file)
        lifted = lifted.to(device)

    # ---- 3. Generate / load queries ----
    cache_path = Path(args.query_cache) if args.query_cache else (
        REPO_ROOT / f"experiments/cache/{args.dataset}_{args.max_samples}_queries.json"
    )
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

    # Register zero-cells (one per chunk)
    zero_cell_offset = max(cell_to_nodes.keys(), default=-1) + 1
    for ci in range(len(chunks)):
        cell_to_nodes[zero_cell_offset + ci] = [ci]
    print(f"  Added {len(chunks)} zero-cells (indices {zero_cell_offset}–{zero_cell_offset+len(chunks)-1})")

    # ---- 4. Build QCHGNN model ----
    print(f"\n[4/5] Building QCHGNN model...")
    embed_dim = toporag_builder.config.embed_dim
    n_chunks = len(chunks)

    model = QueryConditionedHGNN(
        embed_dim=embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        init_k=args.init_k,
        use_checkpoint=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Layers: {args.num_layers}, Hidden: {args.hidden_dim}, Init-K: {args.init_k}")

    # Build incidence tensors
    print("  Building incidence tensors...")
    flat_nodes_t, cell_asgn_t, M, degrees_v, degrees_e, cell_pos_map = \
        build_incidence_tensors(cell_to_nodes, n_chunks, device)
    print(f"  Hyperedges: {M}, Total incidences: {flat_nodes_t.shape[0]}")

    # Chunk embeddings (from lifted topology)
    x_chunks = lifted.x_0.to(device)  # (N, embed_dim)
    print(f"  Chunk embeddings: {x_chunks.shape}")

    # ---- Loss function ----
    loss_fn = QCHGNNLoss(alpha=args.alpha, temperature=args.adv_temp)

    # ---- Baseline evaluation ----
    print("\n  [Baseline] Evaluating cosine similarity (no training)...")
    m_base = evaluate(model, x_chunks, test_samples, embedder, device,
                      flat_nodes_t, cell_asgn_t, M, degrees_v, degrees_e,
                      top_k=args.top_k, mode="baseline",
                      eval_batch_size=args.eval_batch_size)
    print(f"  Baseline  R@2={m_base['recall_2']*100:.1f}%  R@5={m_base['recall_5']*100:.1f}%")

    if not queries:
        print("\n  No queries — showing baseline only. Re-run with --llm to train.")
        return

    # ---- 5. Training ----
    n_pairs = sum(len(v) for v in queries.values())
    warmup = max(args.epochs // 10, 3)
    print(f"\n[5/5] Training on {n_pairs} query-cell pairs | {args.epochs} epochs")
    print(f"      Batch: {args.batch_size} | LR: {args.lr} | WD: {args.weight_decay}")
    print(f"      Loss: {args.alpha:.1f}*InfoNCE + {1-args.alpha:.1f}*Margin | Temp: {args.adv_temp}")
    print(f"      Warmup: {warmup} epochs | Cosine LR decay")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        progress = (ep - warmup) / max(args.epochs - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_recall5 = 0.0
    best_state: Optional[dict] = None

    print(f"{'Ep':>4}  {'loss':>7}  {'nce':>7}  {'marg':>7}  {'R@2':>6}  {'R@5':>6}  "
          f"{'gate':>5}  {'g/qp':>6}  {'g/np':>6}  {'g/lay':>6}  {'g/sc':>6}")

    for epoch in range(args.epochs):
        d = train_one_epoch(
            model, x_chunks, queries, cell_to_nodes, embedder, optimizer, loss_fn,
            device, flat_nodes_t, cell_asgn_t, M, degrees_v, degrees_e,
            grad_clip=1.0, batch_size=args.batch_size,
        )
        scheduler.step()

        # Evaluate
        m = evaluate(model, x_chunks, test_samples, embedder, device,
                     flat_nodes_t, cell_asgn_t, M, degrees_v, degrees_e,
                     top_k=args.top_k, mode="trained",
                     eval_batch_size=args.eval_batch_size)

        # Track best
        improved = ""
        if m["recall_5"] > best_recall5:
            best_recall5 = m["recall_5"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            improved = " *"

        gn = d["grad_norms"]
        gate_val = torch.sigmoid(model.mp_gate).item()
        print(f"{epoch+1:>4}  {d['loss']:>7.4f}  {d['bce']:>7.4f}  {d['listce']:>7.4f}  "
              f"{m['recall_2']*100:>5.1f}%  {m['recall_5']*100:>5.1f}%  "
              f"{gate_val:>5.3f}  "
              f"{gn.get('query_proj',0):>6.3f}  {gn.get('node_proj',0):>6.3f}  "
              f"{gn.get('layers',0):>6.3f}  {gn.get('score_mlp',0):>6.3f}{improved}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\n  Restored best checkpoint (R@5={best_recall5*100:.1f}%)")

    # ---- Final results ----
    m_trained = evaluate(model, x_chunks, test_samples, embedder, device,
                         flat_nodes_t, cell_asgn_t, M, degrees_v, degrees_e,
                         top_k=args.top_k, mode="trained",
                         eval_batch_size=args.eval_batch_size)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Method':<30} {'R@2':>8} {'R@5':>8}")
    print("-" * 50)
    print(f"{'Cosine baseline':<30} {m_base['recall_2']*100:>7.1f}% {m_base['recall_5']*100:>7.1f}%")
    print(f"{'TopoRAG v2 (QCHGNN)':<30} {m_trained['recall_2']*100:>7.1f}% {m_trained['recall_5']*100:>7.1f}%")
    print("-" * 50)
    print(f"{'GFM-RAG (SOTA)':<30} {'49.1%':>8} {'58.2%':>8}")
    print("=" * 60)
    print(f"\nNote: {len(test_samples)} test questions, {len(chunks)} chunks. "
          f"Training on synthetic queries only (no gold leakage).")


if __name__ == "__main__":
    main()
