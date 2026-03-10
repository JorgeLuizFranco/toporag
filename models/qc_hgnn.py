"""
Query-Conditioned Hypergraph GNN (QCHGNN) for multi-hop retrieval.

Inspired by GFM-RAG's Neural Bellman-Ford, adapted for entity hypergraphs.
Key idea: the query MODULATES all message passing, so different queries
activate different propagation paths through the hypergraph.

Architecture:
  1. Query-conditioned initialization (top-K cosine mask × chunk+query embedding)
  2. L layers of DistMult-style hypergraph message passing
  3. Per-chunk scoring MLP
  4. BCE + ListCE loss (following GFM-RAG)

Mathematical details in RESEARCH_NOTES.md.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.utils.checkpoint import checkpoint as grad_checkpoint


class QCHGNNLayer(nn.Module):
    """Single layer of query-conditioned hypergraph message passing.

    Message flow:
      1. DistMult: h_modulated = h ⊙ r(q)   (query conditions features)
      2. Node → Hyperedge: scatter-mean of modulated features
      3. Hyperedge → Node: scatter-mean of hyperedge features
      4. Update: h' = LN(h + MLP([msg; h; boundary]))
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.relation_proj = nn.Linear(hidden_dim, hidden_dim)
        # Update MLP: input = [message; current; boundary]
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,           # (B, N, D)
        boundary: torch.Tensor,     # (B, N, D)
        q_proj: torch.Tensor,       # (B, D)
        flat_nodes_t: torch.Tensor, # (K,)
        cell_asgn_t: torch.Tensor,  # (K,)
        M: int,
        degrees_v: torch.Tensor,    # (N,)
        degrees_e: torch.Tensor,    # (M,)
    ) -> torch.Tensor:
        B, N, D = h.shape
        K = flat_nodes_t.shape[0]

        # Query-dependent relation embedding
        r = self.relation_proj(q_proj)  # (B, D)

        # DistMult: query modulates which features propagate
        h_mod = h * r.unsqueeze(1)  # (B, N, D)

        # Node → Hyperedge: scatter-mean
        h_selected = h_mod[:, flat_nodes_t]  # (B, K, D)
        e = torch.zeros(B, M, D, device=h.device)
        idx = cell_asgn_t.view(1, K, 1).expand(B, K, D)
        e.scatter_add_(1, idx, h_selected)
        e = e / degrees_e.view(1, M, 1).clamp(min=1)

        # Hyperedge → Node: scatter-mean
        e_selected = e[:, cell_asgn_t]  # (B, K, D)
        msg = torch.zeros(B, N, D, device=h.device)
        idx_n = flat_nodes_t.view(1, K, 1).expand(B, K, D)
        msg.scatter_add_(1, idx_n, e_selected)
        msg = msg / degrees_v.view(1, N, 1).clamp(min=1)

        # Update with boundary condition
        h_cat = torch.cat([msg, h, boundary], dim=-1)  # (B, N, 3D)
        h_new = self.layer_norm(h + self.dropout(self.update_mlp(h_cat)))
        return h_new


class QueryConditionedHGNN(nn.Module):
    """Query-Conditioned Hypergraph GNN for chunk-level retrieval.

    Architecture (hybrid cosine + message passing):
      score(q, chunk_i) = cos(q, x_i) + MLP([h_i^L(q); q_proj])

    The cosine term provides a strong baseline floor (40.5% R@5).
    The MLP correction learns to boost chunks reached by query-conditioned
    propagation through entity hyperedges. Initialized to ~0 so the model
    starts at pure cosine and learns improvements.

    Initialization: ALL chunks get x_proj (preserves identity). Seeded
    chunks (top-K cosine) additionally get q_proj (propagation source).
    This ensures non-seeded chunks still have meaningful differentiation.

    Args:
        embed_dim: dimension of input chunk/query embeddings (768)
        hidden_dim: dimension of hidden representations (256)
        num_layers: number of message passing layers (3)
        dropout: dropout rate
        init_k: number of top-K chunks for initialization mask
        use_checkpoint: use gradient checkpointing to save memory
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        init_k: int = 20,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.init_k = init_k
        self.use_checkpoint = use_checkpoint

        # Projections for initialization
        self.query_proj = nn.Linear(embed_dim, hidden_dim)
        self.node_proj = nn.Linear(embed_dim, hidden_dim)

        # Message passing layers
        self.layers = nn.ModuleList([
            QCHGNNLayer(hidden_dim, dropout) for _ in range(num_layers)
        ])

        # MP correction head: [h^L; q_proj] → scalar correction
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Learned gate: sigmoid(-1)≈0.27 → starts at 27% correction
        # -3 was too conservative (5%), gate barely moved in training
        self.mp_gate = nn.Parameter(torch.tensor(-1.0))

    def forward(
        self,
        x: torch.Tensor,             # (N, embed_dim) chunk embeddings
        q: torch.Tensor,             # (B, embed_dim) query embeddings
        flat_nodes_t: torch.Tensor,  # (K,) node indices in incidence
        cell_asgn_t: torch.Tensor,   # (K,) cell assignments in incidence
        M: int,                       # number of hyperedges
        degrees_v: torch.Tensor,     # (N,) node degrees
        degrees_e: torch.Tensor,     # (M,) hyperedge sizes
    ) -> torch.Tensor:
        """Score all chunks for each query.

        Returns:
            scores: (B, N) — higher = more relevant.
            Combines cosine baseline + learned MP correction.
        """
        B, N = q.shape[0], x.shape[0]

        # Cosine baseline: always available, provides floor ranking
        q_norm = F.normalize(q, dim=-1)      # (B, D_emb)
        x_norm = F.normalize(x, dim=-1)      # (N, D_emb)
        cos_scores = q_norm @ x_norm.T       # (B, N)

        # Project query and nodes to hidden dim
        q_proj = self.query_proj(q)    # (B, D)
        x_proj = self.node_proj(x)     # (N, D)

        # Hybrid initialization:
        #   ALL chunks get x_proj (preserves chunk identity / cosine-like ranking)
        #   Seeded chunks (top-K cosine) additionally get q_proj (propagation source)
        with torch.no_grad():
            K = min(self.init_k, N)
            topk_idx = cos_scores.topk(K, dim=1).indices  # (B, K)
            mask = torch.zeros(B, N, device=x.device)
            mask.scatter_(1, topk_idx, 1.0)

        # h = x_proj (all chunks) + mask * q_proj (seeded chunks boosted)
        h = x_proj.unsqueeze(0).expand(B, -1, -1) + mask.unsqueeze(-1) * q_proj.unsqueeze(1)
        boundary = h  # save for boundary condition

        # L layers of query-conditioned message passing
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                h = grad_checkpoint(
                    layer,
                    h, boundary, q_proj,
                    flat_nodes_t, cell_asgn_t, M, degrees_v, degrees_e,
                    use_reentrant=False,
                )
            else:
                h = layer(h, boundary, q_proj,
                          flat_nodes_t, cell_asgn_t, M, degrees_v, degrees_e)

        # Score = cosine + gate * MP_correction
        # Gate starts at sigmoid(-3)≈0.05, preventing MP from overwhelming cosine.
        # As model learns useful propagation patterns, gate opens.
        q_expanded = q_proj.unsqueeze(1).expand(B, N, self.hidden_dim)
        mp_correction = self.score_mlp(torch.cat([h, q_expanded], dim=-1)).squeeze(-1)  # (B, N)
        gate = torch.sigmoid(self.mp_gate)
        scores = cos_scores + gate * mp_correction

        return scores


class QCHGNNLoss(nn.Module):
    """Combined InfoNCE + margin ranking loss.

    Scores are cosine + MP correction (unbounded reals, not logits).
    Uses softmax cross-entropy (InfoNCE) over all chunks as primary loss,
    plus a margin loss pushing positives above top negative.

    Args:
        alpha: weight for InfoNCE (1-alpha for margin loss)
        margin: margin for ranking loss
        temperature: softmax temperature for InfoNCE
    """

    def __init__(self, alpha: float = 0.7, margin: float = 0.1, temperature: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.margin = margin
        self.temperature = temperature

    def forward(
        self,
        scores: torch.Tensor,   # (B, N) real-valued scores
        targets: torch.Tensor,  # (B, N) binary labels {0, 1}
    ) -> Tuple[torch.Tensor, dict]:
        B, N = scores.shape

        is_pos = targets > 0.5
        num_pos = is_pos.float().sum(dim=1).clamp(min=1)  # (B,)

        # --- InfoNCE: softmax cross-entropy with multi-positive soft targets ---
        # Target distribution: uniform over positive chunks
        target_dist = targets / num_pos.unsqueeze(1)  # (B, N)
        logits = scores / self.temperature  # (B, N)
        infonce = F.cross_entropy(logits, target_dist, reduction='mean')

        # --- Margin ranking: push best positive above best negative ---
        # For each query, max positive score should exceed max negative by margin
        scores_pos = scores.clone()
        scores_pos[~is_pos] = float('-inf')
        best_pos = scores_pos.max(dim=1).values  # (B,)

        scores_neg = scores.clone()
        scores_neg[is_pos] = float('-inf')
        best_neg = scores_neg.max(dim=1).values  # (B,)

        margin_loss = F.relu(self.margin - (best_pos - best_neg)).mean()

        loss = self.alpha * infonce + (1 - self.alpha) * margin_loss

        return loss, {
            "infonce": infonce.item(),
            "margin": margin_loss.item(),
            "total": loss.item(),
        }
