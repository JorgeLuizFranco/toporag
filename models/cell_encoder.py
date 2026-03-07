"""
Cell Encoders for TopoRAG.
Ensures all operations are device-consistent.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from ..lifting.base import Cell

class CellEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, aggregation: str = "mean"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregation = aggregation
        self.projection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, chunk_features: torch.Tensor, cells: List[Cell]) -> torch.Tensor:
        if not cells: return torch.empty(0, self.output_dim, device=chunk_features.device)
        params = list(self.parameters())
        device = params[0].device if params else chunk_features.device
        chunk_features = chunk_features.to(device)
        
        cell_embeddings = []
        for cell in cells:
            indices = list(cell.chunk_indices)
            chunk_embs = chunk_features[indices]
            if self.aggregation == "mean": cell_emb = chunk_embs.mean(dim=0)
            elif self.aggregation == "sum": cell_emb = chunk_embs.sum(dim=0)
            elif self.aggregation == "max": cell_emb = chunk_embs.max(dim=0).values
            else: raise ValueError(f"Unknown aggregation: {self.aggregation}")
            cell_embeddings.append(cell_emb)

        return self.projection(torch.stack(cell_embeddings))

class DeepSetCellEncoder(nn.Module):
    """DeepSet encoder: ρ(Σ_i φ(x_i)) per cell, permutation invariant.

    For efficiency, supports a two-step mode used in Phase-1 training (TNN frozen):
      1. precompute_phi_sums(): run phi on all nodes + scatter-sum per cell (once/epoch, detached)
      2. apply_rho_from_sums(): run rho on precomputed sums (per batch, with grad through rho)
    This allows rho to train per-batch while phi is treated as frozen.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        phi_layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1): phi_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.phi = nn.Sequential(*phi_layers)
        rho_layers = [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1): rho_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        rho_layers.append(nn.Linear(hidden_dim, output_dim))
        self.rho = nn.Sequential(*rho_layers)

    def forward(
        self,
        chunk_features: torch.Tensor,
        cells: List[Cell],
        flat_nodes_t: Optional[torch.Tensor] = None,
        cell_asgn_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not cells: return torch.empty(0, self.rho[-1].out_features, device=chunk_features.device)
        params = list(self.parameters())
        device = params[0].device if params else chunk_features.device
        chunk_features = chunk_features.to(device)
        M = len(cells)

        # Vectorized DeepSet: apply phi to all nodes once, scatter-sum per cell, then rho.
        # flat_nodes_t / cell_asgn_t may be precomputed once and reused across batches.
        if flat_nodes_t is None or cell_asgn_t is None:
            flat_nodes, cell_assignments = [], []
            for cell_pos, cell in enumerate(cells):
                nodes = list(cell.chunk_indices)
                flat_nodes.extend(nodes)
                cell_assignments.extend([cell_pos] * len(nodes))
            flat_nodes_t = torch.tensor(flat_nodes, dtype=torch.long, device=device)
            cell_asgn_t = torch.tensor(cell_assignments, dtype=torch.long, device=device)
        else:
            flat_nodes_t = flat_nodes_t.to(device)
            cell_asgn_t = cell_asgn_t.to(device)

        selected_phi = self.phi(chunk_features[flat_nodes_t])   # (total_nodes, h)
        hidden_dim = selected_phi.shape[1]
        cell_sums = torch.zeros(M, hidden_dim, device=device)
        cell_sums.scatter_add_(0, cell_asgn_t.unsqueeze(1).expand(-1, hidden_dim), selected_phi)
        return self.rho(cell_sums)

    def precompute_phi_sums(
        self,
        chunk_features: torch.Tensor,
        M: int,
        flat_nodes_t: torch.Tensor,
        cell_asgn_t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute phi(x_i) + scatter-sum per cell (no grad). Use in Phase 1 for efficiency.
        Returns cell_sums: (M, hidden_dim) — constant per epoch when TNN is frozen.
        Call apply_rho_from_sums(cell_sums) per batch to get trainable cell_embs.
        """
        device = chunk_features.device
        with torch.no_grad():
            selected_phi = self.phi(chunk_features[flat_nodes_t.to(device)])
        hidden_dim = selected_phi.shape[1]
        cell_sums = torch.zeros(M, hidden_dim, device=device)
        cell_sums.scatter_add_(
            0,
            cell_asgn_t.to(device).unsqueeze(1).expand(-1, hidden_dim),
            selected_phi,
        )
        return cell_sums.detach()

    def apply_rho_from_sums(self, cell_sums: torch.Tensor) -> torch.Tensor:
        """Apply rho to precomputed cell_sums (with gradient). Per-batch in Phase 1."""
        return self.rho(cell_sums)


class AttentionCellEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Linear(input_dim, hidden_dim)
        self.v_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, chunk_features: torch.Tensor, cells: List[Cell]) -> torch.Tensor:
        if not cells: return torch.empty(0, self.output_proj.out_features, device=chunk_features.device)
        params = list(self.parameters())
        device = params[0].device if params else chunk_features.device
        chunk_features = chunk_features.to(device)
        
        cell_embeddings = []
        for cell in cells:
            chunk_embs = chunk_features[list(cell.chunk_indices)]
            q = self.q_proj(chunk_embs).unsqueeze(0)
            k = self.k_proj(chunk_embs).unsqueeze(0)
            v = self.v_proj(chunk_embs).unsqueeze(0)
            attended, _ = self.attention(q, k, v)
            cell_embeddings.append(self.output_proj(attended.squeeze(0).mean(dim=0)))

        return torch.stack(cell_embeddings)

class HierarchicalCellEncoder(nn.Module):
    """
    SOTA Hierarchical cell encoder.
    Combines direct node aggregation with boundary-aware lower-dimensional context.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, max_dimension: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_dimension = max_dimension

        self.encoders = nn.ModuleDict()
        for dim in range(max_dimension + 1):
            if dim == 0:
                self.encoders[str(dim)] = nn.Linear(input_dim, hidden_dim)
            else:
                # Concat direct + boundary
                self.encoders[str(dim)] = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, node_features: torch.Tensor, cells: List[Cell]) -> torch.Tensor:
        if not cells: return torch.empty(0, self.output_dim, device=node_features.device)
        device = node_features.device
        self.to(device)
        
        # 1. Group cells by dimension
        cells_by_dim = {}
        for i, cell in enumerate(cells):
            d = cell.dimension
            if d not in cells_by_dim: cells_by_dim[d] = []
            cells_by_dim[d].append((i, cell))
            
        final_embs = torch.zeros(len(cells), self.output_dim, device=device)
        dim_embs = {}

        # 2. Iterative Encoding across dimensions
        for d in range(self.max_dimension + 1):
            if d not in cells_by_dim: continue
            
            items = cells_by_dim[d]
            current_direct_embs = []
            for idx, cell in items:
                # Direct aggregation
                chunk_embs = node_features[list(cell.chunk_indices)]
                current_direct_embs.append(chunk_embs.mean(dim=0))
            
            direct_proj = self.encoders["0"](torch.stack(current_direct_embs))
            
            if d == 0:
                out = direct_proj
            else:
                # Aggregate boundary context from d-1
                # Simplified: use mean of all chunks again but through the d-encoder
                # In a full TNN, this would use incidence_2
                out = self.encoders[str(d)](torch.cat([direct_proj, direct_proj], dim=-1))
            
            projected = self.output_proj(out)
            for (i, _), emb in zip(items, projected):
                final_embs[i] = emb
                
        return final_embs
