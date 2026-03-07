"""
LP-TNN: Inductive Link Prediction on Topological Complexes.
SOTA Implementation: Optimized for device consistency, gradient flow, and Gating.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from ..lifting.base import Cell, LiftedTopology
from .cell_encoder import CellEncoder, DeepSetCellEncoder
from .link_predictor import QueryCellLinkPredictor
from .tnn import TNN


class LPTNN(nn.Module):
    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 256,
        cell_encoder_type: str = "deepset",
        link_predictor_type: str = "mlp",
        tnn_config: Optional[dict] = None,
    ):
        super().__init__()
        
        # 1. TNN Backbone
        self.tnn = TNN(
            in_channels=embed_dim,
            hidden_channels=tnn_config.get("hidden_dim", hidden_dim) if tnn_config else hidden_dim,
            out_channels=embed_dim,
            num_layers=tnn_config.get("num_layers", 2) if tnn_config else 2,
            model_type="hypergraph_gps",
            dropout=tnn_config.get("dropout", 0.1) if tnn_config else 0.1,
        )

        # 2. Cell Encoder
        if cell_encoder_type == "deepset":
            self.cell_encoder = DeepSetCellEncoder(embed_dim, hidden_dim, hidden_dim)
        else:
            self.cell_encoder = CellEncoder(embed_dim, hidden_dim)

        # 3. Query Projection
        self.query_proj = nn.Linear(embed_dim, hidden_dim)

        # 4. Gating Mechanism
        self.topo_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.topo_gate[-2].bias, -2.0)

        # 5. Link Predictor
        self.link_predictor = QueryCellLinkPredictor(hidden_dim, hidden_dim, hidden_dim, scoring=link_predictor_type)

        # 6. Score gate for cosine residual — init 0 so tanh(0)=0,
        #    model starts at pure cosine baseline and learns corrections.
        self.score_gate = nn.Parameter(torch.zeros(1))

        # 7. Level 2 cell scoring: uses cell_encoder + query_proj + link_predictor
        #    These modules already exist above but had zero gradients in chunk-only scoring.
        #    Level 2 gives them gradients by scoring dynamically constructed cells.


    def score_cells_from_embeddings(
        self,
        q_tnn: torch.Tensor,
        node_embs: torch.Tensor,
        cell_chunk_indices: List[List[int]],
    ) -> torch.Tensor:
        """Level 2 cell scoring: ψ(q, σ) = link_predictor([query_proj(q̃); cell_encoder(σ̃)]).

        Args:
            q_tnn: (Q, embed_dim) TNN-refined query embeddings
            node_embs: (n, embed_dim) TNN-refined chunk embeddings
            cell_chunk_indices: list of M cells, each a list of chunk indices

        Returns:
            (Q, M) cell scores
        """
        device = next(self.parameters()).device
        q_tnn = q_tnn.to(device)
        node_embs = node_embs.to(device)

        # Build Cell objects for DeepSet encoder
        cells_obj = [
            Cell(cell_id=i, chunk_indices=set(indices), dimension=len(indices) - 1)
            for i, indices in enumerate(cell_chunk_indices)
        ]

        # Cell embeddings via DeepSet: ρ(Σ φ(c̃ᵢ)) → (M, hidden_dim)
        cell_embs = self.cell_encoder(node_embs, cells_obj)  # (M, hidden_dim)

        # Query projection: (Q, embed_dim) → (Q, hidden_dim)
        q_proj = self.query_proj(q_tnn)  # (Q, hidden_dim)

        # Link predictor: (Q, M) scores
        scores = self.link_predictor(q_proj, cell_embs)  # (Q, M)
        return scores

    def forward(self, complex_data: LiftedTopology, query_node_indices: torch.Tensor, target_cells: List[Cell]) -> torch.Tensor:
        device = next(self.parameters()).device
        
        # A. Base Branch
        base_node_feats = complex_data.x_0.to(device)
        base_q_hid = self.query_proj(base_node_feats[query_node_indices])
        base_c_hid = self.cell_encoder(base_node_feats, target_cells)

        # B. Topological Branch
        tnn_out = self.tnn.forward_from_lifted(complex_data)
        refined_nodes = tnn_out.x_0.to(device)
        
        topo_q_hid = self.query_proj(refined_nodes[query_node_indices])
        topo_c_hid = self.cell_encoder(refined_nodes, target_cells)
        
        # C. Gating
        gate = self.topo_gate(topo_q_hid)
        q_final = (1 - gate) * base_q_hid + gate * topo_q_hid
        
        gate_exp = gate.unsqueeze(2)
        c_final = (1 - gate_exp) * base_c_hid.unsqueeze(0) + gate_exp * topo_c_hid.unsqueeze(0)

        # D. Scoring
        if self.link_predictor.scoring == "mlp":
            nq, nc = q_final.shape[0], base_c_hid.shape[0]
            q_exp = q_final.unsqueeze(1).expand(-1, nc, -1)
            return self.link_predictor.mlp(torch.cat([q_exp, c_final], dim=-1)).squeeze(-1)
        else:
            return torch.bmm(q_final.unsqueeze(1), c_final.transpose(1, 2)).squeeze(1)


class LPTNNTrainer:
    def __init__(self, model: LPTNN, learning_rate: float = 1e-4, num_negative_samples: int = 5):
        self.model = model
        self.device = next(model.parameters()).device
        self.num_negative_samples = num_negative_samples
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

    def train_step(self, complex_data: LiftedTopology, q_indices: List[int], pos_indices: List[int]) -> float:
        self.model.train()
        self.optimizer.zero_grad()

        device = self.device
        q_idx_tensor = torch.tensor(q_indices, device=device)
        all_cells = complex_data.get_all_higher_order_cells()
        
        scores_matrix = self.model(complex_data, q_idx_tensor, all_cells)

        total_loss = 0.0
        for i in range(len(q_indices)):
            pos_idx = pos_indices[i]
            pos_score = scores_matrix[i, pos_idx:pos_idx+1]
            
            neg_indices = []
            while len(neg_indices) < self.num_negative_samples:
                idx = torch.randint(0, len(all_cells), (1,)).item()
                if idx != pos_idx: neg_indices.append(idx)
            
            neg_scores = scores_matrix[i, neg_indices]
            total_loss += self.criterion(torch.cat([pos_score, neg_scores]), 
                                         torch.cat([torch.ones(1, device=device), torch.zeros(len(neg_indices), device=device)]))

        total_loss /= len(q_indices)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return total_loss.item()

    def get_grad_norm(self):
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5