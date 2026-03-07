"""
Topological Neural Networks (TNN) for TopoRAG.

Implements message passing on cell/simplicial complexes following TopoModelX conventions.
Modular design keeping all architectures preserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class TNNOutput:
    """Output from TNN forward pass."""
    x_0: torch.Tensor  # Updated node features
    x_1: torch.Tensor  # Updated edge features
    x_2: Optional[torch.Tensor] = None  # Updated 2-cell features


class CWNLayer(nn.Module):
    """Cell Weisfeiler Network Layer."""
    def __init__(self, in_channels_0, in_channels_1, in_channels_2, out_channels, dropout=0.0):
        super().__init__()
        self.node_mlp = nn.Sequential(nn.Linear(in_channels_0 + in_channels_1, out_channels), nn.Dropout(dropout), nn.Linear(out_channels, out_channels))
        self.edge_mlp = nn.Sequential(nn.Linear(in_channels_1 + in_channels_0 + in_channels_2, out_channels), nn.Dropout(dropout), nn.Linear(out_channels, out_channels))
        self.cell_mlp = nn.Sequential(nn.Linear(in_channels_2 + in_channels_1, out_channels), nn.Dropout(dropout), nn.Linear(out_channels, out_channels))

    def forward(self, x_0, x_1, x_2, incidence_1, incidence_2):
        B1 = incidence_1.to_dense() if incidence_1.is_sparse else incidence_1
        x_0_new = self.node_mlp(torch.cat([x_0, B1 @ x_1], dim=-1))
        if x_2 is not None and incidence_2 is not None:
            B2 = incidence_2.to_dense() if incidence_2.is_sparse else incidence_2
            x_1_new = self.edge_mlp(torch.cat([x_1, B1.T @ x_0, B2 @ x_2], dim=-1))
            x_2_new = self.cell_mlp(torch.cat([x_2, B2.T @ x_1], dim=-1))
        else:
            x_1_new = self.edge_mlp(torch.cat([x_1, B1.T @ x_0, torch.zeros_like(x_1)], dim=-1))
            x_2_new = None
        return x_0_new, x_1_new, x_2_new


class SCNLayer(nn.Module):
    """Simplicial Convolutional Network Layer."""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv_0 = nn.Linear(in_channels, out_channels)
        self.conv_1 = nn.Linear(in_channels, out_channels)
        self.conv_2 = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_0, x_1, x_2, L0, L1, L2):
        def norm_L(L):
            L = L.to_dense() if L.is_sparse else L
            return L + torch.eye(L.shape[0], device=L.device)
        x_0_new = self.dropout(F.relu(self.conv_0(norm_L(L0) @ x_0)))
        x_1_new = self.dropout(F.relu(self.conv_1(norm_L(L1) @ x_1)))
        x_2_new = self.dropout(F.relu(self.conv_2(norm_L(L2) @ x_2))) if x_2 is not None else None
        return x_0_new, x_1_new, x_2_new


class HypergraphLayer(nn.Module):
    """Basic Hypergraph Neural Network Layer."""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.node_transform = nn.Linear(in_channels, out_channels)
        self.hyperedge_transform = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_0, x_1, incidence_1):
        H = incidence_1.to_dense() if incidence_1.is_sparse else incidence_1
        D_v = H.sum(dim=1, keepdim=True).clamp(min=1)
        D_e = H.sum(dim=0, keepdim=True).clamp(min=1)
        H_norm = H / torch.sqrt(D_v) / torch.sqrt(D_e)
        x_0_new = self.dropout(x_0 + self.node_transform(H_norm @ (H_norm.T @ x_0)))
        x_1_new = self.dropout(self.hyperedge_transform(H_norm.T @ x_0))
        return x_0_new, x_1_new


class HypergraphAttentionLayer(nn.Module):
    """Attention-based Hypergraph Layer."""
    def __init__(self, in_channels, out_channels, num_heads=4, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.node_to_edge_attn = nn.MultiheadAttention(out_channels, num_heads, dropout=dropout, batch_first=True)
        self.edge_to_node_attn = nn.MultiheadAttention(out_channels, num_heads, dropout=dropout, batch_first=True)
        self.node_proj = nn.Linear(in_channels, out_channels)
        self.edge_proj = nn.Linear(in_channels, out_channels)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_0, x_1, incidence_1):
        H = incidence_1.to_dense() if incidence_1.is_sparse else incidence_1
        x_0_p, x_1_p = self.node_proj(x_0), self.edge_proj(x_1)
        x_1_u, _ = self.node_to_edge_attn(x_1_p.unsqueeze(0), x_0_p.unsqueeze(0), x_0_p.unsqueeze(0), attn_mask=(H.T == 0).unsqueeze(0).expand(self.num_heads, -1, -1))
        x_1_u = self.norm1(x_1_p + self.dropout(x_1_u.squeeze(0)))
        x_0_u, _ = self.edge_to_node_attn(x_0_p.unsqueeze(0), x_1_u.unsqueeze(0), x_1_u.unsqueeze(0), attn_mask=(H == 0).unsqueeze(0).expand(self.num_heads, -1, -1))
        x_0_u = self.norm2(x_0_p + self.dropout(x_0_u.squeeze(0)))
        return x_0_u, x_1_u


class HypergraphGPSLayer(nn.Module):
    """SOTA Hypergraph GPS Layer with Gated Residuals and return-trip message passing.

    Message flow per layer:
      1. Nodes → Hyperedges:  x_1' = x_1 + W_e(B^T x_0)
      2. Hyperedges → Nodes:  x_0' = x_0 + gate_local * W_v(B x_1')
      3. Return trip:         x_1'' = x_1' + gate_return * W_e2(B^T x_0')
         Hyperedges see the updated node features after they processed their
         first message — useful for joint query-chunk graphs where queries
         need to "send back" their updated context to the cells they activated.
      4. Global attention (when n < 500) + FFN on nodes.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.node_transform = nn.Linear(dim, dim)
        self.hyperedge_transform = nn.Linear(dim, dim)
        self.hyperedge_return = nn.Linear(dim, dim)   # return-trip projection
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.gate_local = nn.Parameter(torch.zeros(1))
        self.gate_global = nn.Parameter(torch.zeros(1))
        self.gate_return = nn.Parameter(torch.zeros(1))  # init 0 → tanh(0)=0, no effect at start
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_0, x_1, incidence_1):
        # incidence_1: (num_nodes, num_hyperedges)
        if incidence_1.is_sparse:
            D_v = torch.sparse.sum(incidence_1, dim=1).to_dense().view(-1, 1).clamp(min=1)
            D_e = torch.sparse.sum(incidence_1, dim=0).to_dense().view(1, -1).clamp(min=1)
            msg_to_he = torch.sparse.mm(incidence_1.t(), x_0 / torch.sqrt(D_v)) / torch.sqrt(D_e.t())
            x_1_new = x_1 + self.dropout(self.hyperedge_transform(msg_to_he))
            msg_to_node = torch.sparse.mm(incidence_1, x_1_new / torch.sqrt(D_e.t())) / torch.sqrt(D_v)
        else:
            D_v = incidence_1.sum(dim=1, keepdim=True).clamp(min=1)
            D_e = incidence_1.sum(dim=0, keepdim=True).clamp(min=1)
            H_norm = incidence_1 / torch.sqrt(D_v) / torch.sqrt(D_e)
            x_1_new = x_1 + self.dropout(self.hyperedge_transform(H_norm.T @ x_0))
            msg_to_node = H_norm @ x_1_new

        # Step 2: hyperedges → nodes (gated)
        x_0_local = self.norm1(x_0 + torch.tanh(self.gate_local) * self.dropout(self.node_transform(msg_to_node)))

        # Step 3: return trip — updated nodes → hyperedges (gated, small weight initially)
        if incidence_1.is_sparse:
            ret_msg = torch.sparse.mm(incidence_1.t(), x_0_local / torch.sqrt(D_v)) / torch.sqrt(D_e.t())
        else:
            ret_msg = H_norm.T @ x_0_local
        x_1_new = x_1_new + torch.tanh(self.gate_return) * self.dropout(self.hyperedge_return(ret_msg))

        # Step 4: global attention (small graphs only) + FFN
        if x_0.shape[0] < 500:
            attn_out, _ = self.attention(x_0_local.unsqueeze(0), x_0_local.unsqueeze(0), x_0_local.unsqueeze(0))
            x_0_global = self.norm2(x_0_local + torch.tanh(self.gate_global) * self.dropout(attn_out.squeeze(0)))
        else:
            x_0_global = self.norm2(x_0_local)

        return x_0_global + self.dropout(self.ffn(x_0_global)), x_1_new


class TNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, model_type="hypergraph_gps", dropout=0.1, num_heads=4):
        super().__init__()
        self.model_type = model_type
        self.input_proj_0 = nn.Linear(in_channels, hidden_channels)
        self.input_proj_1 = nn.Linear(in_channels, hidden_channels)
        self.input_proj_2 = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if model_type == "cwn": self.layers.append(CWNLayer(hidden_channels, hidden_channels, hidden_channels, hidden_channels, dropout))
            elif model_type == "scn": self.layers.append(SCNLayer(hidden_channels, hidden_channels, dropout))
            elif model_type == "hypergraph": self.layers.append(HypergraphLayer(hidden_channels, hidden_channels, dropout))
            elif model_type == "hypergraph_attn": self.layers.append(HypergraphAttentionLayer(hidden_channels, hidden_channels, num_heads, dropout))
            elif model_type == "hypergraph_gps": self.layers.append(HypergraphGPSLayer(hidden_channels, num_heads, dropout))
            else: raise ValueError(f"Unknown model type: {model_type}")
        self.output_proj_0 = nn.Linear(hidden_channels, out_channels)
        self.output_proj_1 = nn.Linear(hidden_channels, out_channels)
        self.output_proj_2 = nn.Linear(hidden_channels, out_channels)
        # No special init — the score_gate (tanh(0)=0) protects from random init.
        # Normal (Xavier) init allows TNN to learn larger corrections.

    def forward(self, x_0, x_1, x_2, incidence_1, **kwargs):
        device = self.input_proj_0.weight.device
        x_0, x_1 = x_0.to(device), x_1.to(device)
        x_0_orig, x_0 = x_0, self.input_proj_0(x_0)
        x_1 = self.input_proj_1(x_1)
        for layer in self.layers:
            if self.model_type == "cwn": x_0, x_1, x_2 = layer(x_0, x_1, x_2, incidence_1, kwargs.get('incidence_2'))
            elif self.model_type == "scn": x_0, x_1, x_2 = layer(x_0, x_1, x_2, kwargs.get('L0'), kwargs.get('L1'), kwargs.get('L2'))
            else: x_0, x_1 = layer(x_0, x_1, incidence_1)
        x_0 = self.output_proj_0(x_0)
        return TNNOutput(x_0=x_0 + x_0_orig if x_0.shape == x_0_orig.shape else x_0, x_1=x_1)

    def forward_from_lifted(self, lifted) -> TNNOutput:
        return self.forward(lifted.x_0, lifted.x_1, lifted.x_2, lifted.incidence_1, incidence_2=lifted.incidence_2, L0=lifted.hodge_laplacian_0, L1=lifted.hodge_laplacian_1, L2=lifted.hodge_laplacian_2)