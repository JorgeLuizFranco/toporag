"""
GPS (Graph Transformer) Layer for TopoRAG.

Processes the LP-RAG style graph before TNN lifting.
Similar to graph transformers used in LP-RAG/GFM-RAG.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GPSLayer(nn.Module):
    """
    GPS-style Graph Transformer Layer.

    Combines:
    1. Local message passing (GNN-style)
    2. Global attention (Transformer-style)

    Following the GPS paper architecture.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_edge_features: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Local message passing (GCN-style)
        self.local_conv = nn.Linear(hidden_dim, hidden_dim)

        # Global attention (multi-head)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

        # Use Identity instead of LayerNorm to preserve pre-trained feature scales
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.norm3 = nn.Identity()

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (num_nodes, hidden_dim) node features
            edge_index: (2, num_edges) edge indices
            edge_attr: Optional edge features

        Returns:
            Updated node features (num_nodes, hidden_dim)
        """
        num_nodes = x.shape[0]

        # 1. Local message passing (GCN-style aggregation)
        # Build adjacency
        adj = torch.zeros(num_nodes, num_nodes, device=x.device)
        adj[edge_index[0], edge_index[1]] = 1.0

        # Normalize
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1)
        adj_norm = adj / deg

        # Message passing
        local_out = self.local_conv(adj_norm @ x)
        x = self.norm1(x + self.dropout(local_out))

        # 2. Global attention (treat all nodes as one sequence)
        # Add batch dimension for attention
        x_unsqueezed = x.unsqueeze(0)  # (1, num_nodes, hidden_dim)

        attn_out, _ = self.attention(x_unsqueezed, x_unsqueezed, x_unsqueezed)
        attn_out = attn_out.squeeze(0)  # (num_nodes, hidden_dim)
        x = self.norm2(x + self.dropout(attn_out))

        # 3. FFN
        x = self.norm3(x + self.ffn(x))

        return x


class GPS(nn.Module):
    """
    GPS (Graph Transformer) module for TopoRAG.

    Processes the LP-RAG graph at node level before TNN lifting.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # GPS layers
        self.layers = nn.ModuleList([
            GPSLayer(hidden_channels, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_channels, out_channels)
        
        # Initialize output projection to near-zero to preserve pre-trained features
        with torch.no_grad():
            self.output_proj.weight.fill_(0.0)
            self.output_proj.bias.fill_(0.0)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (num_nodes, in_channels) node features
            edge_index: (2, num_edges) edge indices

        Returns:
            (num_nodes, out_channels) updated node features
        """
        # Project
        x = self.input_proj(x)

        # GPS layers
        for layer in self.layers:
            x = layer(x, edge_index)

        # Output projection
        x = self.output_proj(x)

        return x
