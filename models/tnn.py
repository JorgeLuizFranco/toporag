"""
Topological Neural Networks (TNN) for TopoRAG.

Implements message passing on cell/simplicial complexes following
TopoModelX conventions. These models propagate information across
topological structures using incidence matrices.

Key models:
- CWN (Cell Weisfeiler Network): Message passing on cell complexes
- SCN (Simplicial Convolutional Network): Message passing on simplicial complexes
- HypergraphNN: Message passing on hypergraphs

References:
- Bodnar et al. "Weisfeiler and Lehman Go Topological" (CWN)
- Bunch et al. "Simplicial 2-Complex Convolutional Neural Networks" (SCN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class TNNOutput:
    """Output from TNN forward pass."""
    x_0: torch.Tensor  # Updated node features
    x_1: torch.Tensor  # Updated edge features
    x_2: Optional[torch.Tensor] = None  # Updated 2-cell features


class CWNLayer(nn.Module):
    """
    Cell Weisfeiler Network Layer.

    Performs message passing on cell complexes using:
    - Node → Edge messages (via incidence_1)
    - Edge → Node messages (via incidence_1.T)
    - Edge → 2-cell messages (via incidence_2)
    - 2-cell → Edge messages (via incidence_2.T)

    Following the CWN paper architecture.
    """

    def __init__(
        self,
        in_channels_0: int,
        in_channels_1: int,
        in_channels_2: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Node update MLPs
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels_0 + in_channels_1, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
        )

        # Edge update MLPs
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels_1 + in_channels_0 + in_channels_2, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
        )

        # 2-cell update MLPs
        self.cell_mlp = nn.Sequential(
            nn.Linear(in_channels_2 + in_channels_1, out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_channels, out_channels),
        )

    def forward(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        x_2: Optional[torch.Tensor],
        incidence_1: torch.Tensor,
        incidence_2: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x_0: (num_nodes, in_channels_0) node features
            x_1: (num_edges, in_channels_1) edge features
            x_2: (num_2cells, in_channels_2) 2-cell features (optional)
            incidence_1: (num_nodes, num_edges) node-edge incidence
            incidence_2: (num_edges, num_2cells) edge-2cell incidence (optional)

        Returns:
            Updated (x_0, x_1, x_2)
        """
        # Convert sparse to dense if needed
        B1 = incidence_1.to_dense() if incidence_1.is_sparse else incidence_1

        # Normalize incidence for message passing
        B1_norm = B1 / (B1.sum(dim=0, keepdim=True).clamp(min=1))

        # Edge → Node messages (aggregate edge features to nodes)
        msg_edge_to_node = B1 @ x_1  # (num_nodes, dim)

        # Node → Edge messages (aggregate node features to edges)
        msg_node_to_edge = B1.T @ x_0  # (num_edges, dim)

        # Update nodes
        x_0_new = self.node_mlp(torch.cat([x_0, msg_edge_to_node], dim=-1))

        # Handle 2-cells
        if x_2 is not None and incidence_2 is not None:
            B2 = incidence_2.to_dense() if incidence_2.is_sparse else incidence_2

            # 2-cell → Edge messages
            msg_cell_to_edge = B2 @ x_2  # (num_edges, dim)

            # Edge → 2-cell messages
            msg_edge_to_cell = B2.T @ x_1  # (num_2cells, dim)

            # Update edges (with both node and cell messages)
            x_1_new = self.edge_mlp(
                torch.cat([x_1, msg_node_to_edge, msg_cell_to_edge], dim=-1)
            )

            # Update 2-cells
            x_2_new = self.cell_mlp(torch.cat([x_2, msg_edge_to_cell], dim=-1))
        else:
            # No 2-cells, just use node messages
            x_1_new = self.edge_mlp(
                torch.cat([x_1, msg_node_to_edge, torch.zeros_like(x_1)], dim=-1)
            )
            x_2_new = None

        return x_0_new, x_1_new, x_2_new


class SCNLayer(nn.Module):
    """
    Simplicial Convolutional Network Layer.

    Simpler message passing using Hodge Laplacians.
    L_k = B_{k+1} @ B_{k+1}^T + B_k^T @ B_k
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv_0 = nn.Linear(in_channels, out_channels)
        self.conv_1 = nn.Linear(in_channels, out_channels)
        self.conv_2 = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        x_2: Optional[torch.Tensor],
        hodge_laplacian_0: torch.Tensor,
        hodge_laplacian_1: torch.Tensor,
        hodge_laplacian_2: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward using Hodge Laplacians."""
        L0 = hodge_laplacian_0.to_dense() if hodge_laplacian_0.is_sparse else hodge_laplacian_0
        L1 = hodge_laplacian_1.to_dense() if hodge_laplacian_1.is_sparse else hodge_laplacian_1

        # Normalize Laplacians (add identity for stability)
        L0 = L0 + torch.eye(L0.shape[0], device=L0.device)
        L1 = L1 + torch.eye(L1.shape[0], device=L1.device)

        # Spectral convolution
        x_0_new = F.relu(self.conv_0(L0 @ x_0))
        x_1_new = F.relu(self.conv_1(L1 @ x_1))

        x_0_new = self.dropout(x_0_new)
        x_1_new = self.dropout(x_1_new)

        if x_2 is not None and hodge_laplacian_2 is not None:
            L2 = hodge_laplacian_2.to_dense() if hodge_laplacian_2.is_sparse else hodge_laplacian_2
            L2 = L2 + torch.eye(L2.shape[0], device=L2.device)
            x_2_new = F.relu(self.conv_2(L2 @ x_2))
            x_2_new = self.dropout(x_2_new)
        else:
            x_2_new = None

        return x_0_new, x_1_new, x_2_new


class HypergraphLayer(nn.Module):
    """
    Hypergraph Neural Network Layer.

    For k-NN hypergraph lifting. Uses node-hyperedge incidence only.
    Similar to UniGCN/HyperGCN.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_transform = nn.Linear(in_channels, out_channels)
        self.hyperedge_transform = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        incidence_1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x_0: Node features
            x_1: Hyperedge features
            incidence_1: Node-hyperedge incidence

        Returns:
            Updated (x_0, x_1)
        """
        H = incidence_1.to_dense() if incidence_1.is_sparse else incidence_1

        # Degree normalization
        D_v = H.sum(dim=1, keepdim=True).clamp(min=1)  # Node degree
        D_e = H.sum(dim=0, keepdim=True).clamp(min=1)  # Hyperedge degree

        # Normalize incidence
        H_norm = H / torch.sqrt(D_v) / torch.sqrt(D_e)

        # Message passing: Node → Hyperedge → Node
        msg_to_hyperedge = H_norm.T @ x_0
        msg_to_node = H_norm @ msg_to_hyperedge

        # Update
        x_0_new = F.relu(self.node_transform(x_0 + msg_to_node))
        x_1_new = F.relu(self.hyperedge_transform(msg_to_hyperedge))

        x_0_new = self.dropout(x_0_new)
        x_1_new = self.dropout(x_1_new)

        return x_0_new, x_1_new


class TNN(nn.Module):
    """
    Topological Neural Network for TopoRAG.

    Combines multiple TNN layers for message passing on complexes.
    Supports different architectures: CWN, SCN, Hypergraph.

    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension
        out_channels: Output dimension
        num_layers: Number of message passing layers
        model_type: 'cwn', 'scn', or 'hypergraph'
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        model_type: str = "cwn",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_type = model_type
        self.num_layers = num_layers

        # Input projection
        self.input_proj_0 = nn.Linear(in_channels, hidden_channels)
        self.input_proj_1 = nn.Linear(in_channels, hidden_channels)
        self.input_proj_2 = nn.Linear(in_channels, hidden_channels)

        # TNN layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            if model_type == "cwn":
                self.layers.append(
                    CWNLayer(
                        hidden_channels, hidden_channels, hidden_channels,
                        hidden_channels, dropout
                    )
                )
            elif model_type == "scn":
                self.layers.append(
                    SCNLayer(hidden_channels, hidden_channels, dropout)
                )
            elif model_type == "hypergraph":
                self.layers.append(
                    HypergraphLayer(hidden_channels, hidden_channels, dropout)
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")

        # Output projection
        self.output_proj_0 = nn.Linear(hidden_channels, out_channels)
        self.output_proj_1 = nn.Linear(hidden_channels, out_channels)
        self.output_proj_2 = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        x_2: Optional[torch.Tensor],
        incidence_1: torch.Tensor,
        incidence_2: Optional[torch.Tensor] = None,
        hodge_laplacian_0: Optional[torch.Tensor] = None,
        hodge_laplacian_1: Optional[torch.Tensor] = None,
        hodge_laplacian_2: Optional[torch.Tensor] = None,
    ) -> TNNOutput:
        """
        Forward pass through TNN.

        Args:
            x_0, x_1, x_2: Features at each dimension
            incidence_1, incidence_2: Incidence matrices
            hodge_laplacian_*: Laplacians (for SCN)

        Returns:
            TNNOutput with updated features
        """
        # Store original for residual connection
        x_0_orig = x_0

        # Project inputs
        x_0 = self.input_proj_0(x_0)
        x_1 = self.input_proj_1(x_1)
        if x_2 is not None:
            x_2 = self.input_proj_2(x_2)

        # Message passing
        for layer in self.layers:
            if self.model_type == "cwn":
                x_0, x_1, x_2 = layer(x_0, x_1, x_2, incidence_1, incidence_2)
            elif self.model_type == "scn":
                x_0, x_1, x_2 = layer(
                    x_0, x_1, x_2,
                    hodge_laplacian_0, hodge_laplacian_1, hodge_laplacian_2
                )
            elif self.model_type == "hypergraph":
                x_0, x_1 = layer(x_0, x_1, incidence_1)
                x_2 = None

        # Output projection
        x_0 = self.output_proj_0(x_0)
        x_1 = self.output_proj_1(x_1)
        if x_2 is not None:
            x_2 = self.output_proj_2(x_2)

        # Residual connection for nodes (preserve original embedding quality)
        if x_0.shape == x_0_orig.shape:
            x_0 = x_0 + x_0_orig

        return TNNOutput(x_0=x_0, x_1=x_1, x_2=x_2)

    def forward_from_lifted(self, lifted) -> TNNOutput:
        """
        Convenience method to forward from LiftedTopology.

        Args:
            lifted: LiftedTopology object

        Returns:
            TNNOutput
        """
        return self.forward(
            x_0=lifted.x_0,
            x_1=lifted.x_1,
            x_2=lifted.x_2,
            incidence_1=lifted.incidence_1,
            incidence_2=lifted.incidence_2,
            hodge_laplacian_0=lifted.hodge_laplacian_0,
            hodge_laplacian_1=lifted.hodge_laplacian_1,
            hodge_laplacian_2=lifted.hodge_laplacian_2,
        )
