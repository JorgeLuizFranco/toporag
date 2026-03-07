"""
Link Predictors for TopoRAG.
Ensures all operations are device-consistent.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

class QueryCellLinkPredictor(nn.Module):
    def __init__(self, query_dim: int, cell_dim: int, hidden_dim: int = 128, scoring: str = "mlp"):
        super().__init__()
        self.scoring = scoring
        if scoring == "bilinear":
            self.weight = nn.Parameter(torch.randn(query_dim, cell_dim))
            nn.init.xavier_uniform_(self.weight)
        elif scoring == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(query_dim + cell_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, query_embeddings: torch.Tensor, cell_embeddings: torch.Tensor) -> torch.Tensor:
        params = list(self.parameters())
        device = params[0].device if params else query_embeddings.device
        query_embeddings = query_embeddings.to(device)
        cell_embeddings = cell_embeddings.to(device)
        
        if query_embeddings.dim() == 1:
            query_embeddings = query_embeddings.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        if self.scoring == "dot":
            scores = torch.matmul(query_embeddings, cell_embeddings.T)
        elif self.scoring == "bilinear":
            transformed = torch.matmul(query_embeddings, self.weight)
            scores = torch.matmul(transformed, cell_embeddings.T)
        elif self.scoring == "mlp":
            nq, nc = query_embeddings.shape[0], cell_embeddings.shape[0]
            q_exp = query_embeddings.unsqueeze(1).expand(-1, nc, -1)
            c_exp = cell_embeddings.unsqueeze(0).expand(nq, -1, -1)
            scores = self.mlp(torch.cat([q_exp, c_exp], dim=-1)).squeeze(-1)
        else:
            raise ValueError(f"Unknown scoring: {self.scoring}")

        return scores.squeeze(0) if squeeze else scores

class NCNLinkPredictor(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.query_transform = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.cell_transform = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        layers = []
        in_dim = hidden_dim * 3
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU() if i < num_layers - 1 else nn.Identity()])
            in_dim = out_dim
        self.score_mlp = nn.Sequential(*layers)

    def forward(self, q, c):
        params = list(self.parameters())
        device = params[0].device if params else q.device
        q, c = q.to(device), c.to(device)
        if q.dim() == 1: q, squeeze = q.unsqueeze(0), True
        else: squeeze = False
        q_f, c_f = self.query_transform(q), self.cell_transform(c)
        q_exp = q_f.unsqueeze(1).expand(-1, c.shape[0], -1)
        c_exp = c_f.unsqueeze(0).expand(q.shape[0], -1, -1)
        combined = torch.cat([q_exp, c_exp, q_exp * c_exp], dim=-1)
        scores = self.score_mlp(combined).squeeze(-1)
        return scores.squeeze(0) if squeeze else scores