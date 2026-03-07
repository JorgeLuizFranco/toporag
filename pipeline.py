"""
TopoRAG Pipeline: Optimized for GPU-accelerated SOTA performance.
- Pure PyTorch KNN (GPU)
- O(1) Cell Lookups
- Device consistency
"""

import json
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

from .lifting.base import Cell, LiftedTopology
from .lifting import KNNLifting, CycleLifting, CliqueLifting
from .speculative_queries import SpeculativeQueryGenerator, SpeculativeQuery
from .models.lp_tnn import LPTNN, LPTNNTrainer
from .utils.embedding import TextEmbeddingModel


@dataclass
class TopoRAGConfig:
    embed_model: str = "all-mpnet-v2"
    embed_dim: int = 768
    lifting_type: str = "knn"
    knn_k: int = 5
    max_cycle_length: int = 6
    max_clique_size: int = 5
    hidden_dim: int = 256
    cell_encoder_type: str = "deepset"
    link_predictor_type: str = "mlp"
    learning_rate: float = 1e-4
    num_negative_samples: int = 5
    batch_size: int = 32
    num_epochs: int = 10
    top_k_cells: int = 5
    output_dir: str = "./toporag_output"
    debug: bool = True


class TopoRAGPipeline:
    def __init__(self, config: TopoRAGConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_embedding_model()
        self._init_lifting()
        self._init_model()
        self.chunk_texts: List[str] = []
        self.chunk_features: Optional[torch.Tensor] = None
        self.complex: Optional[LiftedTopology] = None
        self.supervised_complex: Optional[LiftedTopology] = None
        self.queries_by_cell: Dict[int, List[SpeculativeQuery]] = {}

    def _init_embedding_model(self):
        self.embedder = TextEmbeddingModel(model_name=self.config.embed_model, device=str(self.device))
        if self.embedder.dimension: self.config.embed_dim = self.embedder.dimension

    def _init_lifting(self):
        if self.config.lifting_type == "knn": self.lifting = KNNLifting(k=self.config.knn_k)
        elif self.config.lifting_type == "cycle": self.lifting = CycleLifting(max_cycle_length=self.config.max_cycle_length)
        elif self.config.lifting_type == "clique": self.lifting = CliqueLifting(max_clique_size=self.config.max_clique_size)
        else: raise ValueError(f"Unknown lifting type: {self.config.lifting_type}")

    def _init_model(self):
        tnn_config = {"hidden_dim": self.config.hidden_dim, "num_layers": 2, "dropout": 0.1}
        self.model = LPTNN(
            embed_dim=self.config.embed_dim, 
            hidden_dim=self.config.hidden_dim, 
            cell_encoder_type=self.config.cell_encoder_type, 
            link_predictor_type=self.config.link_predictor_type, 
            tnn_config=tnn_config
        )
        self.model.to(self.device)

    def embed_texts(self, texts: List[str], is_query: bool = False) -> torch.Tensor:
        embeddings = self.embedder.encode(texts, is_query=is_query, show_progress=len(texts) > 10)
        return torch.as_tensor(embeddings, device=self.device)

    def build_chunk_graph(self, chunk_features: torch.Tensor, k: int = 10, threshold: float = 0.5) -> torch.Tensor:
        """SOTA GPU-Accelerated Graph Building."""
        # chunk_features: (N, D)
        N = chunk_features.shape[0]
        # Normalize for cosine similarity
        norm_feat = torch.nn.functional.normalize(chunk_features, p=2, dim=1)
        # Similarity matrix: (N, N)
        # Note: If N is very large (>20k), we should do this in batches. For 10k it fits in 8GB.
        sims = torch.mm(norm_feat, norm_feat.t())
        
        # Get top-k neighbors (excluding self)
        vals, indices = torch.topk(sims, k=min(k + 1, N), dim=1)
        
        edges = []
        for i in range(N):
            # indices[i, 0] is usually self (similarity 1.0)
            for j in range(1, indices.shape[1]):
                if vals[i, j] > threshold:
                    edges.extend([[i, indices[i, j].item()], [indices[i, j].item(), i]])
        
        if not edges: return torch.empty((2, 0), dtype=torch.long, device=self.device)
        return torch.tensor(edges, device=self.device).t().unique(dim=1)

    def process_chunks(self, chunk_texts: List[str]) -> LiftedTopology:
        self.chunk_texts = chunk_texts
        self.chunk_features = self.embed_texts(chunk_texts)
        edge_index = self.build_chunk_graph(self.chunk_features)
        from torch_geometric.data import Data
        data = Data(x=self.chunk_features, edge_index=edge_index).to(self.device)
        self.complex = self.lifting.lift(data).to(self.device)
        return self.complex

    def generate_speculative_queries(self, llm, num_queries_per_cell: int = 2) -> Dict[int, List[SpeculativeQuery]]:
        generator = SpeculativeQueryGenerator(llm=llm, num_queries_per_cell=num_queries_per_cell)
        self.queries_by_cell = generator.generate_for_complex(self.complex, self.chunk_texts)
        return self.queries_by_cell

    def build_supervised_complex(self) -> LiftedTopology:
        if not self.queries_by_cell: raise ValueError("Must generate speculative queries first")
        all_query_texts, query_to_cell = [], []
        for cell_id, queries in self.queries_by_cell.items():
            for q in queries:
                all_query_texts.append(q.query_text); query_to_cell.append(cell_id)
        
        query_features = self.embed_texts(all_query_texts, is_query=True)
        num_chunks = self.chunk_features.shape[0]
        new_x = torch.cat([self.chunk_features, query_features], dim=0)
        
        # O(1) Lookup Optimization
        all_cells = self.complex.get_all_higher_order_cells()
        cell_map = {c.cell_id: c for c in all_cells}
        
        query_edges = []
        for q_idx, cell_id in enumerate(query_to_cell):
            query_node_id = num_chunks + q_idx
            target_cell = cell_map.get(cell_id)
            if target_cell:
                for chunk_idx in target_cell.chunk_indices:
                    query_edges.extend([[query_node_id, chunk_idx], [chunk_idx, query_node_id]])
        
        edge_index = self.build_chunk_graph(self.chunk_features)
        if query_edges:
            edge_index = torch.cat([edge_index, torch.tensor(query_edges, device=self.device).T], dim=1)
            
        from torch_geometric.data import Data
        data = Data(x=new_x, edge_index=edge_index).to(self.device)
        self.supervised_complex = self.lifting.lift(data).to(self.device)
        return self.supervised_complex

    def train(self, num_epochs: Optional[int] = None) -> List[float]:
        if self.supervised_complex is None: self.build_supervised_complex()
        num_chunks = self.chunk_features.shape[0]
        q_indices, pos_indices = [], []
        all_cells = self.supervised_complex.get_all_higher_order_cells()
        # Optimization: Map cell_id to index in all_cells once
        cell_id_to_idx = {c.cell_id: i for i, c in enumerate(all_cells)}
        
        q_count = 0
        for cell_id, queries in self.queries_by_cell.items():
            if cell_id in cell_id_to_idx:
                idx = cell_id_to_idx[cell_id]
                for _ in queries:
                    q_indices.append(num_chunks + q_count); pos_indices.append(idx); q_count += 1
        
        trainer = LPTNNTrainer(model=self.model, learning_rate=self.config.learning_rate)
        losses = []
        batch_size = self.config.batch_size
        for epoch in range(num_epochs or self.config.num_epochs):
            epoch_loss, num_batches = 0.0, 0
            indices = np.random.permutation(len(q_indices))
            for i in range(0, len(q_indices), batch_size):
                batch_idxs = indices[i:i+batch_size]
                b_q = [q_indices[idx] for idx in batch_idxs]
                b_pos = [pos_indices[idx] for idx in batch_idxs]
                loss = trainer.train_step(self.supervised_complex, b_q, b_pos)
                if i == 0: print(f"      Epoch {epoch+1} Batch 0 Grad Norm: {trainer.get_grad_norm():.6f}")
                epoch_loss += loss; num_batches += 1
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            losses.append(avg_loss); print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
        return losses

    def retrieve(self, query_text: str, top_k: Optional[int] = None) -> Tuple[List[Cell], List[int], torch.Tensor]:
        device = self.device
        self.model.to(device); self.model.eval()
        query_emb = self.embed_texts([query_text], is_query=True)[0]
        x_0_base = self.complex.x_0.to(device)
        
        # Situating via GPU
        sims = torch.matmul(query_emb.unsqueeze(0), x_0_base.t()).squeeze(0)
        _, top_neighbors = torch.topk(sims, k=min(5, x_0_base.shape[0]))
        
        x_aug = torch.cat([x_0_base, query_emb.unsqueeze(0)], dim=0)
        q_idx = x_0_base.shape[0]
        new_edges = []
        for n_idx in top_neighbors.tolist(): new_edges.extend([[q_idx, n_idx], [n_idx, q_idx]])
        
        # Use adjacency_0 if available or build base edges
        edge_index_base = self.build_chunk_graph(x_0_base)
        edge_index_aug = torch.cat([edge_index_base, torch.tensor(new_edges, device=device).T], dim=1)
        
        from torch_geometric.data import Data
        data_aug = Data(x=x_aug, edge_index=edge_index_aug).to(device)
        complex_aug = self.lifting.lift(data_aug).to(device)
        
        with torch.no_grad():
            ho_cells = complex_aug.get_all_higher_order_cells()
            target_cells = [c for c in ho_cells if q_idx not in c.chunk_indices]
            if not target_cells: target_cells = self.complex.get_all_higher_order_cells()
            scores = self.model(complex_aug, torch.tensor([q_idx], device=device), target_cells)
            
        k = min(top_k or self.config.top_k_cells, len(target_cells))
        top_scores, top_indices = torch.topk(scores.flatten(), k=k)
        top_cells = [target_cells[i] for i in top_indices.tolist()]
        
        chunk_indices = []
        for cell in top_cells: chunk_indices.extend(list(cell.chunk_indices))
        seen, unique = set(), []
        for idx in chunk_indices:
            if idx not in seen and idx < q_idx: seen.add(idx); unique.append(idx)
        return top_cells, unique, top_scores

    def get_context(self, query_text: str, top_k: Optional[int] = None) -> str:
        _, chunk_indices, _ = self.retrieve(query_text, top_k)
        return "\n\n".join([f"[{i + 1}] {self.chunk_texts[idx]}" for i, idx in enumerate(chunk_indices)])

    def save(self, output_dir: Optional[str] = None):
        out = Path(output_dir or self.config.output_dir); out.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out / "model.pt")
        with open(out / "config.json", "w") as f: json.dump(vars(self.config), f, indent=2)
        if self.chunk_texts:
            with open(out / "chunks.json", "w") as f: json.dump(self.chunk_texts, f)
        if self.chunk_features is not None: torch.save(self.chunk_features, out / "chunk_features.pt")
        if self.queries_by_cell:
            from .speculative_queries import save_queries
            save_queries(self.queries_by_cell, str(out / "queries.json"))
        print(f"Saved pipeline to {out}")

    def load(self, output_dir: str):
        out = Path(output_dir)
        with open(out / "config.json") as f: 
            cfg_dict = json.load(f)
            self.config = TopoRAGConfig(**cfg_dict)
        self.model.load_state_dict(torch.load(out / "model.pt"))
        self.model.to(self.device)
        if (out / "chunks.json").exists():
            with open(out / "chunks.json") as f: self.chunk_texts = json.load(f)
        if (out / "chunk_features.pt").exists():
            self.chunk_features = torch.load(out / "chunk_features.pt").to(self.device)
        if (out / "queries.json").exists():
            from .speculative_queries import load_queries
            self.queries_by_cell = load_queries(str(out / "queries.json"))
        if self.chunk_texts:
            self.process_chunks(self.chunk_texts)
        print(f"Loaded pipeline from {out}")