"""
TopoRAG: Topological Retrieval-Augmented Generation

Main pipeline that integrates:
1. LP-RAG style graph construction
2. Static lifting (cycle, clique, k-NN)
3. TNN message passing
4. Query-cell link prediction for retrieval
5. Speculative query generation

Usage:
    from toporag import TopoRAG, TopoRAGConfig

    config = TopoRAGConfig(lifting="cycle", tnn_type="cwn")
    model = TopoRAG(config)

    # Build from chunks
    model.build_from_chunks(chunks, chunk_to_doc)

    # Retrieve for query
    results = model.retrieve("What is the relationship between X and Y?")
"""

import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from torch_geometric.data import Data

from .graph.chunk_graph import ChunkGraphBuilder, ChunkGraphConfig, add_query_nodes_to_graph
from .lifting import CycleLifting, CliqueLifting, KNNHypergraphLifting, LiftedTopology
from .models.tnn import TNN, TNNOutput
from .models.gps import GPS
from .utils.embedding import TextEmbeddingModel


@dataclass
class TopoRAGConfig:
    """Configuration for TopoRAG."""

    # Embedding
    embed_model: str = "all-mpnet-v2"
    embed_dim: int = 768

    # Graph construction
    intra_doc_k: int = 5
    inter_doc_k: int = 10
    similarity_threshold: float = 0.3

    # Lifting
    lifting: str = "knn"  # 'cycle', 'clique', 'knn'
    max_cycle_length: int = 6
    max_clique_size: int = 4
    knn_k: int = 10

    # GPS (Graph Transformer - like LP-RAG)
    use_gps: bool = True  # Process graph with GPS before TNN
    num_gps_layers: int = 2
    gps_heads: int = 4

    # TNN
    tnn_type: str = "hypergraph_gps"  # 'cwn', 'scn', 'hypergraph', 'hypergraph_attn', 'hypergraph_gps'
    hidden_dim: int = 768  # Match embed_dim to avoid information loss
    num_tnn_layers: int = 2
    dropout: float = 0.1
    use_tnn: bool = True  # Set to False to skip TNN and use raw embeddings
    use_residual: bool = True  # Add residual connection to preserve embedding quality

    # Training
    learning_rate: float = 5e-4
    num_epochs: int = 10
    batch_size: int = 32
    num_negative_samples: int = 5

    # Retrieval
    top_k: int = 5

    # Debug
    debug: bool = False

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TopoRAG(nn.Module):
    """
    TopoRAG: Topological Retrieval-Augmented Generation.

    Combines:
    - LP-RAG graph construction (intra/inter document edges)
    - Static lifting to cell complexes / hypergraphs
    - TNN message passing for feature refinement
    - Query-cell link prediction for retrieval
    """

    def __init__(self, config: Optional[TopoRAGConfig] = None):
        super().__init__()
        self.config = config or TopoRAGConfig()
        self.device = torch.device(self.config.device)

        # Initialize embedding model
        self.embedder = TextEmbeddingModel(
            model_name=self.config.embed_model,
            device=self.config.device,
        )
        self.config.embed_dim = self.embedder.dimension

        # Initialize graph builder
        graph_config = ChunkGraphConfig(
            intra_doc_k=self.config.intra_doc_k,
            inter_doc_k=self.config.inter_doc_k,
            similarity_threshold=self.config.similarity_threshold,
        )
        self.graph_builder = ChunkGraphBuilder(graph_config)

        # Initialize lifting
        if self.config.lifting == "cycle":
            self.lifting = CycleLifting(
                max_cycle_length=self.config.max_cycle_length,
            )
        elif self.config.lifting == "clique":
            self.lifting = CliqueLifting(
                max_clique_size=self.config.max_clique_size,
            )
        elif self.config.lifting == "knn":
            self.lifting = KNNHypergraphLifting(
                k=self.config.knn_k,
            )
        else:
            raise ValueError(f"Unknown lifting: {self.config.lifting}")

        # Initialize GPS (Graph Transformer - processes LP-RAG graph)
        if self.config.use_gps:
            self.gps = GPS(
                in_channels=self.config.embed_dim,
                hidden_channels=self.config.hidden_dim,
                out_channels=self.config.embed_dim,  # Output same dim for residual
                num_layers=self.config.num_gps_layers,
                num_heads=self.config.gps_heads,
                dropout=self.config.dropout,
            )
        else:
            self.gps = None

        # Initialize TNN
        # Output same dimension as input to preserve embedding space for cosine similarity
        self.tnn = TNN(
            in_channels=self.config.embed_dim,
            hidden_channels=self.config.hidden_dim,
            out_channels=self.config.embed_dim,  # Preserve dimension for cosine similarity!
            num_layers=self.config.num_tnn_layers,
            model_type=self.config.tnn_type,
            dropout=self.config.dropout,
        )

        # Query encoder - projects query to same space as TNN output
        # For training with link predictor
        # Use a single layer with residual-like initialization to prevent scrambling
        self.query_encoder = nn.Linear(self.config.embed_dim, self.config.embed_dim)
        with torch.no_grad():
            self.query_encoder.weight.copy_(torch.eye(self.config.embed_dim))
            self.query_encoder.bias.zero_()

        # Gating mechanism: learns to combine GPS (LP-RAG) and TNN (TopoRAG) outputs
        # gate = sigmoid(W @ x + b) → 0 = pure LP-RAG, 1 = pure TopoRAG
        self.topo_gate = nn.Sequential(
            nn.Linear(self.config.embed_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid(),
        )
        # Initialize last layer bias to a large negative value so gate starts near 0
        nn.init.constant_(self.topo_gate[-2].bias, -3.0)
        nn.init.zeros_(self.topo_gate[-2].weight)

        # Learnable branch scales
        self.topo_scale = nn.Parameter(torch.ones([]))
        self.base_scale = nn.Parameter(torch.ones([]))

        # Storage
        self.chunks: List[str] = []
        self.chunk_to_doc: List[int] = []
        self.graph: Optional[Data] = None
        self.lifted: Optional[LiftedTopology] = None

        self.to(self.device)

    def build_from_chunks(
        self,
        chunks: List[str],
        chunk_to_doc: Optional[List[int]] = None,
    ) -> LiftedTopology:
        """
        Build topological structure from chunks.

        Args:
            chunks: List of text chunks
            chunk_to_doc: Optional mapping chunk_idx -> doc_idx
                         (if None, treats all chunks as one document)

        Returns:
            LiftedTopology with the constructed complex
        """
        self.chunks = chunks
        self.chunk_to_doc = chunk_to_doc or [0] * len(chunks)

        print(f"Embedding {len(chunks)} chunks...")
        embeddings = self.embedder.encode(chunks, show_progress=True)
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)

        print("Building chunk graph...")
        if chunk_to_doc is not None:
            self.graph = self.graph_builder.build_graph(embeddings, chunk_to_doc)
        else:
            self.graph = self.graph_builder.build_graph_simple(embeddings)

        print(f"  Graph has {self.graph.edge_index.shape[1]} edges")

        print(f"Applying {self.config.lifting} lifting...")
        self.lifted = self.lifting.lift(self.graph)

        print(f"  Nodes: {self.lifted.num_nodes}")
        print(f"  Edges: {self.lifted.num_edges}")
        print(f"  2-cells: {self.lifted.num_2cells}")

        # Move to device
        self.lifted = self.lifted.to(self.config.device)

        return self.lifted

    def augment_with_query(
        self,
        query_emb: torch.Tensor,
        k: int = 5,
        node_features: Optional[torch.Tensor] = None
    ) -> LiftedTopology:
        """
        Create a new topology with query added as a node.
        
        Args:
            query_emb: (dim,) query embedding
            k: number of nearest neighbors to connect to
            node_features: (num_nodes, dim) optional override for node features (e.g. from GPS)
            
        Returns:
            LiftedTopology with N+1 nodes
        """
        import torch.nn.functional as F
        
        # Use provided features or default to lifted.x_0
        node_embs = node_features if node_features is not None else self.lifted.x_0
        
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)
            
        # Compute cosine similarity
        # Note: we use detach() for finding neighbors to avoid backprop through k-NN selection
        # (unless we want differentiable k-NN which is hard)
        sims = F.cosine_similarity(query_emb.detach(), node_embs.detach())
        top_k_vals, top_k_indices = torch.topk(sims, min(k, len(node_embs)))
        neighbors = top_k_indices.tolist()
        
        # 2. Create new x_0
        x_0_new = torch.cat([node_embs, query_emb], dim=0)
        num_nodes_new = x_0_new.shape[0]
        query_node_idx = num_nodes_new - 1
        
        # 3. Create new incidence_1 (node-hyperedge)
        B1 = self.lifted.incidence_1
        num_edges_old = self.lifted.num_edges
        num_new_hyperedges = len(neighbors)
        
        # New hyperedges: For each neighbor, create a hyperedge [neighbor, query]
        # (This can be extended later to include neighbor's other neighbors)
        new_rows = []
        new_cols = []
        new_vals = []
        
        for i, neighbor_idx in enumerate(neighbors):
            he_idx = num_edges_old + i
            # Connection to neighbor
            new_rows.append(neighbor_idx)
            new_cols.append(he_idx)
            new_vals.append(1.0)
            # Connection to query
            new_rows.append(query_node_idx)
            new_cols.append(he_idx)
            new_vals.append(1.0)
            
        new_indices = torch.tensor([new_rows, new_cols], device=self.device)
        new_values = torch.tensor(new_vals, device=self.device)
        
        if B1.is_sparse:
            B1 = B1.coalesce()
            combined_indices = torch.cat([B1.indices(), new_indices], dim=1)
            combined_values = torch.cat([B1.values(), new_values])
            
            B1_new = torch.sparse_coo_tensor(
                combined_indices, 
                combined_values, 
                size=(num_nodes_new, num_edges_old + num_new_hyperedges)
            ).coalesce()
        else:
            # Dense fallback
            B1_new = torch.zeros((num_nodes_new, num_edges_old + num_new_hyperedges), device=self.device)
            B1_new[:self.lifted.num_nodes, :num_edges_old] = B1
            B1_new[new_rows, new_cols] = 1.0
            
        # 4. Create new x_1 (hyperedge features)
        neighbor_feats = node_embs[top_k_indices]
        query_feat_expanded = query_emb.expand(len(neighbors), -1)
        new_edge_feats = (neighbor_feats + query_feat_expanded) / 2
        
        if self.lifted.x_1 is not None:
            x_1_new = torch.cat([self.lifted.x_1, new_edge_feats], dim=0)
        else:
            x_1_new = new_edge_feats
            
        # 5. Handle incidence_2 (edges -> 2-cells)
        # Pad with zeros for new edges
        if self.lifted.incidence_2 is not None:
            B2 = self.lifted.incidence_2
            num_2cells = self.lifted.num_2cells
            
            if B2.is_sparse:
                B2 = B2.coalesce()
                B2_new = torch.sparse_coo_tensor(
                    B2.indices(),
                    B2.values(),
                    size=(num_edges_old + num_new_hyperedges, num_2cells)
                ).coalesce()
            else:
                B2_new = torch.zeros((num_edges_old + num_new_hyperedges, num_2cells), device=self.device)
                B2_new[:num_edges_old, :] = B2
        else:
            B2_new = None
            
        return LiftedTopology(
            x_0=x_0_new,
            x_1=x_1_new,
            x_2=self.lifted.x_2,
            incidence_1=B1_new,
            incidence_2=B2_new,
            num_nodes=num_nodes_new,
            num_edges=num_edges_old + num_new_hyperedges,
            num_2cells=self.lifted.num_2cells,
        )

    def retrieve_graph_interaction(
        self,
        query_text: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Retrieve using query-as-node graph interaction.
        """
        if self.lifted is None:
            raise RuntimeError("Must build graph first")
            
        # 1. Encode query
        # Use query_encoder to match training (encoder is identity-initialized, safe even if untrained)
        query_emb = self.encode_query(query_text, use_encoder=True)
            
        # 2. Apply GPS to existing chunks (if enabled)
        if self.config.use_gps and self.graph is not None:
             x_0 = self.forward_gps(self.lifted.x_0, self.graph.edge_index.to(self.device))
        else:
             x_0 = self.lifted.x_0

        # 3. Augment graph
        # Connect query to closest nodes based on CURRENT embedding state (x_0)
        augmented_lifted = self.augment_with_query(
            query_emb.squeeze(0), 
            k=5, 
            node_features=x_0
        )
        
        # 4. Run TNN on augmented graph
        tnn_output = self.tnn.forward_from_lifted(augmented_lifted)

        # 5. Get embeddings from both pathways
        # TopoRAG: after TNN message passing
        q_topo = tnn_output.x_0[-1:]  # Query after TNN
        c_topo = tnn_output.x_0[:-1]  # Chunks after TNN

        # LP-RAG style: GPS only (no TNN)
        q_gps = query_emb  # Query (no TNN)
        c_gps = x_0  # Chunks after GPS only

        # 6. Gated combination with RESIDUAL baseline preservation
        # Use learnable scaling factors to match magnitudes instead of restrictive LayerNorm
        if not hasattr(self, 'topo_scale'):
            self.topo_scale = nn.Parameter(torch.ones([]))
            self.base_scale = nn.Parameter(torch.ones([]))
            
        gate = self.topo_gate(q_topo.squeeze(0))
        
        # Combine branches with learnable relative scaling
        query_final = (1 - gate) * self.base_scale * query_emb + gate * self.topo_scale * q_topo
        candidate_embs = (1 - gate) * self.base_scale * self.lifted.x_0 + gate * self.topo_scale * c_topo

        if self.config.debug:
            print(f"DEBUG: q_topo mean abs: {q_topo.abs().mean():.6f}, q_gps mean abs: {q_gps.abs().mean():.6f}")
            print(f"DEBUG: gate value: {gate.item():.4f}")
            print(f"DEBUG: candidate_embs mean abs: {candidate_embs.abs().mean():.6f}")

        # 7. Score using cosine similarity on refined embeddings
        # This ensures the TNN is learning to shift the embeddings into a better space
        scores = self.score_cells(query_final, candidate_embs, use_cosine=True)
             
        # 7. Top-k
        k = min(top_k, len(self.chunks))
        top_scores, top_indices = torch.topk(scores, k)
        
        return {
            "chunks": top_indices.tolist(),
            "scores": top_scores.tolist(),
            "chunk_texts": [self.chunks[i] for i in top_indices.tolist() if i < len(self.chunks)]
        }

    def forward_gps(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Run GPS (Graph Transformer) on node features.

        Args:
            x: (num_nodes, embed_dim) node features
            edge_index: (2, num_edges) edge indices

        Returns:
            GPS-processed node features with residual connection
        """
        if self.gps is None:
            return x

        # Run GPS
        gps_out = self.gps(x, edge_index)

        # Residual connection
        if self.config.use_residual:
            return x + gps_out
        else:
            return gps_out

    def forward_tnn(self, lifted: LiftedTopology) -> TNNOutput:
        """Run TNN message passing on lifted topology."""
        return self.tnn.forward_from_lifted(lifted)

    def encode_query(self, query_text: str, use_encoder: bool = True) -> torch.Tensor:
        """Encode query text to embedding.

        Args:
            query_text: The query string
            use_encoder: If True, apply the learned query_encoder MLP.
                        If False, return raw embedding (for pure similarity retrieval).
        """
        with torch.no_grad():
            query_emb = self.embedder.encode([query_text], is_query=True)
            if not isinstance(query_emb, torch.Tensor):
                query_emb = torch.tensor(query_emb)
            # Clone to detach from inference mode
            query_emb = query_emb.clone().detach()

        if use_encoder:
            return self.query_encoder(query_emb.to(self.device))
        else:
            return query_emb.to(self.device)

    def score_cells(
        self,
        query_emb: torch.Tensor,
        cell_embeddings: torch.Tensor,
        use_cosine: bool = True,
    ) -> torch.Tensor:
        """Score query-cell pairs using normalized dot product.
        
        Args:
            query_emb: (1, dim) query embedding
            cell_embeddings: (num_cells, dim) cell embeddings
            use_cosine: Ignored, always use cosine for stability.
        """
        # Normalize for cosine similarity
        query_norm = query_emb / query_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cell_norm = cell_embeddings / cell_embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        
        if query_norm.dim() == 1:
            query_norm = query_norm.unsqueeze(0)
            
        # Standard contrastive scale (1/0.05 = 20.0)
        scores = (query_norm @ cell_norm.T).squeeze(0)
        return scores * 20.0

    def retrieve(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        retrieve_nodes: bool = True,  # NEW: retrieve at node level for 100% coverage
    ) -> Dict[str, Any]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query_text: Query string
            top_k: Number of chunks/cells to retrieve
            retrieve_nodes: If True, retrieve at node (chunk) level for 100% coverage.
                          If False, retrieve at cell level (may miss chunks not in cells).

        Returns:
            Dict with:
            - cells: List of retrieved cell indices (if retrieve_nodes=False)
            - chunks: List of chunk indices
            - scores: Retrieval scores
            - chunk_texts: List of chunk texts
        """
        if self.lifted is None:
            raise RuntimeError("Must call build_from_chunks first")

        top_k = top_k or self.config.top_k

        # Get node embeddings through GPS + TNN pipeline
        if self.config.use_gps or self.config.use_tnn:
            # Start with raw embeddings
            node_emb = self.lifted.x_0

            # Apply GPS (Graph Transformer) if enabled
            if self.config.use_gps and self.graph is not None:
                node_emb = self.forward_gps(node_emb, self.graph.edge_index.to(self.device))

            # Apply TNN if enabled
            if self.config.use_tnn:
                # Update lifted topology with GPS-processed features
                lifted_updated = LiftedTopology(
                    x_0=node_emb,
                    x_1=self.lifted.x_1,
                    x_2=self.lifted.x_2,
                    incidence_1=self.lifted.incidence_1,
                    incidence_2=self.lifted.incidence_2,
                    adjacency_0=self.lifted.adjacency_0,
                    adjacency_1=self.lifted.adjacency_1,
                    adjacency_2=self.lifted.adjacency_2,
                    hodge_laplacian_0=self.lifted.hodge_laplacian_0,
                    hodge_laplacian_1=self.lifted.hodge_laplacian_1,
                    hodge_laplacian_2=self.lifted.hodge_laplacian_2,
                    num_nodes=self.lifted.num_nodes,
                    num_edges=self.lifted.num_edges,
                    num_2cells=self.lifted.num_2cells,
                    cells=self.lifted.cells,
                    cell_to_nodes=self.lifted.cell_to_nodes,
                )
                tnn_output = self.forward_tnn(lifted_updated)
                node_embeddings = tnn_output.x_0
            else:
                node_embeddings = node_emb
        else:
            # Use raw embeddings
            node_embeddings = self.lifted.x_0

        # Encode query
        # If GPS/TNN are trained, use query_encoder to map query to same trained space
        # This ensures consistency between training (BCE loss) and inference
        if self.config.use_gps or self.config.use_tnn:
            query_emb = self.encode_query(query_text, use_encoder=True)
        else:
            query_emb = self.encode_query(query_text, use_encoder=False)

        if self.config.debug:
            print(f"DEBUG: node_embs mean abs: {node_embeddings.abs().mean():.6f}")
            print(f"DEBUG: query_emb mean abs: {query_emb.abs().mean():.6f}")

        if retrieve_nodes:

            # Score nodes (chunks) using normalized dot product
            scores = self.score_cells(query_emb, node_embeddings)

            # Get top-k chunks directly
            k = min(top_k, len(self.chunks))
            top_scores, top_indices = torch.topk(scores, k)

            return {
                "cells": [],  # Not used in node-level retrieval
                "chunks": top_indices.tolist(),
                "scores": top_scores.tolist(),
                "chunk_texts": [self.chunks[i] for i in top_indices.tolist() if i < len(self.chunks)],
            }
        else:
            # Original cell-level retrieval (may miss chunks not in cells)
            # Requires TNN to be enabled for cell embeddings
            if not self.config.use_tnn:
                raise RuntimeError("Cell-level retrieval requires use_tnn=True")

            # Need to run TNN if not already done
            if self.config.use_gps and self.graph is not None:
                node_emb = self.forward_gps(self.lifted.x_0, self.graph.edge_index.to(self.device))
                lifted_updated = LiftedTopology(
                    x_0=node_emb,
                    x_1=self.lifted.x_1,
                    x_2=self.lifted.x_2,
                    incidence_1=self.lifted.incidence_1,
                    incidence_2=self.lifted.incidence_2,
                    num_nodes=self.lifted.num_nodes,
                    num_edges=self.lifted.num_edges,
                    num_2cells=self.lifted.num_2cells,
                    cells=self.lifted.cells,
                    cell_to_nodes=self.lifted.cell_to_nodes,
                )
                tnn_output = self.forward_tnn(lifted_updated)
            else:
                tnn_output = self.forward_tnn(self.lifted)

            if self.config.lifting in ["cycle", "clique"] and tnn_output.x_2 is not None:
                cell_embeddings = tnn_output.x_2
                cells = self.lifted.cells
            else:
                cell_embeddings = tnn_output.x_1
                cells = self.lifted.cells

            if cell_embeddings.shape[0] == 0:
                return {
                    "cells": [],
                    "chunks": [],
                    "scores": [],
                    "chunk_texts": [],
                }

            # Score cells using normalized dot product
            scores = self.score_cells(query_emb, cell_embeddings)

            # Get top-k cells
            k = min(top_k, len(cells))
            top_scores, top_indices = torch.topk(scores, k)

            # Extract chunks from top cells
            retrieved_chunks = []
            for idx in top_indices.tolist():
                if idx in self.lifted.cell_to_nodes:
                    retrieved_chunks.extend(self.lifted.cell_to_nodes[idx])

            # Deduplicate while preserving order
            seen = set()
            unique_chunks = []
            for c in retrieved_chunks:
                if c not in seen:
                    seen.add(c)
                    unique_chunks.append(c)

            return {
                "cells": top_indices.tolist(),
                "chunks": unique_chunks,
                "scores": top_scores.tolist(),
                "chunk_texts": [self.chunks[i] for i in unique_chunks if i < len(self.chunks)],
            }

    def get_training_data(
        self,
        speculative_queries: Dict[int, List[str]],  # cell_idx -> queries
    ) -> List[Tuple[torch.Tensor, int]]:
        """
        Prepare training data from speculative queries.

        Args:
            speculative_queries: Dict mapping cell_idx to list of query strings

        Returns:
            List of (query_embedding, cell_idx) tuples
        """
        train_data = []

        for cell_idx, queries in speculative_queries.items():
            for query in queries:
                with torch.no_grad():
                    query_emb = self.embedder.encode([query], is_query=True)
                    if not isinstance(query_emb, torch.Tensor):
                        query_emb = torch.tensor(query_emb)
                    # Clone to detach from inference mode
                    query_emb = query_emb.clone().detach()
                train_data.append((query_emb.squeeze(0), cell_idx))

        return train_data

    def train_step(
        self,
        query_embs: torch.Tensor,
        positive_cell_indices: List[int],
        cell_embeddings: torch.Tensor,
    ) -> float:
        """
        Single training step.

        Args:
            query_embs: (batch_size, embed_dim) query embeddings
            positive_cell_indices: List of positive cell indices
            cell_embeddings: (num_cells, hidden_dim) cell embeddings

        Returns:
            Loss value
        """
        self.train()
        batch_size = query_embs.shape[0]
        num_cells = cell_embeddings.shape[0]

        # Encode queries
        query_hidden = self.query_encoder(query_embs.to(self.device))

        total_loss = 0.0

        for i in range(batch_size):
            q = query_hidden[i]
            pos_idx = positive_cell_indices[i]

            # Positive score
            pos_cell = cell_embeddings[pos_idx]
            pos_score = self.link_predictor(torch.cat([q, pos_cell]))

            # Negative scores
            neg_indices = torch.randint(0, num_cells, (self.config.num_negative_samples,))
            neg_indices = neg_indices[neg_indices != pos_idx][:self.config.num_negative_samples]

            if len(neg_indices) > 0:
                neg_cells = cell_embeddings[neg_indices]
                q_expanded = q.unsqueeze(0).expand(len(neg_indices), -1)
                neg_scores = self.link_predictor(
                    torch.cat([q_expanded, neg_cells], dim=-1)
                )

                # BCE loss
                pos_label = torch.ones(1, device=self.device)
                neg_labels = torch.zeros(len(neg_indices), device=self.device)

                scores = torch.cat([pos_score, neg_scores.squeeze(-1)])
                labels = torch.cat([pos_label, neg_labels])

                loss = nn.functional.binary_cross_entropy_with_logits(scores, labels)
                total_loss += loss

        return total_loss.item() / batch_size

    def save(self, path: str):
        """Save model state."""
        torch.save({
            "config": self.config,
            "state_dict": self.state_dict(),
            "chunks": self.chunks,
            "chunk_to_doc": self.chunk_to_doc,
        }, path)

    @classmethod
    def load(cls, path: str) -> "TopoRAG":
        """Load model from checkpoint."""
        checkpoint = torch.load(path)
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.chunks = checkpoint["chunks"]
        model.chunk_to_doc = checkpoint["chunk_to_doc"]
        return model
