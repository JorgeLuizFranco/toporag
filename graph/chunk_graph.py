"""
Chunk Graph Construction for TopoRAG.

Follows LP-RAG approach:
- Intra-document edges: k-NN within same document
- Inter-document edges: k-NN across different documents

This creates a graph where chunks are connected based on:
1. Semantic similarity (embedding cosine similarity)
2. Document structure (chunks from same doc are more likely connected)
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops, to_undirected
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ChunkGraphConfig:
    """Configuration for chunk graph construction."""
    intra_doc_k: int = 5       # k-NN within same document
    inter_doc_k: int = 10      # k-NN across documents
    similarity_threshold: float = 0.0  # Min similarity for edge
    add_self_loops: bool = False


class ChunkGraphBuilder:
    """
    Builds a chunk graph following LP-RAG methodology.

    Creates edges based on:
    - Intra-document: Connect chunks within same document via k-NN
    - Inter-document: Connect chunks across documents via k-NN

    This captures both local (within-doc) and global (cross-doc) relationships.
    """

    def __init__(self, config: Optional[ChunkGraphConfig] = None):
        self.config = config or ChunkGraphConfig()

    def build_graph(
        self,
        chunk_embeddings: torch.Tensor,
        chunk_to_doc: List[int],  # Maps chunk_idx -> doc_idx
    ) -> Data:
        """
        Build chunk graph from embeddings and document assignments.

        Args:
            chunk_embeddings: (num_chunks, embed_dim) tensor
            chunk_to_doc: List mapping each chunk to its source document

        Returns:
            PyG Data object with:
            - x: chunk embeddings
            - edge_index: graph edges
            - edge_attr: edge weights (similarity scores)
        """
        # Use GPU for similarity computation, with batching for large graphs
        device = chunk_embeddings.device if chunk_embeddings.is_cuda else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        num_chunks = chunk_embeddings.shape[0]
        chunk_to_doc_t = torch.tensor(chunk_to_doc, device=device)

        # Normalize embeddings for cosine similarity
        embeddings_norm = torch.nn.functional.normalize(
            chunk_embeddings.to(device), p=2, dim=-1
        )

        intra_k = min(self.config.intra_doc_k, num_chunks - 1)
        inter_k = min(self.config.inter_doc_k, num_chunks - 1)

        # Batch size for row-wise processing (avoid OOM on large graphs)
        # Each batch needs: batch_size × num_chunks floats for similarity + masks
        mem_per_row = num_chunks * 4 * 3  # 3 tensors (sim, intra, inter) × float32
        try:
            free_mem = torch.cuda.mem_get_info(device)[0] if device.type == 'cuda' else 4e9
        except Exception:
            free_mem = 4e9
        batch_size = max(64, int(free_mem * 0.5 / mem_per_row))
        batch_size = min(batch_size, num_chunks)

        # Collect top-k results per row batch
        all_intra_vals, all_intra_idx = [], []
        all_inter_vals, all_inter_idx = [], []

        for start in range(0, num_chunks, batch_size):
            end = min(start + batch_size, num_chunks)
            # Similarity for this batch of rows against all columns
            sim_batch = torch.mm(embeddings_norm[start:end], embeddings_norm.T)  # (bs, n)

            # Document masks for this batch
            doc_batch = chunk_to_doc_t[start:end].unsqueeze(1) == chunk_to_doc_t.unsqueeze(0)  # (bs, n)
            intra_mask = doc_batch.float()
            inter_mask = 1.0 - intra_mask

            # Remove self-loops from intra
            diag_indices = torch.arange(end - start, device=device)
            offsets = torch.arange(start, end, device=device)
            intra_mask[diag_indices, offsets] = 0

            # Intra-doc top-k
            if intra_k > 0:
                intra_scores = sim_batch * intra_mask
                iv, ii = torch.topk(intra_scores, k=min(intra_k, intra_scores.shape[1]), dim=-1)
                all_intra_vals.append(iv.cpu())
                all_intra_idx.append(ii.cpu())

            # Inter-doc top-k
            if inter_k > 0:
                inter_scores = sim_batch * inter_mask
                ev, ei = torch.topk(inter_scores, k=min(inter_k, inter_scores.shape[1]), dim=-1)
                all_inter_vals.append(ev.cpu())
                all_inter_idx.append(ei.cpu())

            del sim_batch, intra_mask, inter_mask
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        # Concatenate batch results
        if intra_k > 0 and all_intra_vals:
            intra_topk_values = torch.cat(all_intra_vals, dim=0)
            intra_topk_indices = torch.cat(all_intra_idx, dim=0)
        else:
            intra_topk_values = torch.empty(num_chunks, 0)
            intra_topk_indices = torch.empty(num_chunks, 0, dtype=torch.long)

        if inter_k > 0 and all_inter_vals:
            inter_topk_values = torch.cat(all_inter_vals, dim=0)
            inter_topk_indices = torch.cat(all_inter_idx, dim=0)
        else:
            inter_topk_values = torch.empty(num_chunks, 0)
            inter_topk_indices = torch.empty(num_chunks, 0, dtype=torch.long)

        chunk_embeddings = chunk_embeddings.cpu()

        # Build edge list
        edges = []
        edge_weights = []

        for src in range(num_chunks):
            # Intra-document edges
            for j, dst in enumerate(intra_topk_indices[src]):
                weight = intra_topk_values[src, j].item()
                if weight > self.config.similarity_threshold:
                    edges.append([src, dst.item()])
                    edge_weights.append(weight)

            # Inter-document edges
            for j, dst in enumerate(inter_topk_indices[src]):
                weight = inter_topk_values[src, j].item()
                if weight > self.config.similarity_threshold:
                    edges.append([src, dst.item()])
                    edge_weights.append(weight)

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).T
            edge_attr = torch.tensor(edge_weights, dtype=torch.float)

            # Make undirected and remove duplicates
            edge_index, edge_attr = to_undirected(edge_index, edge_attr, reduce='mean')
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty(0, dtype=torch.float)

        return Data(
            x=chunk_embeddings,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_chunks,
        )

    def build_graph_simple(
        self,
        chunk_embeddings: torch.Tensor,
        k: int = 10,
        threshold: float = 0.5,
    ) -> Data:
        """
        Simple k-NN graph construction (no document structure).

        Use this when document boundaries are not available.

        Args:
            chunk_embeddings: (num_chunks, embed_dim) tensor
            k: Number of nearest neighbors
            threshold: Minimum similarity threshold

        Returns:
            PyG Data object
        """
        # Move to CPU for graph construction
        chunk_embeddings = chunk_embeddings.cpu()
        num_chunks = chunk_embeddings.shape[0]

        # Normalize for cosine similarity
        embeddings_norm = torch.nn.functional.normalize(chunk_embeddings, p=2, dim=-1)
        similarities = torch.mm(embeddings_norm, embeddings_norm.T)

        # Zero out diagonal
        similarities.fill_diagonal_(0)

        # Get top-k
        k = min(k, num_chunks - 1)
        topk_values, topk_indices = torch.topk(similarities, k=k, dim=-1)

        # Build edges
        edges = []
        edge_weights = []

        for src in range(num_chunks):
            for j, dst in enumerate(topk_indices[src]):
                weight = topk_values[src, j].item()
                if weight > threshold:
                    edges.append([src, dst.item()])
                    edge_weights.append(weight)

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).T
            edge_attr = torch.tensor(edge_weights, dtype=torch.float)
            edge_index, edge_attr = to_undirected(edge_index, edge_attr, reduce='mean')
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty(0, dtype=torch.float)

        return Data(
            x=chunk_embeddings,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_chunks,
        )


def add_query_nodes_to_graph(
    graph: Data,
    query_embeddings: torch.Tensor,
    query_to_chunk: List[int],  # Maps query_idx -> target_chunk_idx
) -> Data:
    """
    Add synthetic query nodes to the graph (LP-RAG style).

    Each query is connected to its associated chunk(s).

    Args:
        graph: Existing chunk graph
        query_embeddings: (num_queries, embed_dim) tensor
        query_to_chunk: Maps each query to its target chunk

    Returns:
        Extended graph with query nodes
    """
    num_chunks = graph.num_nodes
    num_queries = query_embeddings.shape[0]

    # Concatenate embeddings
    new_x = torch.cat([graph.x, query_embeddings], dim=0)

    # Create query-chunk edges
    query_edges = []
    for q_idx, chunk_idx in enumerate(query_to_chunk):
        query_node_id = num_chunks + q_idx
        query_edges.append([chunk_idx, query_node_id])
        query_edges.append([query_node_id, chunk_idx])  # Bidirectional

    if query_edges:
        query_edge_index = torch.tensor(query_edges, dtype=torch.long).T
        new_edge_index = torch.cat([graph.edge_index, query_edge_index], dim=1)

        # Add edge weights (1.0 for query-chunk edges)
        query_edge_attr = torch.ones(query_edge_index.shape[1])
        if graph.edge_attr is not None:
            new_edge_attr = torch.cat([graph.edge_attr, query_edge_attr])
        else:
            new_edge_attr = query_edge_attr
    else:
        new_edge_index = graph.edge_index
        new_edge_attr = graph.edge_attr

    return Data(
        x=new_x,
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        num_nodes=num_chunks + num_queries,
        num_chunks=num_chunks,  # Store original chunk count
        num_queries=num_queries,
    )
