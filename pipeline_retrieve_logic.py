    def retrieve(
        self,
        query_text: str,
        top_k: Optional[int] = None,
    ) -> Tuple[List[Cell], List[int], torch.Tensor]:
        """
        SOTA Inductive Retrieval: Situates query node in complex and runs TNN.
        """
        if self.complex is None:
            raise ValueError("Must call process_chunks first")

        top_k = top_k or self.config.top_k_cells
        device = self.device
        
        # 1. Embed query
        query_emb = self.embed_texts([query_text], is_query=True)[0].to(device)

        # 2. Augment complex with query node (Inductive Situating)
        x_0_base = self.complex.x_0.to(device)
        sims = torch.matmul(query_emb.unsqueeze(0), x_0_base.t()).squeeze(0)
        _, top_neighbors = torch.topk(sims, k=min(5, x_0_base.shape[0]))
        
        x_aug = torch.cat([x_0_base, query_emb.unsqueeze(0)], dim=0)
        q_idx = x_0_base.shape[0]
        
        new_edges = []
        for n_idx in top_neighbors.tolist():
            new_edges.append([q_idx, n_idx])
            new_edges.append([n_idx, q_idx])
            
        base_indices = self.complex.adjacency_0.indices().to(device)
        new_edges_tensor = torch.tensor(new_edges, device=device).t()
        edge_index_aug = torch.cat([base_indices, new_edges_tensor], dim=1)
        
        from torch_geometric.data import Data
        data_aug = Data(x=x_aug, edge_index=edge_index_aug).to(device)
        complex_aug = self.lifting.lift(data_aug).to(device)
        
        # 3. Predict via LP-TNN
        self.model.eval()
        self.model.to(device)
        with torch.no_grad():
            ho_cells = complex_aug.get_all_higher_order_cells()
            # We want to retrieve original higher-order cells (knowledge units)
            # Filter out cells that contain the query node index
            target_cells = [c for c in ho_cells if q_idx not in c.chunk_indices]
            
            if not target_cells:
                target_cells = self.complex.get_all_higher_order_cells()

            # Batch scoring call
            q_node_indices = torch.tensor([q_idx], device=device)
            scores = self.model(complex_aug, q_node_indices, target_cells)
            
        # 4. Top-k Selection
        k = min(top_k, len(target_cells))
        top_scores, top_indices = torch.topk(scores.flatten(), k=k)
        top_cells = [target_cells[i] for i in top_indices.tolist()]
        
        chunk_indices = []
        for cell in top_cells:
            chunk_indices.extend(list(cell.chunk_indices))
            
        # Deduplicate retrieved chunks while preserving ranking
        seen = set()
        unique = []
        for idx in chunk_indices:
            if idx not in seen and idx < q_idx: # Ensure we don't return the query node
                seen.add(idx)
                unique.append(idx)

        return top_cells, unique, top_scores
