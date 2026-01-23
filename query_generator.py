"""
Synthetic Query Generator for TopoRAG.

Generates speculative queries for cells (not just chunks) following the paper idea:
- For each cell σ, generate queries that require JOINT information from ALL chunks in σ
- These queries serve as supervision for training the retrieval model

This is different from LP-RAG which generates queries per chunk.
TopoRAG generates queries per CELL requiring multi-hop reasoning.
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SyntheticQuery:
    """A synthetic query associated with a cell."""
    query_text: str
    cell_idx: int
    chunk_indices: List[int]  # Chunks in the cell


def generate_cell_queries(
    chunks: List[str],
    cell_to_nodes: Dict[int, List[int]],
    llm_provider: str = "groq",
    queries_per_cell: int = 2,
    max_cells: Optional[int] = None,
) -> List[SyntheticQuery]:
    """
    Generate synthetic queries for cells.

    For each cell, prompts LLM to generate queries that require
    synthesizing information from ALL chunks in the cell.

    Args:
        chunks: List of chunk texts
        cell_to_nodes: Dict mapping cell_idx -> list of chunk indices
        llm_provider: LLM provider ('groq', 'openai')
        queries_per_cell: Number of queries to generate per cell
        max_cells: Maximum number of cells to process (for rate limiting)

    Returns:
        List of SyntheticQuery objects
    """
    from .llms import get_llm

    llm = get_llm(llm_provider)
    queries = []

    cells_to_process = list(cell_to_nodes.items())
    if max_cells:
        cells_to_process = cells_to_process[:max_cells]

    print(f"Generating synthetic queries for {len(cells_to_process)} cells...")

    for cell_idx, node_indices in cells_to_process:
        # Get chunk texts for this cell
        cell_chunks = [chunks[i] for i in node_indices if i < len(chunks)]
        if len(cell_chunks) < 2:
            # Skip cells with only 1 chunk (no multi-hop needed)
            continue

        # Build context from chunks
        context = "\n\n".join([
            f"[Passage {i+1}]: {chunk[:500]}"  # Truncate long chunks
            for i, chunk in enumerate(cell_chunks[:5])  # Max 5 chunks
        ])

        # Prompt for multi-hop query
        prompt = f"""You are given multiple related text passages. Generate {queries_per_cell} question(s) that can ONLY be answered by combining information from MULTIPLE passages.

The questions should:
1. Require synthesis/reasoning across passages (not just lookup from one)
2. Be answerable from the given passages
3. Be natural questions a user might ask

Passages:
{context}

Generate {queries_per_cell} multi-hop question(s), one per line. Output ONLY the questions, nothing else."""

        try:
            response = llm.generate(prompt)
            if response:
                # Parse response into individual queries
                generated = [q.strip() for q in response.strip().split('\n') if q.strip()]
                for q in generated[:queries_per_cell]:
                    # Clean up numbering if present
                    if q[0].isdigit() and (q[1] == '.' or q[1] == ')'):
                        q = q[2:].strip()
                    queries.append(SyntheticQuery(
                        query_text=q,
                        cell_idx=cell_idx,
                        chunk_indices=node_indices,
                    ))
        except Exception as e:
            print(f"  Error generating for cell {cell_idx}: {e}")

        # Rate limiting
        import time
        time.sleep(0.5)

    print(f"  Generated {len(queries)} synthetic queries")
    return queries


def generate_chunk_queries(
    chunks: List[str],
    llm_provider: str = "groq",
    queries_per_chunk: int = 2,
    max_chunks: Optional[int] = None,
) -> List[Tuple[str, int]]:
    """
    Generate synthetic queries for individual chunks (LP-RAG style).

    For each chunk, generate queries whose answers are in that chunk.

    Args:
        chunks: List of chunk texts
        llm_provider: LLM provider
        queries_per_chunk: Number of queries per chunk
        max_chunks: Maximum chunks to process

    Returns:
        List of (query_text, chunk_idx) tuples
    """
    from .llms import get_llm

    llm = get_llm(llm_provider)
    queries = []

    chunks_to_process = list(enumerate(chunks))
    if max_chunks:
        chunks_to_process = chunks_to_process[:max_chunks]

    print(f"Generating synthetic queries for {len(chunks_to_process)} chunks...")

    for chunk_idx, chunk_text in chunks_to_process:
        prompt = f"""Given the following text passage, generate {queries_per_chunk} question(s) that can be answered using information from this passage.

Passage:
{chunk_text[:1000]}

Generate {queries_per_chunk} question(s), one per line. Output ONLY the questions."""

        try:
            response = llm.generate(prompt)
            if response:
                generated = [q.strip() for q in response.strip().split('\n') if q.strip()]
                for q in generated[:queries_per_chunk]:
                    if q[0].isdigit() and len(q) > 2 and (q[1] == '.' or q[1] == ')'):
                        q = q[2:].strip()
                    queries.append((q, chunk_idx))
        except Exception as e:
            print(f"  Error for chunk {chunk_idx}: {e}")

        import time
        time.sleep(0.3)

    print(f"  Generated {len(queries)} synthetic queries")
    return queries
