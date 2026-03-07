"""
Speculative Query Generation for TopoRAG.
ZERO-SHOT PROMPT: Eliminates hallucinations by removing naming examples.
"""

import json
import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from .lifting.base import Cell, LiftedTopology


@dataclass
class SpeculativeQuery:
    query_text: str
    cell_id: int
    cell_dimension: int
    chunk_indices: set
    query_id: int

    def to_dict(self) -> dict:
        return {
            "query_text": self.query_text,
            "cell_id": self.cell_id,
            "cell_dimension": self.cell_dimension,
            "chunk_indices": list(self.chunk_indices),
            "query_id": self.query_id,
        }


# ZERO-SHOT SOTA PROMPT
PROMPT_CELL_QUERY_GENERATION = """Write a single complex question that requires information from ALL the text chunks provided below to answer.

### INSTRUCTIONS:
1. NO SINGLE CHUNK: The question must not be answerable using only one chunk.
2. SYNTHESIS: The question must force the reader to link facts across different chunks.
3. SPECIFICITY: Use the exact names and details from the provided chunks.
4. DIRECT: Ask the question directly. Do not include meta-text.
5. NO OUTSIDE INFO: Do not use any names or facts NOT present in the chunks.

### CHUNKS:
{chunk_set_text}

Your Answer (ONE Question):
"""

class SpeculativeQueryGenerator:
    def __init__(self, llm, num_queries_per_cell: int = 1, min_cell_size: int = 2, max_cell_size: int = 10):
        self.llm = llm
        self.num_queries_per_cell = num_queries_per_cell
        self.min_cell_size = min_cell_size
        self.max_cell_size = max_cell_size

    def _format_chunk_set(self, cell: Cell, chunk_texts: List[str]) -> str:
        lines = []
        for i, chunk_idx in enumerate(sorted(cell.chunk_indices)):
            text = chunk_texts[chunk_idx] if isinstance(chunk_idx, int) and chunk_idx < len(chunk_texts) else str(chunk_idx)
            lines.append(f"Chunk {i}: {text}")
        return "\n".join(lines)

    def _parse_queries(self, response: str) -> List[str]:
        # Simple extraction
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if '?' in line:
                line = re.sub(r"^(Question|1)\s*[:\.]\s*", "", line, flags=re.I)
                line = line.replace('"', '').replace('[', '').replace(']', '')
                if line.endswith('?'): return [line]
        return []

    def generate_for_cell(self, cell: Cell, chunk_texts: List[str]) -> List[SpeculativeQuery]:
        if not (self.min_cell_size <= len(cell.chunk_indices) <= self.max_cell_size): return []
        chunk_set_text = self._format_chunk_set(cell, chunk_texts)
        all_cell_queries = []
        for v_idx in range(1, self.num_queries_per_cell + 1):
            prompt = PROMPT_CELL_QUERY_GENERATION.format(chunk_set_text=chunk_set_text)
            try:
                response = self.llm.generate(prompt)
                query_texts = self._parse_queries(response)
                for q_text in query_texts:
                    all_cell_queries.append(SpeculativeQuery(q_text, cell.cell_id, cell.dimension, cell.chunk_indices, v_idx))
                    if len(all_cell_queries) < 2: print(f"    [GEN]: {q_text}")
            except Exception as e: print(f"    Error: {e}")
        return all_cell_queries

    def generate_for_complex(self, complex_, chunk_texts: List[str]):
        all_queries = {}
        cells = complex_.get_all_higher_order_cells()
        print(f"  Generating queries for {len(cells)} cells...")
        for i, cell in enumerate(cells):
            if i % 100 == 0: print(f"    Progress: {i}/{len(cells)}")
            queries = self.generate_for_cell(cell, chunk_texts)
            if queries: all_queries[cell.cell_id] = queries
        return all_queries


def save_queries(queries_by_cell: Dict[int, List[SpeculativeQuery]], output_path: str):
    data = {str(k): [q.to_dict() for q in v] for k, v in queries_by_cell.items()}
    with open(output_path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)

def load_queries(input_path: str) -> Dict[int, List[SpeculativeQuery]]:
    with open(input_path, "r", encoding="utf-8") as f: data = json.load(f)
    return {int(k): [SpeculativeQuery(q["query_text"], q["cell_id"], q["cell_dimension"], set(q["chunk_indices"]), q["query_id"]) for q in v] for k, v in data.items()}