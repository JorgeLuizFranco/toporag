"""
Speculative Query Generation for TopoRAG.

This module generates synthetic queries at the CELL level (not chunk level).
This is a key innovation from paperidea.tex (Idea I2):

"For each cell σ, we prompt an LLM to generate a set of synthetic queries
Q_σ whose correct answers require the JOINT information contained across
ALL chunks associated with σ."

The key difference from LP-RAG:
- LP-RAG: Generates 2 queries per individual chunk
- TopoRAG: Generates queries per cell (group of chunks)

This enables the model to learn retrieval of coherent multi-hop groups.
"""

import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .liftings.base import Cell, TopologicalComplex


@dataclass
class SpeculativeQuery:
    """
    A speculative (synthetic) query generated for a cell.

    Attributes:
        query_text: The generated question text
        cell_id: ID of the cell this query is associated with
        cell_dimension: Dimension of the associated cell
        chunk_indices: Set of chunk indices that comprise the cell
        query_id: Unique identifier for this query
    """
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


# Prompt for generating cell-level queries
# This prompts the LLM to generate questions that require JOINT information
# from ALL chunks in the cell
PROMPT_CELL_QUERY_GENERATION = """You are an AI annotator that generates questions requiring information from multiple text chunks.

You will be given a SET of related text chunks that together form a coherent semantic unit. Your task is to generate questions whose answers require synthesizing information from ALL or MOST of the provided chunks.

Important guidelines:
1. The question should NOT be answerable from just one chunk alone
2. The question should require combining information across multiple chunks
3. Generate questions that a human might naturally ask when given this information
4. Focus on relationships, comparisons, or synthesis across the chunks

Below is an example:

System's Input:
[The Start of Chunk Set]
[Chunk 0]
Albert Einstein was born in Ulm, Germany in 1879.
[Chunk 1]
Einstein developed the theory of special relativity in 1905.
[Chunk 2]
The famous equation E=mc² was published as part of special relativity.
[The End of Chunk Set]

Your Answer:
[Multi-hop Question 1]
[What year did the physicist born in Ulm publish the equation E=mc²?]
[Multi-hop Question 2]
[Which German-born scientist developed the theory that includes E=mc²?]

Now generate {num_queries} questions for the following chunk set:

System's Input:
[The Start of Chunk Set]
{chunk_set_text}
[The End of Chunk Set]

Your Answer:
"""

# Simpler prompt for smaller cells (2-3 chunks)
PROMPT_SMALL_CELL_QUERY = """Generate {num_queries} questions that require information from BOTH of these related text chunks to answer:

[Chunk Set]
{chunk_set_text}

Generate questions that combine information from multiple chunks. Format your response as:
[Question 1]
[Your question here?]
[Question 2]
[Your question here?]
"""


class SpeculativeQueryGenerator:
    """
    Generates speculative queries for cells in a topological complex.

    For each cell (group of chunks), this generates synthetic queries
    whose answers require the joint information from all chunks in the cell.

    Args:
        llm: Language model for generating queries (OpenAI, Groq, etc.)
        num_queries_per_cell: Number of queries to generate per cell
        min_cell_size: Minimum cell size to generate queries for
        max_cell_size: Maximum cell size to generate queries for
    """

    def __init__(
        self,
        llm,
        num_queries_per_cell: int = 2,
        min_cell_size: int = 2,
        max_cell_size: int = 10,
    ):
        self.llm = llm
        self.num_queries_per_cell = num_queries_per_cell
        self.min_cell_size = min_cell_size
        self.max_cell_size = max_cell_size

    def _format_chunk_set(
        self, cell: Cell, chunk_texts: List[str]
    ) -> str:
        """Format chunks in a cell as text for the prompt."""
        lines = []
        for i, chunk_idx in enumerate(sorted(cell.chunk_indices)):
            lines.append(f"[Chunk {i}]")
            lines.append(chunk_texts[chunk_idx])
        return "\n".join(lines)

    def _parse_queries(self, response: str) -> List[str]:
        """Parse generated queries from LLM response."""
        # Try multiple patterns
        patterns = [
            r"\[(?:Multi-hop )?Question \d+\]\s*\n?\[([^\]]+\?)\]",
            r"\[([^\]]+\?)\]",
            r"Question \d+[:\s]+([^\n]+\?)",
        ]

        queries = []
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                queries.extend(matches)
                break

        # Fallback: look for any question marks
        if not queries:
            lines = response.split("\n")
            for line in lines:
                line = line.strip()
                if line.endswith("?") and len(line) > 10:
                    # Clean up the line
                    line = re.sub(r"^\d+[\.\)]\s*", "", line)
                    line = re.sub(r"^\[|\]$", "", line)
                    queries.append(line)

        return queries[: self.num_queries_per_cell]

    def generate_for_cell(
        self,
        cell: Cell,
        chunk_texts: List[str],
    ) -> List[SpeculativeQuery]:
        """
        Generate speculative queries for a single cell.

        Args:
            cell: The cell to generate queries for
            chunk_texts: List of all chunk texts (indexed by chunk_idx)

        Returns:
            List of SpeculativeQuery objects
        """
        # Skip cells that are too small or too large
        if len(cell.chunk_indices) < self.min_cell_size:
            return []
        if len(cell.chunk_indices) > self.max_cell_size:
            return []

        # Format the chunk set
        chunk_set_text = self._format_chunk_set(cell, chunk_texts)

        # Choose prompt based on cell size
        if len(cell.chunk_indices) <= 3:
            prompt = PROMPT_SMALL_CELL_QUERY.format(
                num_queries=self.num_queries_per_cell,
                chunk_set_text=chunk_set_text,
            )
        else:
            prompt = PROMPT_CELL_QUERY_GENERATION.format(
                num_queries=self.num_queries_per_cell,
                chunk_set_text=chunk_set_text,
            )

        # Generate queries using LLM
        try:
            if hasattr(self.llm, "openia_generate"):
                response, _ = self.llm.openia_generate("", prompt, 1)
            elif hasattr(self.llm, "generate"):
                response = self.llm.generate(prompt)
            else:
                raise ValueError("LLM must have 'generate' or 'openia_generate' method")
        except Exception as e:
            print(f"Error generating queries for cell {cell.cell_id}: {e}")
            return []

        # Parse queries from response
        query_texts = self._parse_queries(response)

        # Create SpeculativeQuery objects
        queries = []
        for i, query_text in enumerate(query_texts):
            query = SpeculativeQuery(
                query_text=query_text,
                cell_id=cell.cell_id,
                cell_dimension=cell.dimension,
                chunk_indices=cell.chunk_indices,
                query_id=len(queries),
            )
            queries.append(query)

        return queries

    def generate_for_complex(
        self,
        complex_: TopologicalComplex,
        chunk_texts: List[str],
        target_dimensions: Optional[List[int]] = None,
    ) -> Dict[int, List[SpeculativeQuery]]:
        """
        Generate speculative queries for all cells in a topological complex.

        Args:
            complex_: The topological complex containing cells
            chunk_texts: List of all chunk texts
            target_dimensions: If specified, only generate for these dimensions

        Returns:
            Dict mapping cell_id -> list of queries for that cell
        """
        all_queries = {}
        global_query_id = 0

        # Get cells to process
        cells_to_process = complex_.get_all_higher_order_cells()

        if target_dimensions is not None:
            cells_to_process = [
                c for c in cells_to_process if c.dimension in target_dimensions
            ]

        for cell in cells_to_process:
            queries = self.generate_for_cell(cell, chunk_texts)

            # Assign global query IDs
            for q in queries:
                q.query_id = global_query_id
                global_query_id += 1

            if queries:
                all_queries[cell.cell_id] = queries

        return all_queries


def save_queries(
    queries_by_cell: Dict[int, List[SpeculativeQuery]],
    output_path: str,
):
    """Save generated queries to JSON file."""
    data = {
        str(cell_id): [q.to_dict() for q in queries]
        for cell_id, queries in queries_by_cell.items()
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_queries(input_path: str) -> Dict[int, List[SpeculativeQuery]]:
    """Load queries from JSON file."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries_by_cell = {}
    for cell_id_str, query_dicts in data.items():
        cell_id = int(cell_id_str)
        queries = [
            SpeculativeQuery(
                query_text=d["query_text"],
                cell_id=d["cell_id"],
                cell_dimension=d["cell_dimension"],
                chunk_indices=set(d["chunk_indices"]),
                query_id=d["query_id"],
            )
            for d in query_dicts
        ]
        queries_by_cell[cell_id] = queries

    return queries_by_cell
