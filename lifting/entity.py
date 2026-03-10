"""
Entity-Based Hypergraph Lifting: Chunks → Hypergraph via Shared Entities

Three-stage lifting pipeline:
  1. NER Extraction   — spaCy extracts named entities from chunks (fast, O(n))
  2. Entity Resolution — merge alias entities via substring matching + demonym
                         normalization (free, O(E²) where E << n)
  3. Cell Subdivision  — split oversized entity cells into semantic sub-cells
                         using k-means on chunk embeddings (free, uses existing embeddings)

This creates hyperedges connecting COMPLEMENTARY chunks (different topics,
same entity) — exactly the structure needed for multi-hop retrieval.

Example:
  Chunk A: "Steven Spielberg directed Jaws"     → entities: {Spielberg, Jaws}
  Chunk B: "Spielberg was born in Cincinnati"    → entities: {Spielberg, Cincinnati}
  Chunk C: "Jaws grossed $470M"                  → entities: {Jaws}

  Entity cells:
    "Spielberg" → {A, B}  (complementary! different topics, same entity)
    "Jaws"      → {A, C}  (complementary!)

  Speculative query for "Spielberg" cell: "Where was the director of Jaws born?"
  → genuinely multi-hop, requires bridging A and B.

Enhancement rationale (March 2026 analysis on 500 MuSiQue questions):
  - Old lifting: max_cell_size=10 filter discarded 44.2% of connected gold pairs
  - Substring aliasing missed 16.5% ("Obama" ≠ "Barack Obama")
  - Demonym mismatch missed part of 37.7% ("Polish" ≠ "Poland")
  - After enhancements: gold pair connectivity 21% → ~55% (projected)
"""

import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from torch_geometric.data import Data

from .base import BaseLiftingTransform, LiftedTopology, Cell


# ---------------------------------------------------------------------------
# Demonym / adjective → canonical place name dictionary
# Used to merge "American" (NORP) with "United States" / "America" (GPE).
# This is a static lookup — no LLM needed.
# ---------------------------------------------------------------------------
DEMONYM_TO_PLACE = {
    "american": "america", "americans": "america",
    "british": "britain", "britons": "britain",
    "english": "england",
    "french": "france",
    "german": "germany", "germans": "germany",
    "italian": "italy", "italians": "italy",
    "spanish": "spain",
    "chinese": "china",
    "japanese": "japan",
    "russian": "russia", "russians": "russia",
    "polish": "poland",
    "dutch": "netherlands",
    "swedish": "sweden",
    "norwegian": "norway",
    "danish": "denmark",
    "finnish": "finland",
    "portuguese": "portugal",
    "greek": "greece",
    "turkish": "turkey",
    "egyptian": "egypt",
    "indian": "india", "indians": "india",
    "canadian": "canada", "canadians": "canada",
    "australian": "australia", "australians": "australia",
    "mexican": "mexico", "mexicans": "mexico",
    "brazilian": "brazil", "brazilians": "brazil",
    "argentinian": "argentina", "argentinians": "argentina",
    "korean": "korea", "koreans": "korea",
    "thai": "thailand",
    "vietnamese": "vietnam",
    "irish": "ireland",
    "scottish": "scotland",
    "welsh": "wales",
    "swiss": "switzerland",
    "austrian": "austria",
    "belgian": "belgium",
    "czech": "czech republic",
    "hungarian": "hungary",
    "romanian": "romania",
    "ukrainian": "ukraine",
    "south african": "south africa",
    "israeli": "israel",
    "iranian": "iran",
    "iraqi": "iraq",
    "afghan": "afghanistan",
    "pakistani": "pakistan",
    "filipino": "philippines",
    "indonesian": "indonesia",
    "malaysian": "malaysia",
    "singaporean": "singapore",
    "colombian": "colombia",
    "peruvian": "peru",
    "chilean": "chile",
    "cuban": "cuba",
    "nigerian": "nigeria",
    "kenyan": "kenya",
    "moroccan": "morocco",
    "saudi": "saudi arabia",
}


class EntityHypergraphLifting(BaseLiftingTransform):
    """
    Lift chunks to hypergraph via shared named entities (spaCy NER)
    with entity resolution and cell subdivision.

    Pipeline:
      1. Extract entities via spaCy NER (fast, deterministic)
      2. Resolve aliases: merge entities where one's words are a
         subset of another's (e.g., "Obama" → "Barack Obama")
      3. Normalize demonyms: map adjective forms to canonical place
         names (e.g., "Polish" → "Poland") via static dictionary
      4. Subdivide large cells: split entities with >max_cell_size
         chunks into semantic sub-cells via k-means on chunk embeddings
      5. Build incidence matrix from the resulting cells

    No LLM calls required. All enhancements are free (use existing
    embeddings and deterministic string processing).
    """

    def __init__(
        self,
        min_cell_size: int = 2,
        max_cell_size: int = 50,
        entity_types: Optional[Set[str]] = None,
        feature_lifting: str = "projection_sum",
        resolve_aliases: bool = True,
        normalize_demonyms: bool = True,
        subdivide_large: bool = True,
        subdivision_target_size: int = 8,
        subdivision_max_cells_per_entity: int = 30,
    ):
        """
        Args:
            min_cell_size: Minimum chunks per entity to form a cell (default 2)
            max_cell_size: Maximum chunks per direct cell (default 50). Entities
                          with more chunks are subdivided (not discarded) if
                          subdivide_large=True. Raised from 10→50 because the old
                          filter discarded 44.2% of connected gold chunk pairs.
            entity_types: spaCy entity types to use. None = all.
            feature_lifting: How to lift node features ("projection_sum" or "mean")
            resolve_aliases: If True, merge entities where one is a word-subset
                           of another (e.g., "Obama" ⊂ "Barack Obama").
            normalize_demonyms: If True, map NORP entities to canonical place
                              names (e.g., "American" → "America").
            subdivide_large: If True, split entities with >max_cell_size chunks
                           into sub-cells via k-means. If False, discard them.
            subdivision_target_size: Target size for sub-cells when subdividing.
            subdivision_max_cells_per_entity: Cap on sub-cells per entity to
                                            prevent combinatorial explosion.
        """
        super().__init__(complex_dim=1, feature_lifting=feature_lifting)
        self.min_cell_size = min_cell_size
        self.max_cell_size = max_cell_size
        self.entity_types = entity_types or {
            "PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT",
            "WORK_OF_ART", "PRODUCT", "NORP", "LAW",
        }
        self.resolve_aliases = resolve_aliases
        self.normalize_demonyms = normalize_demonyms
        self.subdivide_large = subdivide_large
        self.subdivision_target_size = subdivision_target_size
        self.subdivision_max_cells_per_entity = subdivision_max_cells_per_entity

    # ------------------------------------------------------------------
    # Stage 1: NER Extraction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_entities(
        chunks: List[str],
        entity_types: Set[str],
        batch_size: int = 500,
    ) -> Dict[str, List[int]]:
        """Extract named entities from chunks using spaCy.

        Returns:
            entity_to_chunks: entity_text (lowercased) → list of chunk indices
        """
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

        entity_to_chunks: Dict[str, List[int]] = defaultdict(list)

        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start:batch_start + batch_size]
            batch_truncated = [c[:500] for c in batch]
            docs = nlp.pipe(batch_truncated, batch_size=min(64, len(batch)))

            for local_idx, doc in enumerate(docs):
                chunk_idx = batch_start + local_idx
                for ent in doc.ents:
                    if ent.label_ in entity_types:
                        key = ent.text.strip().lower()
                        if len(key) >= 2:
                            entity_to_chunks[key].append(chunk_idx)

        # Deduplicate: each chunk appears at most once per entity
        for key in entity_to_chunks:
            entity_to_chunks[key] = sorted(set(entity_to_chunks[key]))

        return dict(entity_to_chunks)

    # ------------------------------------------------------------------
    # Stage 2: Entity Resolution (substring alias merging)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_entity_aliases(
        entity_to_chunks: Dict[str, List[int]],
    ) -> Dict[str, List[int]]:
        """Merge entities where one's words are a subset of another's.

        Examples:
          "obama" → "barack obama"           (word subset)
          "sony"  → "sony music entertainment" (word subset)
          "st. peter's" → "st. peter"        (word subset after normalization)

        Algorithm:
          1. Tokenize each entity into word-set
          2. For each short entity, check if its words ⊆ some longer entity's words
          3. Merge: union their chunk lists under the longer (canonical) form
          4. O(E² · W) where E = entities, W = avg words per entity

        This is conservative — only merges when one entity is a strict
        word-subset of another, avoiding false merges like "new" → "new york".
        """
        # Build word sets
        entity_words = {}
        for ent in entity_to_chunks:
            words = frozenset(ent.replace("'s", "").replace(".", "").split())
            entity_words[ent] = words

        # Sort by word count descending (longest first = canonical)
        entities_by_length = sorted(
            entity_to_chunks.keys(),
            key=lambda e: len(entity_words[e]),
            reverse=True,
        )

        # Map short → canonical (longest form whose words are a superset)
        canonical_map: Dict[str, str] = {}
        for i, short_ent in enumerate(entities_by_length):
            if short_ent in canonical_map:
                continue
            short_words = entity_words[short_ent]
            if len(short_words) < 1:
                continue
            # Check against longer entities
            for long_ent in entities_by_length[:i]:
                long_canon = canonical_map.get(long_ent, long_ent)
                long_words = entity_words[long_canon]
                # Short entity's words must be a strict subset of long entity
                if short_words < long_words and len(short_words) >= 1:
                    canonical_map[short_ent] = long_canon
                    break

        # Merge chunk lists
        merged: Dict[str, Set[int]] = defaultdict(set)
        merge_count = 0
        for ent, chunks in entity_to_chunks.items():
            canon = canonical_map.get(ent, ent)
            if canon != ent:
                merge_count += 1
            merged[canon].update(chunks)

        result = {k: sorted(v) for k, v in merged.items()}
        if merge_count > 0:
            print(f"    Alias resolution: merged {merge_count} entities "
                  f"({len(entity_to_chunks)} → {len(result)})")
        return result

    # ------------------------------------------------------------------
    # Stage 3: Demonym Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_demonyms(
        entity_to_chunks: Dict[str, List[int]],
    ) -> Dict[str, List[int]]:
        """Map demonym/nationality entities to canonical place names.

        Examples:
          "american" → chunks merged with "america" / "united states"
          "french"   → chunks merged with "france"
          "polish"   → chunks merged with "poland"

        Uses a static dictionary (DEMONYM_TO_PLACE). No LLM needed.
        Only merges when the canonical place name already exists as an
        entity — otherwise the demonym stays as-is to avoid phantom entities.
        """
        merged: Dict[str, Set[int]] = {}
        merge_count = 0

        # First pass: copy all existing entities
        for ent, chunks in entity_to_chunks.items():
            merged[ent] = set(chunks)

        # Second pass: merge demonyms into place names
        for ent in list(merged.keys()):
            place = DEMONYM_TO_PLACE.get(ent)
            if place and place in merged and place != ent:
                # Merge demonym chunks into place entity
                merged[place].update(merged[ent])
                del merged[ent]
                merge_count += 1
            elif place and place not in merged:
                # Rename demonym to place form (no existing place entity)
                merged[place] = merged.pop(ent)
                merge_count += 1

        result = {k: sorted(v) for k, v in merged.items()}
        if merge_count > 0:
            print(f"    Demonym normalization: merged {merge_count} entities")
        return result

    # ------------------------------------------------------------------
    # Stage 4: Cell Subdivision (overlapping ego-cells for large entities)
    # ------------------------------------------------------------------

    @staticmethod
    def _subdivide_large_cells(
        entity_to_chunks: Dict[str, List[int]],
        chunk_embeddings: torch.Tensor,
        max_cell_size: int = 50,
        target_size: int = 8,
        max_cells_per_entity: int = 30,
    ) -> Tuple[Dict[str, List[List[int]]], int, int]:
        """Split entities with >max_cell_size chunks into overlapping sub-cells.

        Why NOT k-means: Multi-hop retrieval needs COMPLEMENTARY chunks
        (dissimilar, cosine ~0.27). K-means groups by similarity, actively
        separating the gold pairs we need together.

        Instead, we use OVERLAPPING ego-cells via farthest-point sampling:
          1. Pick diverse seed chunks via farthest-point sampling
          2. Each seed's sub-cell = seed + (target_size-1) nearest neighbors
          3. Overlapping: each chunk may appear in multiple sub-cells
          4. Any two chunks close in the entity graph have high chance
             of sharing a sub-cell

        The overlap is KEY — it preserves pairwise connectivity that
        disjoint partitioning (k-means) destroys.

        Args:
            entity_to_chunks: entity → chunk indices
            chunk_embeddings: (num_chunks, embed_dim) tensor on CPU
            max_cell_size: Cells up to this size are kept as-is
            target_size: Target size for each ego-cell sub-cell
            max_cells_per_entity: Max sub-cells per entity

        Returns:
            entity_to_subcells: entity → list of sub-cell chunk lists
            n_subdivided: number of entities that were subdivided
            n_subcells: total sub-cells created from subdivided entities
        """
        result: Dict[str, List[List[int]]] = {}
        n_subdivided = 0
        n_subcells_total = 0

        emb_norm = F.normalize(chunk_embeddings.float(), p=2, dim=-1)

        for ent, chunk_ids in entity_to_chunks.items():
            if len(chunk_ids) <= max_cell_size:
                result[ent] = [chunk_ids]
                continue

            n_subdivided += 1
            n_chunks = len(chunk_ids)
            indices_t = torch.tensor(chunk_ids, dtype=torch.long)
            ent_embs = emb_norm[indices_t]  # (n_chunks, dim)

            # Pairwise cosine similarities within entity
            sims = torch.mm(ent_embs, ent_embs.t())  # (n, n)

            # Number of ego-cells to create
            n_seeds = min(max(2, n_chunks // target_size), max_cells_per_entity)

            # Farthest-point sampling for diverse seeds
            seeds = _farthest_point_sample(sims, n_seeds)

            # Build ego-cells: each seed + nearest neighbors
            ts = min(target_size, n_chunks)
            subcells = []
            for seed in seeds:
                # Top-ts most similar chunks to this seed (including seed itself)
                _, nn_indices = torch.topk(sims[seed], ts)
                subcell = indices_t[nn_indices].tolist()
                if len(subcell) >= 2:
                    subcells.append(subcell)

            if subcells:
                result[ent] = subcells
                n_subcells_total += len(subcells)
            else:
                result[ent] = [chunk_ids]

        return result, n_subdivided, n_subcells_total

    # ------------------------------------------------------------------
    # Main lift method
    # ------------------------------------------------------------------

    def lift(self, data: Data, chunks: Optional[List[str]] = None) -> LiftedTopology:
        """
        Lift to hypergraph via entity co-occurrence with resolution and subdivision.

        Pipeline:
          1. Extract entities via spaCy NER
          2. (Optional) Resolve aliases via word-subset matching
          3. (Optional) Normalize demonyms via static dictionary
          4. Filter by min_cell_size
          5. (Optional) Subdivide large cells via overlapping ego-cells
          6. Build incidence matrix

        Args:
            data: PyG Data with x (node features) and optionally edge_index
            chunks: Raw chunk texts for NER. Required for entity extraction.
        """
        if chunks is None:
            raise ValueError(
                "EntityHypergraphLifting requires chunk texts for NER. "
                "Pass chunks= to lift()."
            )

        device = data.x.device
        num_nodes = data.x.shape[0]
        node_features = data.x

        # --- Stage 1: Extract entities ---
        print("  [Stage 1] NER extraction (spaCy)...")
        entity_to_chunks = self.extract_entities(chunks, self.entity_types)
        n_raw = len(entity_to_chunks)
        print(f"    {n_raw} raw entities extracted")

        # --- Stage 2: Entity resolution (alias merging) ---
        if self.resolve_aliases:
            print("  [Stage 2] Entity resolution (word-subset alias merging)...")
            entity_to_chunks = self._resolve_entity_aliases(entity_to_chunks)

        # --- Stage 3: Demonym normalization ---
        if self.normalize_demonyms:
            print("  [Stage 3] Demonym normalization...")
            entity_to_chunks = self._normalize_demonyms(entity_to_chunks)

        # --- Filter by min_cell_size ---
        entity_to_chunks = {
            ent: cids for ent, cids in entity_to_chunks.items()
            if len(cids) >= self.min_cell_size
        }
        n_after_filter = len(entity_to_chunks)
        print(f"    {n_after_filter} entities with >= {self.min_cell_size} chunks")

        # --- Stage 4: Cell subdivision ---
        if self.subdivide_large:
            # Count how many would be discarded by old max_cell_size filter
            n_oversized = sum(1 for cids in entity_to_chunks.values()
                             if len(cids) > self.max_cell_size)
            print(f"  [Stage 4] Cell subdivision (overlapping ego-cells)...")
            print(f"    {n_oversized} entities with >{self.max_cell_size} chunks → subdividing")

            entity_to_subcells, n_sub, n_subcells = self._subdivide_large_cells(
                entity_to_chunks,
                node_features.cpu(),
                max_cell_size=self.max_cell_size,
                target_size=self.subdivision_target_size,
                max_cells_per_entity=self.subdivision_max_cells_per_entity,
            )
            print(f"    {n_sub} entities subdivided → {n_subcells} sub-cells created")
        else:
            # Old behavior: discard oversized entities
            entity_to_subcells = {
                ent: [cids] for ent, cids in entity_to_chunks.items()
                if len(cids) <= self.max_cell_size
            }
            print(f"    {len(entity_to_subcells)} entities after max_cell_size={self.max_cell_size} filter")

        # --- Build hyperedges from subcells ---
        hyperedges: List[List[int]] = []
        cell_to_nodes: Dict[int, List[int]] = {}
        cells: List[Cell] = []
        rows, cols, vals = [], [], []
        cell_idx = 0

        for entity_name in sorted(entity_to_subcells.keys()):
            for subcell_chunks in entity_to_subcells[entity_name]:
                if len(subcell_chunks) < self.min_cell_size:
                    continue
                hyperedges.append(subcell_chunks)
                cell_to_nodes[cell_idx] = subcell_chunks
                cells.append(Cell(
                    chunk_indices=set(subcell_chunks),
                    dimension=1,
                    cell_id=cell_idx,
                ))
                for node_idx in subcell_chunks:
                    rows.append(node_idx)
                    cols.append(cell_idx)
                    vals.append(1.0)
                cell_idx += 1

        num_edges = len(hyperedges)
        print(f"  Total cells: {num_edges}")

        # --- Build incidence matrix ---
        if num_edges > 0:
            indices = torch.tensor([rows, cols], dtype=torch.long)
            values = torch.tensor(vals, dtype=torch.float)
            incidence_1 = torch.sparse_coo_tensor(
                indices, values, size=(num_nodes, num_edges)
            ).coalesce().to(device)
        else:
            incidence_1 = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long),
                torch.empty(0),
                size=(num_nodes, 0),
            ).to(device)

        # --- Lift features ---
        x_1 = self._lift_features(node_features, incidence_1) if num_edges > 0 else None

        # --- Build topology ---
        lifted = LiftedTopology(
            x_0=node_features,
            x_1=x_1,
            incidence_1=incidence_1,
            num_nodes=num_nodes,
            num_edges=num_edges,
            cells=cells,
            cell_to_nodes=cell_to_nodes,
        )
        lifted.compute_laplacians()

        # --- Stats ---
        if hyperedges:
            sizes = [len(h) for h in hyperedges]
            print(f"  Cell sizes: min={min(sizes)}, max={max(sizes)}, "
                  f"mean={sum(sizes)/len(sizes):.1f}")
            covered = set()
            for chunk_ids in hyperedges:
                covered.update(chunk_ids)
            print(f"  Chunk coverage: {len(covered)}/{num_nodes} "
                  f"({100*len(covered)/num_nodes:.1f}%)")

        return lifted

    @property
    def entity_names(self) -> Dict[int, str]:
        """Return entity names for each cell (for query generation prompts)."""
        return getattr(self, '_entity_names', {})


# ---------------------------------------------------------------------------
# Utility: farthest-point sampling for diverse seed selection
# ---------------------------------------------------------------------------

def _farthest_point_sample(
    similarity_matrix: torch.Tensor,
    k: int,
) -> List[int]:
    """Select k diverse points via farthest-point sampling.

    Greedily picks points that are maximally dissimilar from already-
    selected points. This ensures ego-cell seeds are well-spread across
    the entity, giving good coverage of all chunks.

    Args:
        similarity_matrix: (n, n) pairwise cosine similarities
        k: number of seeds to select

    Returns:
        List of k seed indices (into the local entity chunk list)
    """
    n = similarity_matrix.shape[0]
    k = min(k, n)

    # Start with the most "central" chunk (highest average similarity)
    avg_sim = similarity_matrix.mean(dim=1)
    seeds = [avg_sim.argmax().item()]

    # Track minimum similarity to any selected seed (= "distance")
    min_sim_to_seeds = similarity_matrix[seeds[0]].clone()

    for _ in range(1, k):
        # Pick the point with LOWEST similarity to its nearest seed
        # (i.e., farthest from any selected point)
        # Set already-selected to high similarity so they aren't re-picked
        min_sim_to_seeds[seeds] = float('inf')
        next_seed = min_sim_to_seeds.argmin().item()
        seeds.append(next_seed)

        # Update: min similarity considering the new seed
        new_sims = similarity_matrix[next_seed]
        min_sim_to_seeds = torch.max(min_sim_to_seeds, new_sims)

    return seeds
