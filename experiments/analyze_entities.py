#!/usr/bin/env python3
"""
Analyze WHY gold chunk pairs aren't connected by entity cells.

For each MuSiQue question with 2+ supporting chunks, checks whether the gold
pair shares an entity cell. For missed pairs (no shared cell), diagnoses
the cause:
  - Substring match (e.g., "Obama" in "Barack Obama")
  - Lemmatized match (e.g., "United States" vs "United States")
  - Fuzzy match (ratio > 0.8, e.g., "Spielberg" vs "Steven Spielberg")
  - No entity overlap at all

Also prints cell size distribution.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations
from typing import Dict, List, Set, Tuple

# ── Setup paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "LPGNN-retriever/datasets/musique/musique.json"
TOPO_CACHE = PROJECT_ROOT / "toporag/experiments/cache/topology/musique_500_entity.pt"
MAX_SAMPLES = 500


def load_data():
    """Load MuSiQue dataset (first MAX_SAMPLES), returning chunks and samples."""
    with open(DATA_PATH) as f:
        data = json.load(f)

    chunks, chunk_to_doc, samples = [], [], []

    for q_idx, item in enumerate(data[:MAX_SAMPLES]):
        paragraphs = item.get("paragraphs", [])
        local_indices = []

        for p in paragraphs:
            text = f"{p.get('title', '')}: {p.get('paragraph_text', '')}"
            global_idx = len(chunks)
            chunks.append(text)
            chunk_to_doc.append(q_idx)
            local_indices.append(global_idx)

        supporting_global = [
            local_indices[i]
            for i, p in enumerate(paragraphs)
            if p.get("is_supporting", False) and i < len(local_indices)
        ]

        samples.append({
            "question": item["question"],
            "answer": item.get("answer", ""),
            "all_chunks": local_indices,
            "supporting": supporting_global,
        })

    return chunks, chunk_to_doc, samples


def extract_entities_per_chunk(chunks: List[str]) -> Dict[int, Set[str]]:
    """Extract named entities per chunk using spaCy (same as entity.py)."""
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

    entity_types = {
        "PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT",
        "WORK_OF_ART", "PRODUCT", "NORP", "LAW",
    }

    chunk_entities: Dict[int, Set[str]] = defaultdict(set)

    batch_size = 500
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start:batch_start + batch_size]
        batch_truncated = [c[:500] for c in batch]
        docs = nlp.pipe(batch_truncated, batch_size=64)

        for local_idx, doc in enumerate(docs):
            chunk_idx = batch_start + local_idx
            for ent in doc.ents:
                if ent.label_ in entity_types:
                    key = ent.text.strip().lower()
                    if len(key) >= 2:
                        chunk_entities[chunk_idx].add(key)

    return dict(chunk_entities)


def extract_entities_with_lemma(chunks: List[str]) -> Dict[int, Set[str]]:
    """Extract entities WITH lemmatization enabled."""
    import spacy
    nlp = spacy.load("en_core_web_sm")  # lemmatizer enabled

    entity_types = {
        "PERSON", "ORG", "GPE", "LOC", "FAC", "EVENT",
        "WORK_OF_ART", "PRODUCT", "NORP", "LAW",
    }

    chunk_lemma_entities: Dict[int, Set[str]] = defaultdict(set)

    batch_size = 500
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start:batch_start + batch_size]
        batch_truncated = [c[:500] for c in batch]
        docs = nlp.pipe(batch_truncated, batch_size=64)

        for local_idx, doc in enumerate(docs):
            chunk_idx = batch_start + local_idx
            for ent in doc.ents:
                if ent.label_ in entity_types:
                    # Use lemma of each token in the entity span
                    lemma_key = " ".join(tok.lemma_.lower() for tok in ent).strip()
                    if len(lemma_key) >= 2:
                        chunk_lemma_entities[chunk_idx].add(lemma_key)

    return dict(chunk_lemma_entities)


def build_entity_to_chunks(chunk_entities: Dict[int, Set[str]]) -> Dict[str, Set[int]]:
    """Invert: entity -> set of chunk indices."""
    entity_to_chunks: Dict[str, Set[int]] = defaultdict(set)
    for chunk_idx, entities in chunk_entities.items():
        for ent in entities:
            entity_to_chunks[ent].add(chunk_idx)
    return dict(entity_to_chunks)


def chunks_share_cell(
    c1: int,
    c2: int,
    entity_to_chunks: Dict[str, Set[int]],
    min_size: int = 2,
    max_size: int = 10,
) -> bool:
    """Check if two chunks appear together in any valid entity cell."""
    for ent, chunk_set in entity_to_chunks.items():
        if min_size <= len(chunk_set) <= max_size:
            if c1 in chunk_set and c2 in chunk_set:
                return True
    return False


def check_substring_match(ents_a: Set[str], ents_b: Set[str]) -> List[Tuple[str, str]]:
    """Find pairs where one entity is a substring of another."""
    matches = []
    for ea in ents_a:
        for eb in ents_b:
            if ea == eb:
                continue
            if ea in eb or eb in ea:
                matches.append((ea, eb))
    return matches


def check_fuzzy_match(ents_a: Set[str], ents_b: Set[str], threshold: float = 80.0) -> List[Tuple[str, str, float]]:
    """Find pairs with high fuzzy similarity (ratio > threshold)."""
    from rapidfuzz import fuzz
    matches = []
    for ea in ents_a:
        for eb in ents_b:
            if ea == eb:
                continue
            ratio = fuzz.ratio(ea, eb)
            if ratio >= threshold:
                matches.append((ea, eb, ratio))
    return matches


def main():
    import torch

    print("=" * 70)
    print("Entity Connectivity Analysis for Gold MuSiQue Pairs")
    print("=" * 70)

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("\n[1/6] Loading dataset...")
    chunks, chunk_to_doc, samples = load_data()
    print(f"  {len(chunks)} chunks from {len(samples)} questions")

    # ── 2. Load cached topology ──────────────────────────────────────────
    print(f"\n[2/6] Loading cached topology from {TOPO_CACHE}...")
    cache_data = torch.load(TOPO_CACHE, weights_only=False)
    cell_to_nodes = cache_data["cell_to_nodes"]
    lifted = cache_data["lifted"]
    print(f"  {lifted.num_nodes} nodes, {lifted.num_edges} cells")

    # ── 3. Cell size distribution ────────────────────────────────────────
    print(f"\n[3/6] Cell size distribution:")
    sizes = [len(v) for v in cell_to_nodes.values()]
    size_counts = Counter(sizes)
    total_cells = len(sizes)
    for sz in sorted(size_counts.keys()):
        cnt = size_counts[sz]
        pct = 100 * cnt / total_cells
        bar = "#" * int(pct)
        print(f"  Size {sz:2d}: {cnt:5d} cells ({pct:5.1f}%) {bar}")
    print(f"  Total: {total_cells} cells")
    print(f"  Mean size: {sum(sizes)/len(sizes):.2f}")

    # ── 4. Extract entities ──────────────────────────────────────────────
    print(f"\n[4/6] Extracting entities from chunks (spaCy NER)...")
    chunk_entities = extract_entities_per_chunk(chunks)
    entity_to_chunks = build_entity_to_chunks(chunk_entities)
    total_entities = len(entity_to_chunks)
    valid_entities = {e: cs for e, cs in entity_to_chunks.items() if 2 <= len(cs) <= 10}
    print(f"  {total_entities} unique entities extracted")
    print(f"  {len(valid_entities)} valid entities (size 2-10)")
    print(f"  {len(chunk_entities)}/{len(chunks)} chunks have at least one entity "
          f"({100*len(chunk_entities)/len(chunks):.1f}%)")

    # Chunks with NO entities at all
    no_entity_chunks = set(range(len(chunks))) - set(chunk_entities.keys())
    print(f"  {len(no_entity_chunks)} chunks have ZERO entities")

    # ── 5. Analyze gold pairs ────────────────────────────────────────────
    print(f"\n[5/6] Analyzing gold chunk pair connectivity...")

    gold_pairs = []
    for s in samples:
        sup = s["supporting"]
        if len(sup) >= 2:
            for c1, c2 in combinations(sup, 2):
                gold_pairs.append((c1, c2))
    print(f"  {len(gold_pairs)} gold chunk pairs from {len(samples)} questions")

    # Check connectivity via entity cells
    connected_pairs = []
    disconnected_pairs = []

    for c1, c2 in gold_pairs:
        if chunks_share_cell(c1, c2, entity_to_chunks):
            connected_pairs.append((c1, c2))
        else:
            disconnected_pairs.append((c1, c2))

    n_conn = len(connected_pairs)
    n_disc = len(disconnected_pairs)
    n_total = len(gold_pairs)
    print(f"\n  Connected (share entity cell): {n_conn}/{n_total} ({100*n_conn/n_total:.1f}%)")
    print(f"  Disconnected (NO shared cell): {n_disc}/{n_total} ({100*n_disc/n_total:.1f}%)")

    # ── 6. Diagnose disconnected pairs ───────────────────────────────────
    print(f"\n[6/6] Diagnosing {n_disc} disconnected gold pairs...")

    # Diagnostic categories
    no_entities_either = 0      # At least one chunk has no entities
    no_entities_one = 0         # Exactly one chunk has no entities
    exact_match_exists = 0      # Exact entity match but filtered by size
    substring_recoverable = 0   # Would connect via substring match
    lemma_recoverable = 0       # Would connect via lemmatization
    fuzzy_recoverable = 0       # Would connect via fuzzy match (>80%)
    truly_disconnected = 0      # No entity overlap at all

    # For detailed examples
    substring_examples = []
    fuzzy_examples = []
    exact_but_filtered = []
    lemma_examples = []
    no_overlap_examples = []

    # Also extract lemmatized entities
    print("  Extracting lemmatized entities (this takes a moment)...")
    chunk_lemma_entities = extract_entities_with_lemma(chunks)
    lemma_entity_to_chunks = build_entity_to_chunks(chunk_lemma_entities)

    for c1, c2 in disconnected_pairs:
        ents1 = chunk_entities.get(c1, set())
        ents2 = chunk_entities.get(c2, set())

        # Case: missing entities
        if not ents1 and not ents2:
            no_entities_either += 1
            continue
        if not ents1 or not ents2:
            no_entities_one += 1
            continue

        # Case: exact entity match but cell filtered out (size > 10 or < 2)
        shared_exact = ents1 & ents2
        if shared_exact:
            # They share an entity, but no valid cell includes both
            # Check if the entity cell was filtered by size
            for ent in shared_exact:
                all_chunks_with_ent = entity_to_chunks.get(ent, set())
                if len(all_chunks_with_ent) > 10:
                    exact_but_filtered.append((c1, c2, ent, len(all_chunks_with_ent)))
            exact_match_exists += 1
            continue

        # Case: substring match
        sub_matches = check_substring_match(ents1, ents2)
        if sub_matches:
            substring_recoverable += 1
            if len(substring_examples) < 10:
                substring_examples.append((c1, c2, sub_matches[:3]))
            continue

        # Case: lemmatized match
        lemma1 = chunk_lemma_entities.get(c1, set())
        lemma2 = chunk_lemma_entities.get(c2, set())
        shared_lemma = lemma1 & lemma2
        if shared_lemma and not (ents1 & ents2):
            lemma_recoverable += 1
            if len(lemma_examples) < 10:
                lemma_examples.append((c1, c2, list(shared_lemma)[:3]))
            continue

        # Case: fuzzy match
        fuz_matches = check_fuzzy_match(ents1, ents2, threshold=80.0)
        if fuz_matches:
            fuzzy_recoverable += 1
            if len(fuzzy_examples) < 10:
                fuzzy_examples.append((c1, c2, fuz_matches[:3]))
            continue

        # Truly disconnected: no overlap at all
        truly_disconnected += 1
        if len(no_overlap_examples) < 10:
            no_overlap_examples.append((c1, c2, list(ents1)[:5], list(ents2)[:5]))

    # ── Print results ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS: Why gold pairs are disconnected")
    print("=" * 70)
    print(f"\nTotal gold pairs:        {n_total}")
    print(f"Connected (shared cell): {n_conn} ({100*n_conn/n_total:.1f}%)")
    print(f"Disconnected:            {n_disc} ({100*n_disc/n_total:.1f}%)")
    print()

    print("Breakdown of disconnected pairs:")
    print(f"  Neither chunk has entities:        {no_entities_either:4d} ({100*no_entities_either/max(n_disc,1):.1f}%)")
    print(f"  One chunk has no entities:          {no_entities_one:4d} ({100*no_entities_one/max(n_disc,1):.1f}%)")
    print(f"  Exact match but cell filtered:     {exact_match_exists:4d} ({100*exact_match_exists/max(n_disc,1):.1f}%)")
    print(f"  Recoverable by SUBSTRING:          {substring_recoverable:4d} ({100*substring_recoverable/max(n_disc,1):.1f}%)")
    print(f"  Recoverable by LEMMATIZATION:      {lemma_recoverable:4d} ({100*lemma_recoverable/max(n_disc,1):.1f}%)")
    print(f"  Recoverable by FUZZY (>80%):       {fuzzy_recoverable:4d} ({100*fuzzy_recoverable/max(n_disc,1):.1f}%)")
    print(f"  Truly disconnected (no overlap):   {truly_disconnected:4d} ({100*truly_disconnected/max(n_disc,1):.1f}%)")

    accum_recovered = substring_recoverable + lemma_recoverable + fuzzy_recoverable + exact_match_exists
    print(f"\n  TOTAL potentially recoverable:     {accum_recovered:4d} ({100*accum_recovered/max(n_disc,1):.1f}% of disconnected)")
    new_connected = n_conn + accum_recovered
    print(f"  New connectivity rate:             {new_connected}/{n_total} ({100*new_connected/n_total:.1f}%)")

    # ── Examples ─────────────────────────────────────────────────────────
    if exact_but_filtered:
        print(f"\n{'─'*70}")
        print(f"Examples: Exact match but cell filtered (entity too popular):")
        for c1, c2, ent, sz in exact_but_filtered[:5]:
            print(f"  Chunks {c1},{c2}: entity '{ent}' appears in {sz} chunks (>10, filtered)")

    if substring_examples:
        print(f"\n{'─'*70}")
        print(f"Examples: Substring matches:")
        for c1, c2, matches in substring_examples[:5]:
            for ea, eb in matches[:2]:
                print(f"  Chunks {c1},{c2}: '{ea}' <-> '{eb}'")

    if lemma_examples:
        print(f"\n{'─'*70}")
        print(f"Examples: Lemmatization matches:")
        for c1, c2, shared in lemma_examples[:5]:
            print(f"  Chunks {c1},{c2}: shared lemmas = {shared}")

    if fuzzy_examples:
        print(f"\n{'─'*70}")
        print(f"Examples: Fuzzy matches (>80% ratio):")
        for c1, c2, matches in fuzzy_examples[:5]:
            for ea, eb, ratio in matches[:2]:
                print(f"  Chunks {c1},{c2}: '{ea}' <-> '{eb}' (ratio={ratio:.1f})")

    if no_overlap_examples:
        print(f"\n{'─'*70}")
        print(f"Examples: Truly disconnected (no entity overlap):")
        for c1, c2, e1, e2 in no_overlap_examples[:5]:
            print(f"  Chunk {c1} entities: {e1}")
            print(f"  Chunk {c2} entities: {e2}")
            # Show chunk text snippets
            print(f"    Text A: {chunks[c1][:120]}...")
            print(f"    Text B: {chunks[c2][:120]}...")
            print()

    # ── Additional: entity type breakdown for missing pairs ──────────────
    print(f"\n{'='*70}")
    print("Entity type distribution in gold supporting chunks")
    print("=" * 70)

    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

    # Collect entity types for supporting chunks
    supporting_chunk_ids = set()
    for s in samples:
        supporting_chunk_ids.update(s["supporting"])

    type_counts = Counter()
    for chunk_idx in sorted(supporting_chunk_ids):
        doc = nlp(chunks[chunk_idx][:500])
        for ent in doc.ents:
            type_counts[ent.label_] += 1

    for label, count in type_counts.most_common():
        print(f"  {label:15s}: {count:5d}")

    # ── Summary stats ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"Gold pair connectivity rate (current):  {100*n_conn/n_total:.1f}%")
    if n_disc > 0:
        print(f"Would-be rate with enhancements:        {100*new_connected/n_total:.1f}%")
        print(f"  +exact-but-filtered recovery:         +{exact_match_exists} pairs")
        print(f"  +substring normalization:             +{substring_recoverable} pairs")
        print(f"  +lemmatization:                       +{lemma_recoverable} pairs")
        print(f"  +fuzzy matching (>80%):               +{fuzzy_recoverable} pairs")
        remaining = n_disc - accum_recovered
        print(f"Still disconnected after all fixes:     {remaining} pairs ({100*remaining/n_total:.1f}%)")
        print(f"  Of which: no entities = {no_entities_either + no_entities_one}, truly no overlap = {truly_disconnected}")


if __name__ == "__main__":
    main()
