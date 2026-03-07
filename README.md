# TopoRAG

**Preferential Higher-Order Networks for Reliable Retrieval**

A novel RAG framework that lifts chunk graphs to higher-order topological structures (hypergraphs) and trains a two-level retrieval system using Topological Neural Networks.

## Architecture

**Level 1 — Chunk Recall**: Scores individual chunks via gated cosine residual on TNN-refined embeddings. High recall, filters candidates.

**Level 2 — Cell Precision**: Constructs dynamic cells (chunk subsets) from Level 1 candidates, scores them with DeepSet + MLP link predictor. Boosts chunks that form coherent multi-hop groups.

```
L1: s(q, ci) = cos(q_raw, ci_raw) + tanh(gate) * cos(q_tnn, ci_tnn)
L2: psi(q, sigma) = MLP([query_proj(q_tnn); DeepSet(ci_tnn for ci in sigma)])
Final: score = s_L1 + weight * normalized_cell_boost
```

## Results (MuSiQue 50-sample subset)

| Method | R@2 | R@5 |
|--------|-----|-----|
| Cosine baseline | 42.8% | 57.2% |
| TopoRAG L1 (chunk only) | 41.7% | 58.5% |
| **TopoRAG L1+L2 (two-level)** | **47.7%** | **63.7%** |
| GFM-RAG (SOTA) | 49.1% | 58.2% |

## Setup

```bash
git clone https://github.com/JorgeLuizFranco/toporag.git
cd toporag
pip install -r requirements.txt
```

## Training

```bash
# Level 1 only (backward compatible)
python experiments/train_toporag.py \
  --query_cache experiments/cache/musique_50_queries.json \
  --gold_train --epochs 100 --batch_size 16

# Two-level (Level 1 + Level 2 cell scoring)
python experiments/train_toporag.py \
  --query_cache experiments/cache/musique_50_queries.json \
  --gold_train --epochs 100 --batch_size 16 \
  --lambda_l2 1.0 --k1_candidates 20 --cell_size 2 --l2_eval_weight 0.1
```

**Dataset setup**: Place MuSiQue dataset at `../LPGNN-retriever/datasets/musique/musique.json` (relative to this repo).

## Project Structure

```
toporag/
  models/          # TNN, LPTNN, cell encoder, link predictor
  lifting/         # k-NN, clique, cycle hypergraph lifting
  llms/            # LLM backends (local, Groq, OpenAI)
  utils/           # Embeddings, data loading
  evaluation/      # Metrics
  experiments/     # Training scripts
  toporag.py       # Main pipeline builder
  pipeline.py      # Retrieval pipeline
```
