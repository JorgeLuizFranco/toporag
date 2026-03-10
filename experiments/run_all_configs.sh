#!/bin/bash
# Sequential config runner - runs one config at a time
# Each config: train on ALL data, eval on ALL data, 100 epochs + 30 BT epochs
set -e
cd /app/TopoRAG
export PYTHONPATH=/app:$PYTHONPATH
export PYTHONUNBUFFERED=1
mkdir -p logs results/qchgnn_v2

STATUS="logs/sweep_master.status"

run() {
    local name=$1; shift
    echo "[$(date)] START $name" >> $STATUS
    conda run --no-capture-output -n difflifting python -u experiments/train_qchgnn_v2.py \
        --datasets musique --max_samples 1000 --no_cv --epochs 100 --patience 25 \
        --n_synth 500 --bt_epochs 30 \
        "$@" > "logs/${name}.log" 2>&1 || true
    echo "[$(date)] DONE $name" >> $STATUS
    # Append result summary
    grep -E "R@2|R@5|FINAL|QCHGNN|Cosine|BT result" "logs/${name}.log" | tail -10 >> $STATUS 2>/dev/null
    echo "---" >> $STATUS
}

echo "=== SWEEP START $(date) ===" > $STATUS

# --- Already running: agg01 will finish first, then this script picks up ---

run "agg02_h256_l2" --hidden_dim 256 --num_layers 2 --lr 3e-4 --dropout 0.2 --batch_size 8 --hard_neg_k 100
run "agg03_h64_l2"  --hidden_dim 64  --num_layers 2 --lr 8e-4 --dropout 0.1 --batch_size 16 --hard_neg_k 100
run "agg04_h128_l3" --hidden_dim 128 --num_layers 3 --lr 5e-4 --dropout 0.15 --batch_size 8 --hard_neg_k 100
run "agg05_h256_l3" --hidden_dim 256 --num_layers 3 --lr 3e-4 --dropout 0.2 --batch_size 4 --hard_neg_k 100
run "agg06_lr1e3"   --hidden_dim 128 --num_layers 2 --lr 1e-3 --dropout 0.15 --batch_size 8 --hard_neg_k 100
run "agg07_lr2e4"   --hidden_dim 128 --num_layers 2 --lr 2e-4 --dropout 0.15 --batch_size 8 --hard_neg_k 100
run "agg08_hneg50"  --hidden_dim 128 --num_layers 2 --lr 5e-4 --dropout 0.15 --batch_size 8 --hard_neg_k 50
run "agg09_hneg200" --hidden_dim 128 --num_layers 2 --lr 5e-4 --dropout 0.15 --batch_size 8 --hard_neg_k 200
run "agg10_fullnce" --hidden_dim 128 --num_layers 2 --lr 5e-4 --dropout 0.15 --batch_size 8 --hard_neg_k 0
run "agg11_h512_l2" --hidden_dim 512 --num_layers 2 --lr 2e-4 --dropout 0.25 --batch_size 4 --hard_neg_k 150
run "agg12_h128_l4" --hidden_dim 128 --num_layers 4 --lr 5e-4 --dropout 0.15 --batch_size 8 --hard_neg_k 100
run "agg13_nosynth"  --hidden_dim 128 --num_layers 2 --lr 5e-4 --dropout 0.15 --batch_size 8 --hard_neg_k 100 --n_synth 0
run "agg14_synth1k"  --hidden_dim 128 --num_layers 2 --lr 5e-4 --dropout 0.15 --batch_size 8 --hard_neg_k 100 --n_synth 1000
run "agg15_temp01"   --hidden_dim 128 --num_layers 2 --lr 5e-4 --dropout 0.15 --batch_size 8 --hard_neg_k 100 --temperature 0.1
run "agg16_temp002"  --hidden_dim 128 --num_layers 2 --lr 5e-4 --dropout 0.15 --batch_size 8 --hard_neg_k 100 --temperature 0.02

echo "=== ALL DONE $(date) ===" >> $STATUS
