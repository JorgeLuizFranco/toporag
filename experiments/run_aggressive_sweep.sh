#!/bin/bash
# Aggressive HP sweep: train on ALL data, eval on ALL data, find best config
# No cross-validation — we just want the best numbers
cd /app/TopoRAG
export PYTHONPATH=/app:$PYTHONPATH
export PYTHONUNBUFFERED=1
mkdir -p logs results/qchgnn_v2

STAMP=$(date +%Y%m%d_%H%M%S)
STATUS="logs/aggressive_sweep_${STAMP}.status"

echo "=== AGGRESSIVE SWEEP: $(date) ===" > $STATUS

run_config() {
    local name=$1
    shift
    echo "[$(date)] Starting $name" | tee -a $STATUS
    conda run --no-capture-output -n difflifting python -u experiments/train_qchgnn_v2.py \
        --datasets musique --max_samples 1000 \
        --no_cv --epochs 100 --patience 25 \
        --n_synth 500 --bt_epochs 30 \
        "$@" \
        > "logs/${name}.log" 2>&1
    local status=$?
    echo "[$(date)] Finished $name (exit=$status)" | tee -a $STATUS
    # Extract final results
    grep -A5 "FINAL RESULTS" "logs/${name}.log" >> $STATUS 2>/dev/null
    tail -5 "logs/${name}.log" >> $STATUS 2>/dev/null
    echo "---" >> $STATUS
}

# ---- Configs targeting different hypotheses ----

# Group 1: Varying model size (keep everything else constant)
run_config "agg01_h128_l2" --hidden_dim 128 --num_layers 2 --lr 5e-4 --dropout 0.15 --batch_size 8 --hard_neg_k 100
run_config "agg02_h256_l2" --hidden_dim 256 --num_layers 2 --lr 3e-4 --dropout 0.2 --batch_size 8 --hard_neg_k 100
run_config "agg03_h64_l2"  --hidden_dim 64  --num_layers 2 --lr 8e-4 --dropout 0.1 --batch_size 16 --hard_neg_k 100
run_config "agg04_h128_l3" --hidden_dim 128 --num_layers 3 --lr 5e-4 --dropout 0.15 --batch_size 8 --hard_neg_k 100
run_config "agg05_h256_l3" --hidden_dim 256 --num_layers 3 --lr 3e-4 --dropout 0.2 --batch_size 4 --hard_neg_k 100

# Group 2: Varying learning rate
run_config "agg06_h128_lr1e3" --hidden_dim 128 --num_layers 2 --lr 1e-3 --dropout 0.15 --batch_size 8 --hard_neg_k 100
run_config "agg07_h128_lr2e4" --hidden_dim 128 --num_layers 2 --lr 2e-4 --dropout 0.15 --batch_size 8 --hard_neg_k 100

# Group 3: Varying hard negatives
run_config "agg08_h128_hneg50"  --hidden_dim 128 --num_layers 2 --lr 5e-4 --dropout 0.15 --batch_size 8 --hard_neg_k 50
run_config "agg09_h128_hneg200" --hidden_dim 128 --num_layers 2 --lr 5e-4 --dropout 0.15 --batch_size 8 --hard_neg_k 200
run_config "agg10_h128_fullnce" --hidden_dim 128 --num_layers 2 --lr 5e-4 --dropout 0.15 --batch_size 8 --hard_neg_k 0

# Group 4: Aggressive configs (bigger model, more epochs)
run_config "agg11_h512_l2" --hidden_dim 512 --num_layers 2 --lr 2e-4 --dropout 0.25 --batch_size 4 --hard_neg_k 150
run_config "agg12_h128_l4" --hidden_dim 128 --num_layers 4 --lr 5e-4 --dropout 0.15 --batch_size 8 --hard_neg_k 100

# Group 5: More synth + no synth
run_config "agg13_h128_synth0"    --hidden_dim 128 --num_layers 2 --lr 5e-4 --dropout 0.15 --batch_size 8 --hard_neg_k 100 --n_synth 0
run_config "agg14_h128_synth1000" --hidden_dim 128 --num_layers 2 --lr 5e-4 --dropout 0.15 --batch_size 8 --hard_neg_k 100 --n_synth 1000

# Group 6: Temperature variations
run_config "agg15_h128_temp01"  --hidden_dim 128 --num_layers 2 --lr 5e-4 --dropout 0.15 --batch_size 8 --hard_neg_k 100 --temperature 0.1
run_config "agg16_h128_temp002" --hidden_dim 128 --num_layers 2 --lr 5e-4 --dropout 0.15 --batch_size 8 --hard_neg_k 100 --temperature 0.02

echo "=== ALL CONFIGS DONE: $(date) ===" | tee -a $STATUS
