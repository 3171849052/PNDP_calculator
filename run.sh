#!/bin/bash

# ============================================================
# MNIST Decentralized Training - Parameter Settings & Launcher
# ============================================================

# ---------- Parameters (edit here) ----------
DATASET="MNIST"
GRAPH="florentine_families"
BATCH_SIZE=256
MAX_PHYSICAL_BATCH_SIZE=64
T_LOCAL_STEPS=20
R_ROUNDS=20
K_GOSSIP=1
EPSILON=3.0
DELTA=1e-5
CLIP_NORM=1
LR=1e-3
GPU=2
FRAMEWORK="GDP"
ALGORITHM="PNDP_strict"
ENABLE_PRIVACY=true
SET_NM=None
CACHE_NOISE=true
# --------------------------------------------

export DATASET GRAPH BATCH_SIZE MAX_PHYSICAL_BATCH_SIZE T_LOCAL_STEPS R_ROUNDS K_GOSSIP
export EPSILON DELTA CLIP_NORM LR GPU FRAMEWORK ALGORITHM ENABLE_PRIVACY SET_NM CACHE_NOISE

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="exps/${TIMESTAMP}_${DATASET}_${GRAPH}_E${EPSILON}_DP_${ENABLE_PRIVACY}_R${R_ROUNDS}_T${T_LOCAL_STEPS}_B${BATCH_SIZE}_K${K_GOSSIP}_CN${CLIP_NORM}_LR${LR}_A${ALGORITHM}_F${FRAMEWORK}"
mkdir -p "$OUT_DIR"
export OUT_DIR

nohup python train_example_mnist.py > "$OUT_DIR/train.log" 2>&1 &
PID=$!
echo $PID > "$OUT_DIR/pid.txt"

echo "=========================================="
echo "kill $PID"
echo ""
echo "tail -f $OUT_DIR/train.log"
echo "=========================================="
