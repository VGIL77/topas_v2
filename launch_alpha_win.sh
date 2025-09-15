#!/bin/bash
# Alpha-ARC X Neural-Guided Search 2.0 Launch Script
# Using the alpha_win.json winning configuration

echo "🚀 Launching Alpha-ARC X Neural-Guided Search 2.0"
echo "📊 Configuration: configs/alpha_win.json"
echo "🎯 Target: Systematic breakthrough performance with PUCT + SC-STaR + Near-Miss"

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="alpha_arc_x_${TIMESTAMP}.log"

echo "📝 Logging to: ${LOG_FILE}"
echo ""

# Launch the training with your winning configuration
venv/bin/python train_simple_hrm_topas.py \
  --device cuda \
  --dataset arc2 \
  --epochs 150 \
  --lr 2e-05 \
  --grad-clip 0.5 \
  --eval-interval 5 \
  --enable-dream \
  --dream-micro-ticks 1 \
  --dream-full-every 10 \
  --dream-pretrain-epochs 3 \
  --dream-pretrain-freeze-model \
  --search-alg puct \
  --puct-nodes 1500 \
  --puct-depth 6 \
  --c-puct 1.25 \
  --root-dirichlet-alpha 0.3 \
  --root-dirichlet-eps 0.25 \
  --sc-star \
  --replay-cap 100000 \
  --near-miss-hamming-pct 5.0 \
  --breakthrough-threshold 0.33 \
  --nightmare-alpha 0.08 \
  --nightmare-min-interval 200 \
  --nightmare-max-interval 1000 \
  --relmem-enable \
  --relmem-reg-alpha 0.001 \
  --relmem-reg-beta 0.0005 \
  --relmem-bind-iou 0.25 \
  --relmem-bias-ramp-start 10 \
  --relmem-bias-max 0.5 \
  --selfplay-enable \
  --selfplay-interval 250 \
  --selfplay-weight 0.1 \
  --selfplay-topk 3 \
  --selfplay-buffer-size 200 \
  --monologue-interval 200 \
  --monologue-consistency-target 0.85 \
  --monologue-selfplay-bonus 0.05 \
  --model-width 640 \
  --model-slots 64 \
  --batch-size 1 \
  --verbose 2>&1 | tee "${LOG_FILE}"

echo ""
echo "🎉 Alpha-ARC X training completed!"
echo "📊 Results logged to: ${LOG_FILE}"