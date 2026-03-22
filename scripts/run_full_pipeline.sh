#!/bin/bash
# Full training pipeline: FFSA → LightGBM → Transformer/TCN
# Runs after build_features.py completes

set -e
cd /Users/mithunghosh/ClaudeCoding/stock-screening-agent/stockbot

echo "$(date) ═══ Step 1/3: FFSA (SHAP feature selection) ═══"
python scripts/run_ffsa.py --top-n 40 --sample-pct 10

echo ""
echo "$(date) ═══ Step 2/3: LightGBM retraining ═══"
python scripts/train_lgbm.py --top-n 40

echo ""
echo "$(date) ═══ Step 3/3: Transformer + TCN retraining ═══"
python scripts/train_models.py --top-n 40 --epochs 20 --warmup-epochs 5 --accum-steps 2 --thermal-sleep 15

echo ""
echo "$(date) ═══ PIPELINE COMPLETE ═══"
echo "Results saved to:"
echo "  - FFSA report:  reports/drift/ffsa_*.json"
echo "  - LightGBM:     models/lgbm/"
echo "  - Transformer:  models/transformer/"
echo "  - TCN:          models/tcn/"
