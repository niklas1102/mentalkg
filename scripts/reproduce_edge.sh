#!/usr/bin/env bash
# Reproduce the XLM-R edge experiment: full data, with-text + no-text ablation,
# 3 seeds each. CUDA GPU required (RTX 4090 ~90 min per seed per arm).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "no CUDA GPU detected. The edge full-data 3-seed sweep takes many hours"
  echo "on CPU/MPS. Use the released checkpoints instead:"
  echo "  python $HERE/scripts/download_models.py"
  exit 0
fi

cd "$HERE"
for SEED in 42 43 44; do
  python modeling/baselines/edge_bert.py \
    --data-dir modeling/data_v2/edge_prediction \
    --art-dir "modeling/baselines/artifacts_v2_bert/edge_full_s${SEED}" \
    --tag "full_s${SEED}" \
    --train-entry-frac 1.0 --lr 2e-5 --epochs 5 --seed "$SEED"

  python modeling/baselines/edge_bert.py \
    --data-dir modeling/data_v2/edge_prediction \
    --art-dir "modeling/baselines/artifacts_v2_bert/edge_full_notext_s${SEED}" \
    --tag "full_notext_s${SEED}" \
    --train-entry-frac 1.0 --lr 2e-5 --epochs 5 --seed "$SEED" --no-text
done
echo "done."
