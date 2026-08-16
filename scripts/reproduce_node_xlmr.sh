#!/usr/bin/env bash
# Reproduce the XLM-R node checkpoint with the winning config.
# CUDA GPU required for the full run in reasonable time (RTX 4090 ~2 h).
# CPU or MPS: exits cleanly after warning.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  echo "no CUDA GPU detected. The XLM-R full run (20k train × 9 epochs) takes"
  echo "many hours on CPU/MPS and is not part of the on-Mac verification path."
  echo "Use the released checkpoints instead: python $HERE/scripts/download_models.py"
  exit 0
fi

cd "$HERE"
python modeling/baselines/node_bert.py \
  --language all \
  --data-dir modeling/data_v2/node_prediction \
  --art-dir modeling/baselines/artifacts_v2_bert/node_xlmr_all_rerun \
  --head-bias-init --lr 3e-5 --epochs 9 --seed 42
echo "done. metrics under modeling/baselines/artifacts_v2_bert/node_xlmr_all_rerun/"
