#!/usr/bin/env bash
# Reproduce the TF-IDF node baseline (CPU, ~minutes).
# Requires modeling/data_v2/ to exist. If missing, run:
#   python modeling/prepare_data.py --negatives hard --output modeling/data_v2
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ ! -d "$HERE/modeling/data_v2/node_prediction" ]; then
  echo "modeling/data_v2/node_prediction/ missing"
  echo "run: python $HERE/modeling/prepare_data.py --negatives hard --output modeling/data_v2"
  exit 1
fi

cd "$HERE"
python modeling/baselines/node_baseline.py --language all
python modeling/baselines/node_baseline.py --language en
python modeling/baselines/node_baseline.py --language de
echo "done. artifacts under modeling/baselines/artifacts_v2/node_{all,en,de}/"
