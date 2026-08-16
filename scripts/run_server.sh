#!/usr/bin/env bash
# Start the extraction server. Fails fast with a helpful message if the two
# model checkpoints are missing under models/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NODE_DIR="$HERE/models/mentalkg-xlmr-node"
EDGE_DIR="$HERE/models/mentalkg-xlmr-edge"

for D in "$NODE_DIR" "$EDGE_DIR"; do
  if [ ! -f "$D/config.json" ]; then
    echo "missing model at $D"
    echo "run: python $HERE/scripts/download_models.py"
    exit 1
  fi
done

cd "$HERE/server"
PY="${PYTHON:-python3}"
exec "$PY" -m uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}" "$@"
