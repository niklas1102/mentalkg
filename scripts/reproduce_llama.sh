#!/usr/bin/env bash
# Reproduce the faithful Llama-3.2-3B zero-shot baseline via Ollama.
# Requires ollama running locally with llama3.2:3b pulled.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v ollama >/dev/null; then
  echo "ollama not found. install from https://ollama.com and run:  ollama pull llama3.2:3b"
  exit 1
fi
if ! ollama list | grep -q '^llama3.2:3b'; then
  echo "model not pulled. run:  ollama pull llama3.2:3b"
  exit 1
fi

cd "$HERE"
python modeling/baselines/llama_v2.py node --backend ollama --split test
python modeling/baselines/llama_v2.py edge --backend ollama --per-cell 1000
echo "done. results under results/llama_v2/"
