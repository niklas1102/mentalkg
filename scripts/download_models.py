#!/usr/bin/env python3
"""Download the two trained MentalKG XLM-R checkpoints from Hugging Face into
models/. Idempotent: skips files already present with matching hash.

Usage:
    python scripts/download_models.py [--dest models]
        [--node Niklas1102/mentalkg-xlmr-node]
        [--edge Niklas1102/mentalkg-xlmr-edge]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DEST = REPO / "models"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dest", default=str(DEFAULT_DEST))
    p.add_argument("--node", default="Niklas1102/mentalkg-xlmr-node")
    p.add_argument("--edge", default="Niklas1102/mentalkg-xlmr-edge")
    args = p.parse_args()

    from huggingface_hub import snapshot_download

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    node_local = dest / "mentalkg-xlmr-node"
    edge_local = dest / "mentalkg-xlmr-edge"

    print(f"[download] {args.node} → {node_local}")
    snapshot_download(repo_id=args.node, local_dir=str(node_local))
    print(f"[download] {args.edge} → {edge_local}")
    snapshot_download(repo_id=args.edge, local_dir=str(edge_local))
    print("[done] models ready:")
    for d in (node_local, edge_local):
        for f in sorted(d.iterdir()):
            print(f"  {d.name}/{f.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
