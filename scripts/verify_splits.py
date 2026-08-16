#!/usr/bin/env python3
"""Check participant disjointness, language stratification, and per-label
train support of the prepared node splits.

Reads:
  modeling/data_v2/splits.json
  modeling/data_v2/node_prediction/{train,dev,test}.jsonl
  modeling/data_v2/node_prediction/labels.json

Usage:
    python scripts/verify_splits.py [--data-dir modeling/data_v2]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO / "modeling" / "data_v2"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA))
    args = p.parse_args()

    root = Path(args.data_dir)
    splits = json.loads((root / "splits.json").read_text())
    labels = json.loads((root / "node_prediction" / "labels.json").read_text())

    part = splits["participants"]
    train_p, dev_p, test_p = set(part["train"]), set(part["dev"]), set(part["test"])
    disjoint = train_p.isdisjoint(dev_p) and train_p.isdisjoint(test_p) and dev_p.isdisjoint(test_p)

    train = read_jsonl(root / "node_prediction" / "train.jsonl")
    dev = read_jsonl(root / "node_prediction" / "dev.jsonl")
    test = read_jsonl(root / "node_prediction" / "test.jsonl")

    def lang_frac(exs):
        c = Counter(e["language"] for e in exs)
        n = sum(c.values())
        return {k: c[k] / n for k in ("en", "de")}, n

    ftr, ntr = lang_frac(train)
    fde, nde = lang_frac(dev)
    fte, nte = lang_frac(test)

    train_ex_pids = {e["participant_id"] for e in train}
    dev_ex_pids = {e["participant_id"] for e in dev}
    test_ex_pids = {e["participant_id"] for e in test}
    ex_disjoint = (train_ex_pids.isdisjoint(dev_ex_pids)
                   and train_ex_pids.isdisjoint(test_ex_pids)
                   and dev_ex_pids.isdisjoint(test_ex_pids))

    support = labels["train_support"]
    common = labels["common_labels"]
    below = [l for l in common if support.get(l, 0) < 20]

    print("=== splits.json ===")
    print(f"participants train/dev/test: {len(train_p)}/{len(dev_p)}/{len(test_p)}")
    print(f"participant disjointness (splits.json): {'PASS' if disjoint else 'FAIL'}")
    print(f"participant disjointness (examples):   {'PASS' if ex_disjoint else 'FAIL'}")
    print()
    print("=== language stratification (node examples) ===")
    print(f"train n={ntr}  en={ftr['en']:.3f}  de={ftr['de']:.3f}")
    print(f"dev   n={nde}  en={fde['en']:.3f}  de={fde['de']:.3f}")
    print(f"test  n={nte}  en={fte['en']:.3f}  de={fte['de']:.3f}")
    max_spread = max(abs(ftr["en"] - fde["en"]),
                     abs(ftr["en"] - fte["en"]),
                     abs(fde["en"] - fte["en"]))
    print(f"max EN-fraction spread across splits: {max_spread:.4f} "
          f"({'PASS' if max_spread < 0.02 else 'FAIL'} <0.02)")
    print()
    print("=== label support (train) ===")
    print(f"vocab labels: {labels['num_labels']}")
    print(f"common labels (>= 20 train support): {labels['num_common_labels']}")
    print(f"common labels actually below 20: {len(below)} "
          f"({'PASS' if not below else 'FAIL'})")

    ok = disjoint and ex_disjoint and max_spread < 0.02 and not below
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
