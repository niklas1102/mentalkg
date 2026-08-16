#!/usr/bin/env python3
"""Recompute the dataset stats table (feeds into data/README.md).

Usage:
    python scripts/stats.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "pipeline" / "output" / "dataset.jsonl"
POOL = REPO / "data" / "variable_pool.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=str(DATASET))
    p.add_argument("--pool", default=str(POOL))
    args = p.parse_args()

    path = Path(args.dataset)
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 2

    n = 0
    dec = Counter()
    lang = Counter()
    lang_acc = Counter()
    styles = Counter()
    ages = []
    nodes_per_entry = []
    edges_per_entry = []
    words_per_entry = []
    participants = set()
    scenarios = Counter()

    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            n += 1
            d = r.get("decision")
            dec[d] += 1
            lg = r.get("language") or r["participant"]["language"]
            lang[lg] += 1
            if d == "accept":
                lang_acc[lg] += 1
            styles[r["participant"].get("writing_style", "?")] += 1
            ages.append(r["participant"]["age"])
            participants.add(r["participant"]["participant_id"])
            scenarios[r.get("scenario", "?")] += 1
            nodes_per_entry.append(len(r["graph"]["nodes"]))
            edges_per_entry.append(len(r["graph"]["edges"]))
            words_per_entry.append(len(r["entry"]["text"].split()))

    pool = json.loads(Path(args.pool).read_text())
    concepts = pool["node_concepts"]
    types = Counter(c["type"] for c in concepts)
    polarities = Counter(c.get("polarity", "?") for c in concepts)

    def pct(v):
        v = sorted(v)
        def q(p): return v[int(round(p * (len(v) - 1)))]
        return q(0.5), q(0.9)

    n_p50, n_p90 = pct(nodes_per_entry)
    e_p50, e_p90 = pct(edges_per_entry)
    w_p50, w_p90 = pct(words_per_entry)

    print("=== corpus ===")
    print(f"samples: {n}   participants: {len(participants)}   "
          f"days/participant: 14")
    print(f"decisions: accept={dec['accept']}  review={dec['review']}  reject={dec['reject']}")
    print(f"language (all): en={lang['en']}  de={lang['de']}")
    print(f"language (accepted): en={lang_acc['en']}  de={lang_acc['de']}")
    print(f"age range: {min(ages)}–{max(ages)}  mean {sum(ages)/len(ages):.1f}")
    print(f"writing styles: {dict(styles.most_common())}")
    print(f"scenarios: {len(scenarios)}")
    print()
    print("=== per-entry ===")
    print(f"nodes/entry: p50={n_p50}  p90={n_p90}")
    print(f"edges/entry: p50={e_p50}  p90={e_p90}")
    print(f"words/entry: p50={w_p50}  p90={w_p90}")
    print()
    print("=== concept pool ===")
    print(f"concepts: {len(concepts)}")
    print(f"by type: {dict(types.most_common())}")
    print(f"by polarity: {dict(polarities.most_common())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
