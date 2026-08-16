#!/usr/bin/env python3
"""Sanity-check pipeline/output/dataset.jsonl.

Reports line count, decision breakdown, language breakdown, connectivity audit
per node-type pair. Prints a PASS/FAIL table against the expected numbers.

Usage:
    python scripts/validate_dataset.py [--dataset pipeline/output/dataset.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO / "pipeline" / "output" / "dataset.jsonl"

EXPECTED = {
    "total": 47714,
    "decisions": {"accept": 41315, "review": 4950, "reject": 1449},
    "languages_accepted": {"en": 20982, "de": 20333},
    "participants": 3410,
}

LOW, HIGH = 0.15, 0.85


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    args = p.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        print(f"FAIL: dataset not found at {ds_path}", file=sys.stderr)
        return 2

    total = 0
    decisions = Counter()
    lang_all = Counter()
    lang_acc = Counter()
    participants = set()
    conn = defaultdict(lambda: {"cooccur": 0, "connected": 0})

    with ds_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            total += 1
            d = r.get("decision")
            decisions[d] += 1
            lang = r.get("language") or r["participant"]["language"]
            lang_all[lang] += 1
            participants.add(r["participant"]["participant_id"])
            if d == "accept":
                lang_acc[lang] += 1
                g = r.get("graph") or {}
                type_by = {n["node_id"]: n["type"] for n in g.get("nodes") or []}
                edge_set = set()
                for e in g.get("edges") or []:
                    a, b = e.get("source_node_id"), e.get("target_node_id")
                    if a in type_by and b in type_by and a != b:
                        edge_set.add(tuple(sorted([a, b])))
                for a, b in combinations(sorted(type_by), 2):
                    key = tuple(sorted([type_by[a], type_by[b]]))
                    conn[key]["cooccur"] += 1
                    if tuple(sorted([a, b])) in edge_set:
                        conn[key]["connected"] += 1

    rows = []
    rows.append(("total lines", total, EXPECTED["total"], total == EXPECTED["total"]))
    for k in ("accept", "review", "reject"):
        rows.append((f"decisions.{k}", decisions.get(k, 0),
                     EXPECTED["decisions"][k], decisions.get(k, 0) == EXPECTED["decisions"][k]))
    for k in ("en", "de"):
        rows.append((f"accepted.{k}", lang_acc.get(k, 0),
                     EXPECTED["languages_accepted"][k],
                     lang_acc.get(k, 0) == EXPECTED["languages_accepted"][k]))
    rows.append(("participants", len(participants), EXPECTED["participants"],
                 len(participants) == EXPECTED["participants"]))

    print("=== dataset.jsonl checks ===")
    print(f"{'field':<24} {'measured':>10} {'expected':>10} verdict")
    for name, m, e, ok in rows:
        print(f"{name:<24} {m:>10} {e:>10} {'PASS' if ok else 'FAIL'}")

    print()
    print("=== connectivity per node-type pair (accepted only) ===")
    print(f"{'pair':<36} {'cooc':>6} {'conn':>6} {'p':>7} verdict")
    ok_all = True
    for key in sorted(conn):
        c = conn[key]
        if c["cooccur"] < 30:
            continue
        p_ = c["connected"] / c["cooccur"]
        ok = LOW <= p_ <= HIGH
        ok_all = ok_all and ok
        print(f"{'|'.join(key):<36} {c['cooccur']:>6} {c['connected']:>6} "
              f"{p_:>7.3f} {'OK' if ok else 'DEGEN'}")

    hard_pass = all(r[3] for r in rows)
    print()
    print(f"overall: {'PASS' if hard_pass else 'FAIL'} on counts, "
          f"connectivity: {'all-in-band' if ok_all else 'some pairs outside [0.15,0.85]'}")
    return 0 if hard_pass else 1


if __name__ == "__main__":
    sys.exit(main())
