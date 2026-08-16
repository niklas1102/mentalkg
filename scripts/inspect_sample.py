#!/usr/bin/env python3
"""Pretty-print one participant-day: graph, generated entry, and verification.

Usage:
    python scripts/inspect_sample.py                            # picks p00001 day 1
    python scripts/inspect_sample.py --participant p00007 --day 5
    python scripts/inspect_sample.py --index 42                 # 0-based line in dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO / "pipeline" / "output" / "dataset.jsonl"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=str(DEFAULT_DATASET))
    p.add_argument("--participant", default="p00001")
    p.add_argument("--day", type=int, default=1)
    p.add_argument("--index", type=int, default=None,
                   help="0-based dataset.jsonl line; overrides participant/day")
    args = p.parse_args()

    path = Path(args.dataset)
    if not path.exists():
        print(f"error: {path} not found", file=sys.stderr)
        return 2

    hit = None
    with path.open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if args.index is not None:
                if i == args.index:
                    hit = r
                    break
            elif (r["participant"]["participant_id"] == args.participant
                  and r.get("day_index") == args.day):
                hit = r
                break

    if hit is None:
        print("no matching sample found", file=sys.stderr)
        return 1

    print(f"=== participant {hit['participant']['participant_id']} "
          f"day {hit.get('day_index')} ({hit.get('language')}) ===")
    print(f"scenario: {hit.get('scenario')}   decision: {hit.get('decision')}")
    print(f"mood_score: {hit.get('mood_score')}")
    print()
    print("--- graph nodes ---")
    for n in hit["graph"]["nodes"]:
        ta = n.get("time_anchor", {}).get("text", "?")
        print(f"  [{n['node_id']}] {n['label']} ({n['type']}, {n['polarity']}, {ta})")
    print()
    print("--- graph edges ---")
    for e in hit["graph"]["edges"]:
        print(f"  {e['source_node_id']} --{e['type']}--> {e['target_node_id']}")
    print()
    print("--- generated entry ---")
    print(hit["entry"]["text"])
    print()
    print("--- verification ---")
    v = hit.get("verification") or {}
    print(f"overall: {v.get('overall')}  reason: {v.get('reason')}")
    print(f"nodes mentioned: {sum(1 for x in v.get('nodes') or [] if x.get('mentioned'))}"
          f"/{len(v.get('nodes') or [])}")
    print(f"edges expressed: {sum(1 for x in v.get('edges') or [] if x.get('expressed'))}"
          f"/{len(v.get('edges') or [])}")
    extras = v.get("extra_content") or []
    if extras:
        print(f"extra content ({len(extras)}):")
        for x in extras[:5]:
            print(f"  - {x.get('content')}: \"{x.get('evidence')}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
