#!/usr/bin/env python3
"""Rebuild the edge_prediction train/dev/test splits locally from dataset.jsonl
and splits.json. The edge splits are not shipped because they duplicate the
same text across every candidate pair from an entry (>10 GB on disk for
~600 MB of source text); this script reproduces them deterministically.

Usage:
    python build_edge_split.py --data-dir . [--negatives hard|random]
        [--neg-ratio 1.0] [--seed 42]

Writes:
    <data-dir>/edge_prediction/{train,dev,test}.jsonl
    <data-dir>/edge_prediction/negative_match.json  (hard mode only)

Pure stdlib, deterministic given the same seed. Matches the exact procedure of
modeling/prepare_data.py in the code repo.
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def record_language(r):
    return r.get("language") or r["participant"].get("language") or "en"


def node_payload(n):
    return {"node_id": n["node_id"], "concept_id": n["concept_id"], "label": n["label"],
            "type": n["type"], "polarity": n.get("polarity"), "time_anchor": n.get("time_anchor")}


def graph_pair_inventory(r):
    nodes = r["graph"]["nodes"]
    edges = r["graph"]["edges"]
    if len(nodes) < 2:
        return None
    node_by_id = {n["node_id"]: n for n in nodes}
    positive = set()
    for e in edges:
        a, b = e["source_node_id"], e["target_node_id"]
        if a == b or a not in node_by_id or b not in node_by_id:
            continue
        positive.add(tuple(sorted([a, b])))
    all_pairs = {tuple(sorted(p)) for p in combinations(sorted(node_by_id), 2)}
    return node_by_id, sorted(positive), sorted(all_pairs - positive)


def type_pair_key(node_by_id, a, b):
    return "__".join(sorted([node_by_id[a]["type"], node_by_id[b]["type"]]))


def build_edge_record(r, node_by_id, a, b, connected):
    return {
        "example_id": f"{r['sample_id']}__{a}__{b}",
        "participant_id": r["participant"]["participant_id"],
        "day_index": r["day_index"],
        "language": record_language(r),
        "text": r["entry"]["text"],
        "node_a": node_payload(node_by_id[a]),
        "node_b": node_payload(node_by_id[b]),
        "connected": connected,
        "graph_num_nodes": len(node_by_id),
        "graph_num_edges": len(r["graph"]["edges"]),
    }


def make_hard(records, neg_ratio, seed, split_name):
    inv = []
    for r in records:
        got = graph_pair_inventory(r)
        if got is None:
            continue
        inv.append((r, *got))

    pos_tp = Counter()
    total_pos = 0
    pool = defaultdict(list)
    for idx, (r, node_by_id, positives, candidates) in enumerate(inv):
        for a, b in positives:
            pos_tp[type_pair_key(node_by_id, a, b)] += 1
            total_pos += 1
        for a, b in candidates:
            pool[type_pair_key(node_by_id, a, b)].append((idx, a, b))

    total_neg_target = int(round(total_pos * neg_ratio))
    raw = {tp: total_neg_target * c / total_pos for tp, c in pos_tp.items()}
    targets = {tp: int(raw[tp]) for tp in raw}
    remainder = total_neg_target - sum(targets.values())
    for tp in sorted(raw, key=lambda t: (-(raw[t] - targets[t]), t))[:remainder]:
        targets[tp] += 1

    rng = random.Random(f"{seed}:{split_name}:hard")
    for tp in sorted(pool):
        pool[tp].sort(key=lambda t: (inv[t[0]][0]["sample_id"], t[1], t[2]))
        rng.shuffle(pool[tp])

    chosen = []
    deficits = {}
    for tp in sorted(targets):
        avail = pool.get(tp, [])
        k = min(targets[tp], len(avail))
        chosen.extend(avail[:k])
        pool[tp] = avail[k:]
        if k < targets[tp]:
            deficits[tp] = targets[tp] - k

    fallback = Counter()
    for tp in sorted(deficits):
        need = deficits[tp]
        tp_types = set(tp.split("__"))
        near = sorted((q for q in pool if pool[q] and set(q.split("__")) & tp_types),
                      key=lambda q: (-len(pool[q]), q))
        far = sorted((q for q in pool if pool[q] and not (set(q.split("__")) & tp_types)),
                     key=lambda q: (-len(pool[q]), q))
        for q in near + far:
            while need > 0 and pool[q]:
                chosen.append(pool[q].pop(0))
                fallback[q] += 1
                need -= 1
            if need == 0:
                break

    achieved = Counter()
    for idx, a, b in chosen:
        achieved[type_pair_key(inv[idx][1], a, b)] += 1

    out = []
    for r, node_by_id, positives, _ in inv:
        for a, b in positives:
            out.append(build_edge_record(r, node_by_id, a, b, True))
    for idx, a, b in chosen:
        r, node_by_id = inv[idx][0], inv[idx][1]
        out.append(build_edge_record(r, node_by_id, a, b, False))
    out.sort(key=lambda x: x["example_id"])

    n_neg = sum(achieved.values())
    tv = 0.5 * sum(abs(pos_tp[tp] / total_pos - achieved.get(tp, 0) / n_neg)
                   for tp in set(pos_tp) | set(achieved)) if n_neg else 1.0
    match = {"split": split_name, "total_positives": total_pos, "total_negatives": n_neg,
             "total_variation_distance": round(tv, 4),
             "deficit_pairs": {tp: deficits[tp] for tp in sorted(deficits)},
             "fallback_taken": {tp: fallback[tp] for tp in sorted(fallback)}}
    return out, match


def make_random(records, neg_ratio, seed):
    out = []
    for r in records:
        got = graph_pair_inventory(r)
        if got is None:
            continue
        node_by_id, positives, candidates = got
        n_neg = min(len(candidates), int(round(len(positives) * neg_ratio)))
        rng = random.Random(f"{seed}:{r['sample_id']}")
        cands = list(candidates)
        rng.shuffle(cands)
        sampled = sorted(cands[:n_neg])
        for a, b in positives:
            out.append(build_edge_record(r, node_by_id, a, b, True))
        for a, b in sampled:
            out.append(build_edge_record(r, node_by_id, a, b, False))
    out.sort(key=lambda x: x["example_id"])
    return out, None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=".", help="dir containing dataset.jsonl + splits.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--negatives", choices=("hard", "random"), default="hard")
    p.add_argument("--neg-ratio", type=float, default=1.0)
    args = p.parse_args()

    root = Path(args.data_dir)
    ds_path = root / "dataset.jsonl"
    splits_path = root / "splits.json"
    if not ds_path.exists() or not splits_path.exists():
        print(f"error: need dataset.jsonl and splits.json in {root}", file=sys.stderr)
        return 2

    records = read_jsonl(ds_path)
    splits = json.loads(splits_path.read_text())
    participant_to_split = {p: s for s, pids in splits["participants"].items() for p in pids}

    include_review = splits.get("include_review", False)
    allowed = {"accept"} | ({"review"} if include_review else set())
    kept = [r for r in records if r.get("decision") in allowed]
    kept.sort(key=lambda r: (r["participant"]["participant_id"], r["day_index"]))

    by_split = {"train": [], "dev": [], "test": []}
    for r in kept:
        s = participant_to_split.get(r["participant"]["participant_id"])
        if s:
            by_split[s].append(r)

    match_all = {}
    for s in ("train", "dev", "test"):
        if args.negatives == "hard":
            examples, match = make_hard(by_split[s], args.neg_ratio, args.seed, s)
        else:
            examples, match = make_random(by_split[s], args.neg_ratio, args.seed)
        write_jsonl(root / "edge_prediction" / f"{s}.jsonl", examples)
        n_pos = sum(1 for e in examples if e["connected"])
        n_neg = len(examples) - n_pos
        print(f"{s}: {len(examples)} examples (pos={n_pos}, neg={n_neg})")
        if match:
            match_all[s] = match
    if match_all:
        write_json(root / "edge_prediction" / "negative_match.json", match_all)
    return 0


if __name__ == "__main__":
    sys.exit(main())
