#!/usr/bin/env python3
"""Prepare node- and edge-prediction datasets from the synthetic pipeline output (v2).

Reads pipeline/output/dataset.jsonl, applies a participant-independent 50/30/20
split STRATIFIED BY LANGUAGE, and writes JSONL files plus a stats report under
modeling/data_v2/. Pure stdlib; deterministic given the same seed.

v2 changes over v1:
- split stratified by participant language (each split keeps ~the same en/de mix)
- every node/edge example records its language
- edge negatives: --negatives hard (default) samples within-graph negatives so
  their TYPE-PAIR distribution matches the positives' type-pair distribution
  (removes the type-pair shortcut); --negatives random keeps the v1 behaviour
- labels.json additionally lists the common-concept filter (train support >= 20)
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
PIPELINE_DEFAULT = REPO_ROOT / "pipeline" / "output" / "dataset.jsonl"
OUTPUT_DEFAULT = THIS_DIR / "data_v2"

COMMON_LABEL_MIN_TRAIN = 20  # same threshold the baselines have always used


# ----- I/O ------------------------------------------------------------------

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


def record_language(record):
    return record.get("language") or record["participant"].get("language") or "en"


# ----- Split (stratified by language) ----------------------------------------

def split_participants_stratified(participant_lang, train_ratio, dev_ratio, test_ratio, seed):
    """participant_lang: {participant_id: language}. Splits each language group
    50/30/20 independently with the same procedure, then merges — so every split
    has ~the language ratio of the whole corpus. Deterministic."""
    total = train_ratio + dev_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {total}")
    splits = {"train": [], "dev": [], "test": []}
    by_lang = defaultdict(list)
    for pid, lang in participant_lang.items():
        by_lang[lang].append(pid)
    for lang in sorted(by_lang):
        ordered = sorted(by_lang[lang])
        rng = random.Random(f"{seed}:{lang}")
        rng.shuffle(ordered)
        n = len(ordered)
        n_train = int(round(n * train_ratio))
        n_dev = int(round(n * dev_ratio))
        splits["train"].extend(ordered[:n_train])
        splits["dev"].extend(ordered[n_train:n_train + n_dev])
        splits["test"].extend(ordered[n_train + n_dev:])
    splits = {k: sorted(v) for k, v in splits.items()}
    assert set(splits["train"]).isdisjoint(splits["dev"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["dev"]).isdisjoint(splits["test"])
    assert set().union(*splits.values()) == set(participant_lang)
    return splits


# ----- Node-prediction dataset ---------------------------------------------

def make_node_example(record):
    nodes = record["graph"]["nodes"]
    concepts = sorted({n["concept_id"] for n in nodes})
    surfaces = sorted({n["label"] for n in nodes})
    return {
        "example_id": record["sample_id"],
        "participant_id": record["participant"]["participant_id"],
        "day_index": record["day_index"],
        "language": record_language(record),
        "text": record["entry"]["text"],
        "labels": concepts,
        "label_surface": surfaces,
        "num_nodes": len(nodes),
    }


def build_label_index(train_examples):
    counts = Counter()
    for ex in train_examples:
        for c in ex["labels"]:
            counts[c] += 1
    sorted_labels = sorted(counts.keys())
    common = [l for l in sorted_labels if counts[l] >= COMMON_LABEL_MIN_TRAIN]
    return {
        "label_to_index": {label: i for i, label in enumerate(sorted_labels)},
        "index_to_label": sorted_labels,
        "num_labels": len(sorted_labels),
        "train_support": {label: counts[label] for label in sorted_labels},
        "common_label_min_train": COMMON_LABEL_MIN_TRAIN,
        "common_labels": common,
        "num_common_labels": len(common),
    }


def annotate_unseen(examples, known_labels):
    for ex in examples:
        ex["unseen_labels"] = sorted(set(ex["labels"]) - known_labels)
    return examples


# ----- Edge-prediction dataset ---------------------------------------------

def node_payload(node):
    return {
        "node_id": node["node_id"],
        "concept_id": node["concept_id"],
        "label": node["label"],
        "type": node["type"],
        "polarity": node.get("polarity"),
        "time_anchor": node.get("time_anchor"),
    }


def graph_pair_inventory(record):
    """Returns (node_by_id, positive_pairs_sorted, negative_candidates_sorted)
    for one record, or None if the graph is too small."""
    nodes = record["graph"]["nodes"]
    edges = record["graph"]["edges"]
    if len(nodes) < 2:
        return None
    node_by_id = {n["node_id"]: n for n in nodes}
    positive_pairs = set()
    for e in edges:
        a, b = e["source_node_id"], e["target_node_id"]
        if a == b or a not in node_by_id or b not in node_by_id:
            continue
        positive_pairs.add(tuple(sorted([a, b])))
    all_pairs = {tuple(sorted(p)) for p in combinations(sorted(node_by_id), 2)}
    candidate_negatives = sorted(all_pairs - positive_pairs)
    return node_by_id, sorted(positive_pairs), candidate_negatives


def type_pair_key(node_by_id, a, b):
    return "__".join(sorted([node_by_id[a]["type"], node_by_id[b]["type"]]))


def _build_edge_record(record, node_by_id, a, b, connected):
    return {
        "example_id": f"{record['sample_id']}__{a}__{b}",
        "participant_id": record["participant"]["participant_id"],
        "day_index": record["day_index"],
        "language": record_language(record),
        "text": record["entry"]["text"],
        "node_a": node_payload(node_by_id[a]),
        "node_b": node_payload(node_by_id[b]),
        "connected": connected,
        "graph_num_nodes": len(node_by_id),
        "graph_num_edges": len(record["graph"]["edges"]),
    }


def make_edge_examples_random(records, neg_ratio, seed):
    """v1 behaviour: per-graph uniform-random negatives, neg_ratio per graph."""
    out = []
    for record in records:
        inv = graph_pair_inventory(record)
        if inv is None:
            continue
        node_by_id, positives, candidates = inv
        target_n_neg = int(round(len(positives) * neg_ratio))
        n_neg = min(len(candidates), target_n_neg)
        rng = random.Random(f"{seed}:{record['sample_id']}")
        cands = list(candidates)
        rng.shuffle(cands)
        sampled = sorted(cands[:n_neg])
        for a, b in positives:
            out.append(_build_edge_record(record, node_by_id, a, b, True))
        for a, b in sampled:
            out.append(_build_edge_record(record, node_by_id, a, b, False))
    out.sort(key=lambda r: r["example_id"])
    return out, None


def make_edge_examples_hard(records, neg_ratio, seed, split_name):
    """v2 hard negatives: sample within-graph negative pairs so the negatives'
    TYPE-PAIR distribution matches the positives' type-pair distribution for
    this split as closely as the candidate pool allows. Where a type pair has
    too few candidates, the deficit is filled from the nearest available pairs
    (sharing >= 1 node type first, then any), and the achieved match is
    reported."""
    inventory = []  # (record, node_by_id, positives, candidates)
    for record in records:
        inv = graph_pair_inventory(record)
        if inv is None:
            continue
        inventory.append((record, *inv))

    pos_tp = Counter()
    total_pos = 0
    pool = defaultdict(list)  # type_pair -> [(inv_index, a, b)]
    for idx, (record, node_by_id, positives, candidates) in enumerate(inventory):
        for a, b in positives:
            pos_tp[type_pair_key(node_by_id, a, b)] += 1
            total_pos += 1
        for a, b in candidates:
            pool[type_pair_key(node_by_id, a, b)].append((idx, a, b))

    total_neg_target = int(round(total_pos * neg_ratio))

    # proportional per-type-pair targets, largest-remainder rounding to hit total
    raw = {tp: total_neg_target * c / total_pos for tp, c in pos_tp.items()}
    targets = {tp: int(raw[tp]) for tp in raw}
    remainder = total_neg_target - sum(targets.values())
    for tp in sorted(raw, key=lambda t: (-(raw[t] - targets[t]), t))[:remainder]:
        targets[tp] += 1

    rng = random.Random(f"{seed}:{split_name}:hard")
    for tp in sorted(pool):
        pool[tp].sort(key=lambda t: (inventory[t[0]][0]["sample_id"], t[1], t[2]))
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

    # fallback: nearest available type pairs (share a node type), then any
    fallback_taken = Counter()
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
                fallback_taken[q] += 1
                need -= 1
            if need == 0:
                break

    achieved_tp = Counter()
    for idx, a, b in chosen:
        achieved_tp[type_pair_key(inventory[idx][1], a, b)] += 1

    out = []
    for idx, (record, node_by_id, positives, _candidates) in enumerate(inventory):
        for a, b in positives:
            out.append(_build_edge_record(record, node_by_id, a, b, True))
    for idx, a, b in chosen:
        record, node_by_id = inventory[idx][0], inventory[idx][1]
        out.append(_build_edge_record(record, node_by_id, a, b, False))
    out.sort(key=lambda r: r["example_id"])

    # match report: positive vs achieved-negative type-pair distributions
    n_neg = sum(achieved_tp.values())
    tv = 0.5 * sum(abs(pos_tp[tp] / total_pos - achieved_tp.get(tp, 0) / n_neg)
                   for tp in set(pos_tp) | set(achieved_tp)) if n_neg else 1.0
    match = {
        "split": split_name,
        "total_positives": total_pos,
        "total_negatives": n_neg,
        "total_variation_distance": round(tv, 4),
        "deficit_pairs": {tp: deficits[tp] for tp in sorted(deficits)},
        "fallback_taken": {tp: fallback_taken[tp] for tp in sorted(fallback_taken)},
        "per_type_pair": {
            tp: {
                "pos": pos_tp.get(tp, 0),
                "pos_frac": round(pos_tp.get(tp, 0) / total_pos, 4) if total_pos else 0,
                "neg_target": targets.get(tp, 0),
                "neg_achieved": achieved_tp.get(tp, 0),
                "neg_frac": round(achieved_tp.get(tp, 0) / n_neg, 4) if n_neg else 0,
            }
            for tp in sorted(set(pos_tp) | set(achieved_tp))
        },
    }
    return out, match


# ----- Stats / report ------------------------------------------------------

def percentiles(values):
    if not values:
        return {k: 0 for k in ("min", "p25", "p50", "p75", "max")}
    v = sorted(values)
    def q(p):
        i = int(round(p * (len(v) - 1)))
        return v[i]
    return {"min": v[0], "p25": q(0.25), "p50": q(0.50), "p75": q(0.75), "max": v[-1]}


def write_report(report_path, args, decision_counts, splits, participant_lang,
                 node_by_split, edge_by_split, label_index, edge_match_by_split):
    lines = []
    lines.append("# Modeling Data v2 — Preparation Report")
    lines.append("")
    lines.append("Auto-generated by `modeling/prepare_data.py` (v2). Inputs and choices:")
    lines.append("")
    lines.append(f"- Input: `{args.input}`")
    lines.append(f"- Output: `{args.output}`")
    lines.append(f"- Seed: {args.seed}")
    lines.append(f"- Include review: {args.include_review}")
    lines.append(f"- Edge negatives: **{args.negatives}** (neg_ratio {args.neg_ratio})")
    lines.append(f"- Split (train/dev/test): {args.train}/{args.dev}/{args.test}, "
                 f"participant-independent, stratified by language")
    lines.append("")

    lines.append("## Filtering")
    lines.append("")
    total = sum(decision_counts.values())
    accept = decision_counts.get("accept", 0)
    lines.append(f"- Source records: {total}")
    lines.append(f"  - accept: {accept} ({accept / total:.1%})" if total else f"  - accept: {accept}")
    lines.append(f"  - review: {decision_counts.get('review', 0)}")
    lines.append(f"  - reject: {decision_counts.get('reject', 0)}")
    kept = accept + (decision_counts.get("review", 0) if args.include_review else 0)
    lines.append(f"- Records kept after filter: {kept}")
    lines.append("")

    lines.append("## Splits (participant-independent, language-stratified)")
    lines.append("")
    lines.append("| split | participants | en | de | node examples | en ex | de ex | en % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for split in ("train", "dev", "test"):
        pids = splits[split]
        lang_c = Counter(participant_lang[p] for p in pids)
        exs = node_by_split[split]
        ex_lang = Counter(e["language"] for e in exs)
        n = len(exs)
        lines.append(f"| {split} | {len(pids)} | {lang_c.get('en', 0)} | {lang_c.get('de', 0)} "
                     f"| {n} | {ex_lang.get('en', 0)} | {ex_lang.get('de', 0)} "
                     f"| {ex_lang.get('en', 0) / n:.1%} |")
    lines.append("")

    lines.append("## Node prediction")
    lines.append("")
    lines.append(f"- Label vocabulary size (train-derived): {label_index['num_labels']}")
    lines.append(f"- Common concepts (train support >= {label_index['common_label_min_train']}): "
                 f"{label_index['num_common_labels']}")
    for split in ("train", "dev", "test"):
        exs = node_by_split[split]
        n = len(exs)
        avg_lbls = sum(len(e["labels"]) for e in exs) / n if n else 0
        unseen = sum(1 for e in exs if e.get("unseen_labels"))
        wcs = [len(e["text"].split()) for e in exs]
        pct = percentiles(wcs)
        lines.append(f"- **{split}**: {n} examples, avg labels/example: {avg_lbls:.2f}, "
                     f"examples with unseen-in-train labels: {unseen}")
        lines.append(f"  - text word count: min {pct['min']} / p25 {pct['p25']} / "
                     f"p50 {pct['p50']} / p75 {pct['p75']} / max {pct['max']}")
    lines.append("")
    support = label_index["train_support"]
    bottom = sorted(support.items(), key=lambda kv: (kv[1], kv[0]))[:15]
    lines.append("### Rarest labels in train")
    lines.append("")
    for label, c in bottom:
        lines.append(f"- `{label}`: {c}")
    lines.append("")

    lines.append("## Edge prediction")
    lines.append("")
    for split in ("train", "dev", "test"):
        exs = edge_by_split[split]
        n = len(exs)
        pos = sum(1 for e in exs if e["connected"])
        lang_c = Counter(e["language"] for e in exs)
        lines.append(f"- **{split}**: {n} examples ({pos} positive / {n - pos} negative; "
                     f"en {lang_c.get('en', 0)} / de {lang_c.get('de', 0)})")
    lines.append("")

    if args.negatives == "hard":
        lines.append("### Hard-negative type-pair match (positives vs sampled negatives)")
        lines.append("")
        for split in ("train", "dev", "test"):
            m = edge_match_by_split.get(split)
            if not m:
                continue
            lines.append(f"**{split}** — total variation distance between the positive and "
                         f"negative type-pair distributions: **{m['total_variation_distance']:.4f}** "
                         f"(0 = exact match)")
            if m["deficit_pairs"]:
                lines.append(f"- deficit pairs (not enough candidates, filled from nearest): "
                             + ", ".join(f"`{tp}` short {v}" for tp, v in m["deficit_pairs"].items()))
            else:
                lines.append("- no deficits: every type pair had enough negative candidates")
            lines.append("")
            lines.append("| type pair | pos n | pos % | neg n | neg % |")
            lines.append("|---|---:|---:|---:|---:|")
            per = m["per_type_pair"]
            for tp in sorted(per, key=lambda t: -per[t]["pos"]):
                r = per[tp]
                lines.append(f"| {tp} | {r['pos']} | {r['pos_frac']:.1%} "
                             f"| {r['neg_achieved']} | {r['neg_frac']:.1%} |")
            lines.append("")
    else:
        lines.append("Negatives sampled uniformly at random within each graph (v1 behaviour).")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


# ----- Main ----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(PIPELINE_DEFAULT))
    parser.add_argument("--output", default=str(OUTPUT_DEFAULT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-review", action="store_true")
    parser.add_argument("--neg-ratio", type=float, default=1.0)
    parser.add_argument("--negatives", choices=("hard", "random"), default="hard",
                        help="edge negative sampling: 'hard' matches the positives' "
                             "type-pair distribution (default), 'random' is v1 behaviour")
    parser.add_argument("--train", type=float, default=0.5)
    parser.add_argument("--dev", type=float, default=0.3)
    parser.add_argument("--test", type=float, default=0.2)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    if not input_path.exists():
        print(f"error: input not found: {input_path}", file=sys.stderr)
        return 2

    records = read_jsonl(input_path)
    decision_counts = Counter(r.get("decision") for r in records)
    allowed = {"accept"} | ({"review"} if args.include_review else set())
    kept = [r for r in records if r.get("decision") in allowed]
    kept.sort(key=lambda r: (r["participant"]["participant_id"], r["day_index"]))
    if not kept:
        print("error: no records left after filtering", file=sys.stderr)
        return 2

    participant_lang = {}
    for r in kept:
        participant_lang[r["participant"]["participant_id"]] = record_language(r)

    splits = split_participants_stratified(
        participant_lang, args.train, args.dev, args.test, args.seed
    )

    participant_to_split = {p: s for s, pids in splits.items() for p in pids}
    records_by_split = {"train": [], "dev": [], "test": []}
    for r in kept:
        records_by_split[participant_to_split[r["participant"]["participant_id"]]].append(r)

    # Node prediction
    node_by_split = {
        s: [make_node_example(r) for r in records_by_split[s]]
        for s in ("train", "dev", "test")
    }
    label_index = build_label_index(node_by_split["train"])
    known = set(label_index["index_to_label"])
    for s in ("train", "dev", "test"):
        annotate_unseen(node_by_split[s], known)

    # Edge prediction
    edge_by_split = {}
    edge_match_by_split = {}
    for s in ("train", "dev", "test"):
        if args.negatives == "hard":
            examples, match = make_edge_examples_hard(
                records_by_split[s], args.neg_ratio, args.seed, s)
        else:
            examples, match = make_edge_examples_random(
                records_by_split[s], args.neg_ratio, args.seed)
        edge_by_split[s] = examples
        if match:
            edge_match_by_split[s] = match

    # Write outputs
    splits_meta = {
        "seed": args.seed,
        "ratio": {"train": args.train, "dev": args.dev, "test": args.test},
        "counts": {s: len(splits[s]) for s in ("train", "dev", "test")},
        "language_counts": {
            s: dict(Counter(participant_lang[p] for p in splits[s]))
            for s in ("train", "dev", "test")
        },
        "include_review": args.include_review,
        "negatives": args.negatives,
        "stratified_by_language": True,
        "participants": splits,
    }
    write_json(output_dir / "splits.json", splits_meta)

    for s in ("train", "dev", "test"):
        write_jsonl(output_dir / "node_prediction" / f"{s}.jsonl", node_by_split[s])
        write_jsonl(output_dir / "edge_prediction" / f"{s}.jsonl", edge_by_split[s])
    write_json(output_dir / "node_prediction" / "labels.json", label_index)
    if edge_match_by_split:
        write_json(output_dir / "edge_prediction" / "negative_match.json", edge_match_by_split)

    write_report(output_dir / "report.md", args, decision_counts, splits,
                 participant_lang, node_by_split, edge_by_split, label_index,
                 edge_match_by_split)

    print(f"Wrote outputs to {output_dir}  (negatives={args.negatives})")
    for s in ("train", "dev", "test"):
        n_node = len(node_by_split[s])
        n_edge = len(edge_by_split[s])
        n_pos = sum(1 for e in edge_by_split[s] if e["connected"])
        lang_c = Counter(e["language"] for e in node_by_split[s])
        tv = (f", neg-match TV={edge_match_by_split[s]['total_variation_distance']:.4f}"
              if s in edge_match_by_split else "")
        print(f"  {s}: node={n_node} (en {lang_c.get('en', 0)}/de {lang_c.get('de', 0)}), "
              f"edge={n_edge} (pos={n_pos}, neg={n_edge - n_pos}){tv}")
    print(f"  node label vocab (train): {label_index['num_labels']} "
          f"(common >= {COMMON_LABEL_MIN_TRAIN}: {label_index['num_common_labels']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
