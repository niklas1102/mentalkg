#!/usr/bin/env python3
"""Run the two trained checkpoints on one input text, print the extracted graph.

Requires models/ to be populated by scripts/download_models.py first.

Usage:
    python scripts/eval_text.py "I could not sleep and the deadline is tomorrow."
    python scripts/eval_text.py --file entry.txt
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"

MAX_LEN = 256
DEFAULT_RELATION = "linked_to"
FALLBACK_TOP_K = 3


def _suffix(marker, node):
    ta = node.get("time_anchor", {}).get("text", "now")
    return f"[{marker}] {node['label']} ({node['type']}, {node['polarity']}, {ta}) [/{marker}]"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("text", nargs="?", default=None)
    p.add_argument("--file", default=None)
    p.add_argument("--node-dir", default=str(MODELS / "mentalkg-xlmr-node"))
    p.add_argument("--edge-dir", default=str(MODELS / "mentalkg-xlmr-edge"))
    p.add_argument("--pool", default=str(REPO / "data" / "variable_pool.json"))
    args = p.parse_args()

    text = args.text or (Path(args.file).read_text() if args.file else None)
    if not text or not text.strip():
        print("error: give a text argument or --file", file=sys.stderr)
        return 2

    node_dir = Path(args.node_dir)
    edge_dir = Path(args.edge_dir)
    for d in (node_dir, edge_dir):
        if not (d / "config.json").exists():
            print(f"error: {d} missing model files; run scripts/download_models.py first",
                  file=sys.stderr)
            return 2

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    node_meta = json.loads((node_dir / "meta.json").read_text())
    edge_meta = json.loads((edge_dir / "meta.json").read_text())
    labels = json.loads((node_dir / "labels.json").read_text())["index_to_label"]
    pool = json.loads(Path(args.pool).read_text())
    by_id = {c["concept_id"]: c for c in pool["node_concepts"]}
    rel = json.loads((edge_dir / "relation_map.json").read_text())
    relation_map = {tuple(k.split("|", 1)): v for k, v in rel["relation"].items()}
    relation_count = {tuple(k.split("|", 1)): v for k, v in rel["count"].items()}

    node_tok = AutoTokenizer.from_pretrained(str(node_dir))
    node_m = AutoModelForSequenceClassification.from_pretrained(str(node_dir)).eval()
    edge_tok = AutoTokenizer.from_pretrained(str(edge_dir))
    edge_m = AutoModelForSequenceClassification.from_pretrained(str(edge_dir)).eval()

    with torch.no_grad():
        enc = node_tok(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        probs = torch.sigmoid(node_m(**enc).logits[0])
    pairs = [(labels[i], float(probs[i])) for i in range(len(labels))]
    above = sorted([q for q in pairs if q[1] >= node_meta["threshold"]], key=lambda q: -q[1])
    if not above:
        above = sorted(pairs, key=lambda q: -q[1])[:FALLBACK_TOP_K]

    nodes = []
    for i, (cid, score) in enumerate(above):
        c = by_id.get(cid, {"type": "unknown", "label": cid, "polarity": "neutral"})
        nodes.append({
            "node_id": f"n{i + 1}", "concept_id": cid,
            "type": c["type"], "label": c["label"],
            "polarity": c.get("polarity", "neutral"),
            "time_anchor": {"text": "now", "day_offset": 0},
            "confidence": score,
        })

    edges = []
    if len(nodes) >= 2:
        combos = list(itertools.combinations(range(len(nodes)), 2))
        ids_list = []
        for i, j in combos:
            for (na, nb) in ((nodes[i], nodes[j]), (nodes[j], nodes[i])):
                bos, sep = edge_tok.bos_token_id, edge_tok.sep_token_id
                body = edge_tok(text, add_special_tokens=False)["input_ids"]
                suf = (edge_tok(_suffix("N1", na), add_special_tokens=False)["input_ids"]
                       + [sep]
                       + edge_tok(_suffix("N2", nb), add_special_tokens=False)["input_ids"])
                budget = MAX_LEN - len(suf) - 4
                if len(body) > budget:
                    body = body[:budget]
                ids_list.append([bos] + body + [sep, sep] + suf + [sep])
        pad = edge_tok.pad_token_id
        max_len = max(len(x) for x in ids_list)
        input_ids = torch.full((len(ids_list), max_len), pad, dtype=torch.long)
        mask = torch.zeros((len(ids_list), max_len), dtype=torch.long)
        for k, ids in enumerate(ids_list):
            input_ids[k, :len(ids)] = torch.tensor(ids)
            mask[k, :len(ids)] = 1
        with torch.no_grad():
            logits = edge_m(input_ids=input_ids, attention_mask=mask).logits[:, 0]
            probs = torch.sigmoid(logits).tolist()
        e_idx = 0
        for k, (i, j) in enumerate(combos):
            p_avg = (probs[2 * k] + probs[2 * k + 1]) / 2.0
            if p_avg < edge_meta["threshold"]:
                continue
            a, b = nodes[i], nodes[j]
            fwd = relation_count.get((a["type"], b["type"]), 0)
            rev = relation_count.get((b["type"], a["type"]), 0)
            if rev > fwd and rev > 0:
                src, dst, rt = b, a, relation_map[(b["type"], a["type"])]
            elif fwd > 0:
                src, dst, rt = a, b, relation_map[(a["type"], b["type"])]
            else:
                src, dst, rt = a, b, DEFAULT_RELATION
            e_idx += 1
            edges.append({
                "edge_id": f"e{e_idx}",
                "source_node_id": src["node_id"], "target_node_id": dst["node_id"],
                "type": rt, "weight": p_avg, "source": "heuristic_type",
            })

    print(json.dumps({
        "text": text,
        "nodes": nodes,
        "edges": edges,
        "node_threshold": node_meta["threshold"],
        "edge_threshold": edge_meta["threshold"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
