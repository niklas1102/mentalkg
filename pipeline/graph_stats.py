"""Compute P(connected | node-type pair) from a set of day graphs.

The concern this measures: with too few templates, whether two nodes are joined by
an edge becomes nearly deterministic given their *types* alone (a structural
shortcut a downstream model can exploit without reading the text). For every
common unordered type pair we want P(connected) strictly inside (0.15, 0.85) --
never 0.0 or 1.0.

"Connected" is per node-pair: for each graph we enumerate every unordered pair of
nodes, bucket it by the unordered pair of their types, and count how many such
co-occurring pairs are joined by a direct edge (in either direction).

Usage:
    python3 graph_stats.py output/graphs.jsonl
"""

import json
import sys
from itertools import combinations
from pathlib import Path

LOW, HIGH = 0.15, 0.85
DEFAULT_MIN_COUNT = 30


def _as_graph(item):
    return item["graph"] if isinstance(item, dict) and "graph" in item else item


def connectivity(graphs):
    """graphs: iterable of graph dicts (or sample dicts containing a 'graph' key).
    Returns {(type_a, type_b): {'cooccur', 'connected', 'p'}} with type_a <= type_b."""
    stats = {}
    for item in graphs:
        g = _as_graph(item)
        nodes = g["nodes"]
        type_by_id = {n["node_id"]: n["type"] for n in nodes}
        edge_set = set()
        for e in g["edges"]:
            a, b = e["source_node_id"], e["target_node_id"]
            edge_set.add((a, b))
            edge_set.add((b, a))
        for a, b in combinations(nodes, 2):
            key = tuple(sorted((type_by_id[a["node_id"]], type_by_id[b["node_id"]])))
            rec = stats.setdefault(key, {"cooccur": 0, "connected": 0})
            rec["cooccur"] += 1
            if (a["node_id"], b["node_id"]) in edge_set:
                rec["connected"] += 1
    for rec in stats.values():
        rec["p"] = rec["connected"] / rec["cooccur"] if rec["cooccur"] else 0.0
    return stats


def out_of_bounds(stats, min_count=DEFAULT_MIN_COUNT):
    """Common pairs (co-occurrence >= min_count) whose P is outside (LOW, HIGH)."""
    bad = []
    for key, rec in stats.items():
        if rec["cooccur"] >= min_count and not (LOW < rec["p"] < HIGH):
            bad.append((key, rec))
    return bad


def format_table(stats, min_count=DEFAULT_MIN_COUNT):
    lines = []
    lines.append(f"{'type_a':<14} {'type_b':<14} {'cooccur':>8} {'connected':>10} "
                 f"{'P(conn)':>8}  {'flag':>6}")
    lines.append("-" * 66)
    for key in sorted(stats, key=lambda k: -stats[k]["cooccur"]):
        rec = stats[key]
        common = rec["cooccur"] >= min_count
        in_bounds = LOW < rec["p"] < HIGH
        flag = "" if not common else ("ok" if in_bounds else "OUT")
        lines.append(f"{key[0]:<14} {key[1]:<14} {rec['cooccur']:>8} "
                     f"{rec['connected']:>10} {rec['p']:>8.3f}  {flag:>6}")
    bad = out_of_bounds(stats, min_count)
    lines.append("")
    if bad:
        lines.append(f"OUT OF BOUNDS (common pairs, P not in ({LOW},{HIGH})): {len(bad)}")
        for key, rec in sorted(bad, key=lambda kr: kr[1]["p"]):
            lines.append(f"  {key[0]}--{key[1]}: P={rec['p']:.3f} (n={rec['cooccur']})")
    else:
        lines.append(f"All common pairs (co-occurrence >= {min_count}) have "
                     f"P(connected) strictly in ({LOW}, {HIGH}).  PASS")
    return "\n".join(lines)


def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "output/graphs.jsonl"
    min_count = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_MIN_COUNT
    graphs = read_jsonl(Path(path))
    stats = connectivity(graphs)
    print(f"Graphs analyzed: {len(graphs)}\n")
    print(format_table(stats, min_count))


if __name__ == "__main__":
    main()
