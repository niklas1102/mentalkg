"""Surface-string-match floor baseline for node prediction (v2).

Predicts concept c iff the surface label of c (from Variable
Pool/variable_pool.json) appears as a lowercase substring of the lowercased
entry text. No training, no threshold — a floor that shows how far simple
lexical matching gets. The label set is the 130-id vocabulary from
data_v2/node_prediction/labels.json (the pool's 3 extra concepts never occur
in any split and could only add false positives). Pool labels are English,
so the German floor is expected to be near zero (loanword hits only).

Usage:
    python modeling/baselines/string_match_floor.py [--split test]
        [--data-dir ...] [--art-dir ...]

Artifacts default to modeling/baselines/artifacts_v2/string_match/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DEFAULT = REPO_ROOT / "modeling" / "data_v2" / "node_prediction"
POOL_DEFAULT = REPO_ROOT / "data" / "variable_pool.json"
ART_DEFAULT = REPO_ROOT / "modeling" / "baselines" / "artifacts_v2" / "string_match"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "n_examples": int(y_true.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test")
    parser.add_argument("--data-dir", default=str(DATA_DEFAULT))
    parser.add_argument("--pool", default=str(POOL_DEFAULT))
    parser.add_argument("--art-dir", default=str(ART_DEFAULT))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    art_dir = Path(args.art_dir)
    art_dir.mkdir(parents=True, exist_ok=True)
    tag = "string_match_floor"

    records = read_jsonl(data_dir / f"{args.split}.jsonl")
    label_index = json.loads((data_dir / "labels.json").read_text())
    vocab = label_index["index_to_label"]
    pool = json.loads(Path(args.pool).read_text())
    surface = {c["concept_id"]: c["label"].lower()
               for c in pool["node_concepts"]}
    missing = [c for c in vocab if c not in surface]
    if missing:
        raise SystemExit(f"concept ids without pool surface label: {missing}")
    print(f"[{tag}] {len(records)} examples ({args.split}), "
          f"{len(vocab)} concept ids", flush=True)

    idx = {c: i for i, c in enumerate(vocab)}
    Y = np.zeros((len(records), len(vocab)), dtype=int)
    P = np.zeros_like(Y)
    for i, r in enumerate(records):
        text = r["text"].lower()
        for c in r["labels"]:
            Y[i, idx[c]] = 1
        for c in vocab:
            if surface[c] in text:
                P[i, idx[c]] = 1

    overall = evaluate(Y, P)
    print(f"[{tag}] OVERALL:", json.dumps(overall), flush=True)
    per_language = {}
    for lg in sorted({r.get("language", "?") for r in records}):
        sel = np.array(
            [i for i, r in enumerate(records) if r.get("language") == lg],
            dtype=int,
        )
        per_language[lg] = evaluate(Y[sel], P[sel])
        print(f"[{tag}] [{lg}]:", json.dumps(per_language[lg]), flush=True)

    metrics = {
        "task": "node_prediction",
        "model": "string_match_floor",
        "split": args.split,
        "label_vocab_size": len(vocab),
        "pool_concepts_unused": sorted(set(surface) - set(vocab)),
        "test": overall,
        "test_per_language": per_language,
    }
    (art_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    lines = [
        "# String-match floor baseline (node prediction, v2)",
        "",
        "Predict concept c iff its English surface label from "
        "`data/variable_pool.json` appears lowercase-substring in "
        f"the lowercased entry text. Split: `{args.split}`, "
        f"{len(records)} examples, {len(vocab)} concept ids "
        "(the pool's 3 extra concepts never occur in any split and are "
        "excluded: " + ", ".join(metrics["pool_concepts_unused"]) + ").",
        "",
        "| slice | micro-F1 | macro-F1 | micro-P | micro-R | n |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in [("overall", overall)] + sorted(per_language.items()):
        lines.append(
            f"| {name} | {m['micro_f1']:.4f} | {m['macro_f1']:.4f} | "
            f"{m['micro_precision']:.4f} | {m['micro_recall']:.4f} | "
            f"{m['n_examples']} |"
        )
    lines += [
        "",
        "Pool labels are English only, so the German floor is near zero — "
        "the few de hits are loanwords (e.g. Deadline, Stress).",
    ]
    (art_dir / "results.md").write_text("\n".join(lines) + "\n")
    print(f"[{tag}] artifacts written to {art_dir}", flush=True)


if __name__ == "__main__":
    main()
