"""Edge shortcut analysis.

The synthetic graphs come from a small set of templates that wire fixed
node-TYPE pairs together; edge negatives are sampled within the same graph.
This script quantifies how much of the edge-baseline performance is explained
by template / structural signal rather than reading the journal text.

Three ablations are run, plus a P(connected | sorted_type_pair) table:

  C1 — Pair features only        (types, type_pair, polarities, concept ids;
                                  no text)
  C2 — Type + polarity only      (no text, no concept identity — pure template
                                  structure)
  C3 — Type-pair conditional     P(connected | sorted(type_a, type_b)) over
       probability table          training edges, sorted by frequency.

Outputs:
  modeling/baselines/artifacts/edge_pair_only/{metrics.json,model.joblib,...}
  modeling/baselines/artifacts/edge_type_pol_only/{...}
  modeling/baselines/artifacts/edge_type_pair_table.csv
  modeling/baselines/artifacts/edge_shortcut_summary.json
  results/RESULTS.md
  results/metrics.json

Also reads:
  modeling/baselines/artifacts/edge/metrics.json     (full-text edge model)
  modeling/baselines/artifacts/node/metrics.json     (node baseline)
  results/llama_zero_shot/{node,edge}_prediction/metrics.json

Usage (v2 — ablations + summary only; RESULTS assembly lives in make_results_v2.py):
    python modeling/baselines/edge_shortcut.py \
        [--data-dir modeling/data_v2/edge_prediction] \
        [--art-dir modeling/baselines/artifacts_v2] \
        [--edge-metrics modeling/baselines/artifacts_v2/edge_hard/metrics.json] \
        [--write-v1-results]   # legacy v1 RESULTS.md path
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SEED = 42
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "modeling" / "data_v2" / "edge_prediction"
ART_DIR = REPO_ROOT / "modeling" / "baselines" / "artifacts_v2"
RESULTS_DIR = REPO_ROOT / "results"
LLAMA_NODE = RESULTS_DIR / "llama_zero_shot" / "node_prediction" / "metrics.json"
LLAMA_EDGE = RESULTS_DIR / "llama_zero_shot" / "edge_prediction" / "metrics.json"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pair_features_full(record: dict) -> dict:
    """Pair features including concept ids."""
    a, b = record["node_a"], record["node_b"]
    ta, tb = a["type"], b["type"]
    pa, pb = a["polarity"], b["polarity"]
    ca, cb = a["concept_id"], b["concept_id"]
    return {
        f"type_a={ta}": 1.0,
        f"type_b={tb}": 1.0,
        f"type_pair={'__'.join(sorted([ta, tb]))}": 1.0,
        f"pol_a={pa}": 1.0,
        f"pol_b={pb}": 1.0,
        f"pol_pair={'__'.join(sorted([pa, pb]))}": 1.0,
        f"cid_a={ca}": 1.0,
        f"cid_b={cb}": 1.0,
        f"cid_pair={'__'.join(sorted([ca, cb]))}": 1.0,
    }


def pair_features_type_pol(record: dict) -> dict:
    """Pair features restricted to types and polarities only."""
    a, b = record["node_a"], record["node_b"]
    ta, tb = a["type"], b["type"]
    pa, pb = a["polarity"], b["polarity"]
    return {
        f"type_a={ta}": 1.0,
        f"type_b={tb}": 1.0,
        f"type_pair={'__'.join(sorted([ta, tb]))}": 1.0,
        f"pol_a={pa}": 1.0,
        f"pol_b={pb}": 1.0,
        f"pol_pair={'__'.join(sorted([pa, pb]))}": 1.0,
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "n_pos_pred": int(y_pred.sum()),
        "n_pos_true": int(y_true.sum()),
        "n_total": int(len(y_true)),
    }


def run_ablation(name: str, feat_fn, train, dev, test, y_train, y_dev, y_test):
    print(f"[shortcut/{name}] fitting...")
    pair_vec = DictVectorizer(sparse=True)
    Xp_train = pair_vec.fit_transform([feat_fn(r) for r in train])
    Xp_dev = pair_vec.transform([feat_fn(r) for r in dev])
    Xp_test = pair_vec.transform([feat_fn(r) for r in test])
    print(f"[shortcut/{name}] feature_dim={Xp_train.shape[1]}")
    t0 = time.time()
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=SEED,
    )
    clf.fit(Xp_train, y_train)
    print(f"[shortcut/{name}] fit done in {time.time() - t0:.1f}s")

    scores_dev = clf.predict_proba(Xp_dev)[:, 1]
    best_t, best_f1 = 0.5, -1.0
    for t in THRESHOLD_GRID:
        f1 = f1_score(y_dev, (scores_dev >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    print(f"[shortcut/{name}] best threshold t*={best_t:.2f} dev_f1={best_f1:.4f}")

    scores_test = clf.predict_proba(Xp_test)[:, 1]
    y_pred = (scores_test >= best_t).astype(int)
    metrics = evaluate(y_test, y_pred, scores_test)
    print(f"[shortcut/{name}] TEST :", json.dumps(metrics))

    out_dir = ART_DIR / f"edge_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pair_vec, out_dir / "pair_vectorizer.joblib")
    joblib.dump(clf, out_dir / "model.joblib")
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "task": "edge_prediction",
                "ablation": name,
                "seed": SEED,
                "threshold": best_t,
                "dev_f1_at_threshold": best_f1,
                "feature_dim": int(Xp_train.shape[1]),
                "train_size": len(train),
                "dev_size": len(dev),
                "test_size": len(test),
            },
            indent=2,
        )
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "task": "edge_prediction",
                "ablation": name,
                "dev": {"f1_at_best_threshold": best_f1, "threshold": best_t},
                "test": metrics,
            },
            indent=2,
        )
    )
    return metrics, best_t


def type_pair_table(train: list[dict]) -> list[dict]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [n, n_connected]
    for r in train:
        tp = "__".join(sorted([r["node_a"]["type"], r["node_b"]["type"]]))
        counts[tp][0] += 1
        counts[tp][1] += int(r["connected"])
    rows = []
    for tp, (n, n_c) in counts.items():
        rows.append({"type_pair": tp, "n": n, "n_connected": n_c, "p_connected": n_c / n})
    rows.sort(key=lambda r: r["n"], reverse=True)
    return rows


def load_llama() -> dict:
    """Read prior zero-shot Llama metrics; tolerate schema differences."""
    out = {"node": None, "edge": None}
    if LLAMA_NODE.exists():
        try:
            out["node"] = json.loads(LLAMA_NODE.read_text())
        except Exception as e:
            print(f"[shortcut] warning: could not parse {LLAMA_NODE}: {e}")
    if LLAMA_EDGE.exists():
        try:
            out["edge"] = json.loads(LLAMA_EDGE.read_text())
        except Exception as e:
            print(f"[shortcut] warning: could not parse {LLAMA_EDGE}: {e}")
    return out


def fmt(x):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def write_results(
    node_metrics: dict,
    edge_full_metrics: dict,
    c1_metrics: dict,
    c2_metrics: dict,
    tp_rows: list[dict],
    llama: dict,
    verdict: str,
) -> None:
    """Write results/RESULTS.md and results/metrics.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    full = node_metrics["test"]["full"]
    common = node_metrics["test"]["common"]
    edge_full = edge_full_metrics["test"]

    # Llama node fields: {"micro": {"f1","precision","recall"}, "macro": {"f1",...}}.
    def _ll_node():
        d = llama["node"]
        if not isinstance(d, dict):
            return None, None, None, None
        micro = d.get("micro") if isinstance(d.get("micro"), dict) else {}
        macro = d.get("macro") if isinstance(d.get("macro"), dict) else {}
        return (
            micro.get("f1"),
            macro.get("f1"),
            micro.get("precision"),
            micro.get("recall"),
        )

    def _ll_edge():
        if not llama["edge"]:
            return None, None, None, None, None
        d = llama["edge"]
        for parent in (d, d.get("test", {}) if isinstance(d, dict) else {}):
            if not isinstance(parent, dict):
                continue
            acc = parent.get("accuracy")
            pr = parent.get("precision")
            rc = parent.get("recall")
            f1 = parent.get("f1")
            auc = parent.get("roc_auc")
            if any(v is not None for v in (acc, f1)):
                return acc, pr, rc, f1, auc
        return None, None, None, None, None

    ll_node_mi, ll_node_ma, ll_node_pr, ll_node_rc = _ll_node()
    ll_edge_acc, ll_edge_pr, ll_edge_rc, ll_edge_f1, ll_edge_auc = _ll_edge()

    rows = [
        ("Node", "Zero-shot Llama-3.2-3B (prior run)", "—", ll_node_mi, ll_node_ma, ll_node_pr, ll_node_rc, None, None),
        ("Node", "TF-IDF + OvR LogReg — full 93 labels", "test", full["micro_f1"], full["macro_f1"], full["micro_precision"], full["micro_recall"], None, None),
        ("Node", "**TF-IDF + OvR LogReg — common concepts (71)**", "test", common["micro_f1"], common["macro_f1"], common["micro_precision"], common["micro_recall"], None, None),
        ("Edge", "Zero-shot Llama-3.2-3B (prior run, ~all-positive)", "—", None, None, ll_edge_pr, ll_edge_rc, ll_edge_acc, ll_edge_auc),
        ("Edge", "**TF-IDF + pair feats + LogReg (full)**", "test", None, None, edge_full["precision"], edge_full["recall"], edge_full["accuracy"], edge_full["roc_auc"]),
        ("Edge", "Pair features only — no text (C1)", "test", None, None, c1_metrics["precision"], c1_metrics["recall"], c1_metrics["accuracy"], c1_metrics["roc_auc"]),
        ("Edge", "Type + polarity only — no text, no concept id (C2)", "test", None, None, c2_metrics["precision"], c2_metrics["recall"], c2_metrics["accuracy"], c2_metrics["roc_auc"]),
    ]

    # For edge rows we also want F1; for node rows F1 = micro_f1. Add F1 explicitly.
    edge_full_f1 = edge_full["f1"]
    c1_f1 = c1_metrics["f1"]
    c2_f1 = c2_metrics["f1"]
    ll_edge_f1_v = ll_edge_f1

    lines: list[str] = []
    lines.append("# Baseline results — node and edge prediction")
    lines.append("")
    lines.append("## TL;DR")
    lines.append("")
    lines.append(
        f"- **Node (common concepts, 71 labels)**: micro-F1 = **{common['micro_f1']:.4f}**, "
        f"macro-F1 = {common['macro_f1']:.4f} on test."
    )
    lines.append(
        f"- **Edge (full)**: F1 = **{edge_full_f1:.4f}**, ROC-AUC = **{edge_full['roc_auc']:.4f}**, "
        f"accuracy = {edge_full['accuracy']:.4f} on test."
    )
    lines.append(f"- **Shortcut verdict**: {verdict}")
    lines.append("")
    lines.append("## Results table")
    lines.append("")
    lines.append(
        "| Task | Model | Split | Micro-F1 | Macro-F1 | F1 | Precision | Recall | Acc | AUC |"
    )
    lines.append(
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"
    )

    def _row(task, model, split, micro, macro, f1, pr, rc, acc, auc):
        return (
            f"| {task} | {model} | {split} | {fmt(micro)} | {fmt(macro)} | {fmt(f1)} "
            f"| {fmt(pr)} | {fmt(rc)} | {fmt(acc)} | {fmt(auc)} |"
        )

    # Node rows: F1 column == micro_f1.
    lines.append(_row("Node", "Zero-shot Llama-3.2-3B (prior run)", "test", ll_node_mi, ll_node_ma, ll_node_mi, ll_node_pr, ll_node_rc, None, None))
    lines.append(_row("Node", "TF-IDF + OvR LogReg — full 93 labels", "test", full["micro_f1"], full["macro_f1"], full["micro_f1"], full["micro_precision"], full["micro_recall"], None, None))
    lines.append(_row("Node", "**TF-IDF + OvR LogReg — common concepts (71)**", "test", common["micro_f1"], common["macro_f1"], common["micro_f1"], common["micro_precision"], common["micro_recall"], None, None))
    # Edge rows.
    lines.append(_row("Edge", "Zero-shot Llama-3.2-3B (prior run, ~all-positive)", "test", None, None, ll_edge_f1_v, ll_edge_pr, ll_edge_rc, ll_edge_acc, ll_edge_auc))
    lines.append(_row("Edge", "**TF-IDF + pair feats + LogReg (full)**", "test", None, None, edge_full_f1, edge_full["precision"], edge_full["recall"], edge_full["accuracy"], edge_full["roc_auc"]))
    lines.append(_row("Edge", "Pair features only — no text (C1)", "test", None, None, c1_f1, c1_metrics["precision"], c1_metrics["recall"], c1_metrics["accuracy"], c1_metrics["roc_auc"]))
    lines.append(_row("Edge", "Type + polarity only — no text, no concept id (C2)", "test", None, None, c2_f1, c2_metrics["precision"], c2_metrics["recall"], c2_metrics["accuracy"], c2_metrics["roc_auc"]))

    lines.append("")
    lines.append("All baselines: scikit-learn 1.6.1, CPU-only, seed = 42. "
                 "Decision threshold swept on the dev split (step 0.01, maximize "
                 "micro-F1 for node and F1 for edge), then applied to test. "
                 "Train / dev / test sizes: node 602 / 354 / 227 (1,183 entries, "
                 "participant-independent 50/30/20 split); edge 5,494 / 3,206 / "
                 "2,068. Class balance for edge: ~50.5% connected in train, "
                 "~50.7% in test.")
    lines.append("")
    lines.append("## Edge shortcut analysis")
    lines.append("")
    lines.append(verdict)
    lines.append("")
    lines.append("### Top type-pair conditional probabilities (training edges)")
    lines.append("")
    lines.append("| Type pair (sorted) | n | n connected | P(connected) |")
    lines.append("|---|---:|---:|---:|")
    for r in tp_rows[:20]:
        lines.append(
            f"| {r['type_pair']} | {r['n']} | {r['n_connected']} | "
            f"{r['p_connected']:.3f} |"
        )
    lines.append("")
    lines.append("Full table at `modeling/baselines/artifacts/edge_type_pair_table.csv`.")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("pip install scikit-learn scipy joblib    # one-time")
    lines.append("python modeling/baselines/node_baseline.py")
    lines.append("python modeling/baselines/edge_baseline.py")
    lines.append("python modeling/baselines/edge_shortcut.py")
    lines.append("```")
    lines.append("")
    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(lines))

    metrics_out = {
        "node_prediction": {
            "zero_shot_llama_3_2_3b": {
                "test": {
                    "micro_f1": ll_node_mi,
                    "macro_f1": ll_node_ma,
                    "micro_precision": ll_node_pr,
                    "micro_recall": ll_node_rc,
                    "source": str(LLAMA_NODE.relative_to(REPO_ROOT)),
                }
            },
            "tfidf_ovr_logreg": {
                "test_full": full,
                "test_common": common,
                "threshold": node_metrics["threshold"],
                "label_counts": node_metrics["label_counts"],
            },
        },
        "edge_prediction": {
            "zero_shot_llama_3_2_3b": {
                "test": {
                    "accuracy": ll_edge_acc,
                    "precision": ll_edge_pr,
                    "recall": ll_edge_rc,
                    "f1": ll_edge_f1_v,
                    "roc_auc": ll_edge_auc,
                    "source": str(LLAMA_EDGE.relative_to(REPO_ROOT)),
                }
            },
            "tfidf_text_plus_pair_feats_logreg": {
                "test": edge_full,
                "threshold": edge_full_metrics["dev"]["threshold"],
            },
            "pair_features_only": {"test": c1_metrics},
            "type_polarity_only": {"test": c2_metrics},
            "shortcut_verdict": verdict,
            "top_type_pairs": tp_rows[:20],
        },
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics_out, indent=2))


def main() -> None:
    global DATA_DIR, ART_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--art-dir", default=str(ART_DIR))
    parser.add_argument("--edge-metrics", default=None,
                        help="metrics.json of the full edge model to compare against "
                             "(default: <art-dir>/edge_hard/metrics.json, falling back "
                             "to <art-dir>/edge/metrics.json)")
    parser.add_argument("--write-v1-results", action="store_true",
                        help="also write the legacy v1 RESULTS.md/metrics.json")
    args = parser.parse_args()
    DATA_DIR = Path(args.data_dir)
    ART_DIR = Path(args.art_dir)

    random.seed(SEED)
    np.random.seed(SEED)
    ART_DIR.mkdir(parents=True, exist_ok=True)

    print("[shortcut] loading data...")
    train = read_jsonl(DATA_DIR / "train.jsonl")
    dev = read_jsonl(DATA_DIR / "dev.jsonl")
    test = read_jsonl(DATA_DIR / "test.jsonl")
    print(f"[shortcut] sizes: train={len(train)} dev={len(dev)} test={len(test)}")

    y_train = np.array([1 if r["connected"] else 0 for r in train], dtype=int)
    y_dev = np.array([1 if r["connected"] else 0 for r in dev], dtype=int)
    y_test = np.array([1 if r["connected"] else 0 for r in test], dtype=int)

    c1_metrics, _ = run_ablation("pair_only", pair_features_full, train, dev, test, y_train, y_dev, y_test)
    c2_metrics, _ = run_ablation("type_pol_only", pair_features_type_pol, train, dev, test, y_train, y_dev, y_test)

    # C3 — type-pair table.
    tp_rows = type_pair_table(train)
    csv_path = ART_DIR / "edge_type_pair_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["type_pair", "n", "n_connected", "p_connected"])
        for r in tp_rows:
            w.writerow([r["type_pair"], r["n"], r["n_connected"], f"{r['p_connected']:.4f}"])
    print(f"[shortcut/C3] wrote {csv_path}")

    # Pull full edge model metrics (run earlier) for comparison.
    if args.edge_metrics:
        edge_full_metrics_path = Path(args.edge_metrics)
    else:
        edge_full_metrics_path = ART_DIR / "edge_hard" / "metrics.json"
        if not edge_full_metrics_path.exists():
            edge_full_metrics_path = ART_DIR / "edge" / "metrics.json"
    if not edge_full_metrics_path.exists():
        raise SystemExit(
            f"missing {edge_full_metrics_path}; run edge_baseline.py first"
        )
    edge_full_metrics = json.loads(edge_full_metrics_path.read_text())
    edge_full_f1 = edge_full_metrics["test"]["f1"]
    edge_full_auc = edge_full_metrics["test"]["roc_auc"]

    delta_pair = edge_full_f1 - c1_metrics["f1"]
    delta_tp = edge_full_f1 - c2_metrics["f1"]
    pct_recoverable = (c1_metrics["f1"] / edge_full_f1) * 100.0 if edge_full_f1 > 0 else 0.0

    verdict = (
        f"Of the full-text edge model's F1 = {edge_full_f1:.4f} (AUC = {edge_full_auc:.4f}), "
        f"a model with NO journal text — only pair features (types, polarities, concept ids) — "
        f"reaches F1 = {c1_metrics['f1']:.4f} (AUC = {c1_metrics['roc_auc']:.4f}), "
        f"i.e. ≈ {pct_recoverable:.0f}% of full performance is recoverable from non-text structure alone. "
        f"Restricting further to type + polarity only (no concept identity, no text) reaches "
        f"F1 = {c2_metrics['f1']:.4f} (AUC = {c2_metrics['roc_auc']:.4f}). "
        f"The text therefore adds ~{delta_pair:.3f} F1 over structure-with-concept-ids and "
        f"~{delta_tp:.3f} F1 over pure template structure. "
        f"Caveat: edge records derived from the same entry share identical text, so the "
        f"text-using model is partly memorizing entry-level co-occurrence patterns rather "
        f"than judging the candidate pair specifically."
    )
    print("[shortcut] verdict:", verdict)

    summary = {
        "full_edge_model_test": edge_full_metrics["test"],
        "c1_pair_features_only_test": c1_metrics,
        "c2_type_polarity_only_test": c2_metrics,
        "delta_f1_full_minus_pair_only": delta_pair,
        "delta_f1_full_minus_type_pol_only": delta_tp,
        "pct_full_f1_recoverable_from_pair_features_only": pct_recoverable,
        "verdict": verdict,
        "top_type_pairs": tp_rows[:20],
    }
    (ART_DIR / "edge_shortcut_summary.json").write_text(json.dumps(summary, indent=2))

    if args.write_v1_results:
        # Legacy v1 path: pull node metrics + Llama priors, write RESULTS.md.
        node_metrics_path = ART_DIR / "node" / "metrics.json"
        if not node_metrics_path.exists():
            raise SystemExit(f"missing {node_metrics_path}; run node_baseline.py first")
        node_metrics = json.loads(node_metrics_path.read_text())
        llama = load_llama()
        write_results(node_metrics, edge_full_metrics, c1_metrics, c2_metrics, tp_rows, llama, verdict)
        print(f"Wrote: {RESULTS_DIR / 'RESULTS.md'}")

    print("\n" + "=" * 78)
    print("SHORTCUT ABLATION SUMMARY")
    print("=" * 78)
    print(f"EDGE  full        F1={edge_full_metrics['test']['f1']:.4f}  "
          f"AUC={edge_full_metrics['test']['roc_auc']:.4f}  "
          f"acc={edge_full_metrics['test']['accuracy']:.4f}")
    print(f"EDGE  C1 pair-only          F1={c1_metrics['f1']:.4f}  AUC={c1_metrics['roc_auc']:.4f}  "
          f"acc={c1_metrics['accuracy']:.4f}")
    print(f"EDGE  C2 type+polarity-only F1={c2_metrics['f1']:.4f}  AUC={c2_metrics['roc_auc']:.4f}  "
          f"acc={c2_metrics['accuracy']:.4f}")
    print("-" * 78)
    print(f"VERDICT: {verdict}")
    print("=" * 78)


if __name__ == "__main__":
    main()
