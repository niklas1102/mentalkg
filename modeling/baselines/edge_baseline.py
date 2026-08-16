"""CPU-only TF-IDF + pair features + Logistic Regression baseline for edge prediction.

Binary classification: given a journal text and two candidate nodes (with
concept_id, type, polarity), predict whether they are connected. Features =
TF-IDF (word 1-2 grams) of the entry text, hstacked with one-hot pair features
(types, type-pair, polarities, polarity-pair, concept ids, concept-id-pair).

Threshold swept on dev for max F1, applied to test. Reports accuracy, precision,
recall, F1, ROC-AUC.

Usage (v2):
    python modeling/baselines/edge_baseline.py \
        [--data-dir modeling/data_v2/edge_prediction] [--art-dir ...] [--tag hard]

Artifacts default to modeling/baselines/artifacts_v2/edge_<tag>/.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
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
DATA_DEFAULT = REPO_ROOT / "modeling" / "data_v2" / "edge_prediction"
ART_ROOT_DEFAULT = REPO_ROOT / "modeling" / "baselines" / "artifacts_v2"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pair_features(record: dict, *, include_cid: bool = True) -> dict:
    a, b = record["node_a"], record["node_b"]
    ta, tb = a["type"], b["type"]
    pa, pb = a["polarity"], b["polarity"]
    type_pair = "__".join(sorted([ta, tb]))
    pol_pair = "__".join(sorted([pa, pb]))
    feats = {
        f"type_a={ta}": 1.0,
        f"type_b={tb}": 1.0,
        f"type_pair={type_pair}": 1.0,
        f"pol_a={pa}": 1.0,
        f"pol_b={pb}": 1.0,
        f"pol_pair={pol_pair}": 1.0,
    }
    if include_cid:
        ca, cb = a["concept_id"], b["concept_id"]
        cid_pair = "__".join(sorted([ca, cb]))
        feats[f"cid_a={ca}"] = 1.0
        feats[f"cid_b={cb}"] = 1.0
        feats[f"cid_pair={cid_pair}"] = 1.0
    return feats


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DEFAULT))
    parser.add_argument("--art-dir", default=None,
                        help="default: artifacts_v2/edge_<tag>/")
    parser.add_argument("--tag", default="hard",
                        help="label for this run (e.g. hard, random)")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    DATA_DIR = Path(args.data_dir)
    ART_DIR = Path(args.art_dir) if args.art_dir else ART_ROOT_DEFAULT / f"edge_{args.tag}"
    ART_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[edge_baseline] loading data from {DATA_DIR} (tag={args.tag})...")
    train = read_jsonl(DATA_DIR / "train.jsonl")
    dev = read_jsonl(DATA_DIR / "dev.jsonl")
    test = read_jsonl(DATA_DIR / "test.jsonl")
    print(f"[edge_baseline] sizes: train={len(train)} dev={len(dev)} test={len(test)}")

    y_train = np.array([1 if r["connected"] else 0 for r in train], dtype=int)
    y_dev = np.array([1 if r["connected"] else 0 for r in dev], dtype=int)
    y_test = np.array([1 if r["connected"] else 0 for r in test], dtype=int)
    print(
        f"[edge_baseline] connected fraction: train={y_train.mean():.3f} "
        f"dev={y_dev.mean():.3f} test={y_test.mean():.3f}"
    )

    print("[edge_baseline] fitting TF-IDF on text...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
    Xt_train = vectorizer.fit_transform([r["text"] for r in train])
    Xt_dev = vectorizer.transform([r["text"] for r in dev])
    Xt_test = vectorizer.transform([r["text"] for r in test])
    print(f"[edge_baseline] text feature_dim={Xt_train.shape[1]}")

    print("[edge_baseline] fitting pair feature DictVectorizer...")
    pair_vec = DictVectorizer(sparse=True)
    Xp_train = pair_vec.fit_transform([pair_features(r) for r in train])
    Xp_dev = pair_vec.transform([pair_features(r) for r in dev])
    Xp_test = pair_vec.transform([pair_features(r) for r in test])
    print(f"[edge_baseline] pair feature_dim={Xp_train.shape[1]}")

    X_train = hstack([Xt_train, Xp_train]).tocsr()
    X_dev = hstack([Xt_dev, Xp_dev]).tocsr()
    X_test = hstack([Xt_test, Xp_test]).tocsr()
    print(f"[edge_baseline] total feature_dim={X_train.shape[1]}")

    print("[edge_baseline] fitting LogisticRegression...")
    t0 = time.time()
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=SEED,
    )
    clf.fit(X_train, y_train)
    print(f"[edge_baseline] fit done in {time.time() - t0:.1f}s")

    print("[edge_baseline] sweeping threshold on dev...")
    scores_dev = clf.predict_proba(X_dev)[:, 1]
    best_t, best_f1 = 0.5, -1.0
    for t in THRESHOLD_GRID:
        f1 = f1_score(y_dev, (scores_dev >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    print(f"[edge_baseline] best threshold t*={best_t:.2f} dev_f1={best_f1:.4f}")

    scores_test = clf.predict_proba(X_test)[:, 1]
    y_pred = (scores_test >= best_t).astype(int)
    test_metrics = evaluate(y_test, y_pred, scores_test)
    print("[edge_baseline] TEST :", json.dumps(test_metrics, indent=None))

    # Save artifacts.
    joblib.dump(vectorizer, ART_DIR / "vectorizer.joblib")
    joblib.dump(pair_vec, ART_DIR / "pair_vectorizer.joblib")
    joblib.dump(clf, ART_DIR / "model.joblib")
    meta = {
        "task": "edge_prediction",
        "model": "tfidf_text_+_pair_feats_+_logreg_balanced",
        "negatives_tag": args.tag,
        "data_dir": str(DATA_DIR),
        "seed": SEED,
        "threshold": best_t,
        "threshold_grid": [float(t) for t in THRESHOLD_GRID],
        "dev_f1_at_threshold": best_f1,
        "text_feature_dim": int(Xt_train.shape[1]),
        "pair_feature_dim": int(Xp_train.shape[1]),
        "total_feature_dim": int(X_train.shape[1]),
        "train_size": len(train),
        "dev_size": len(dev),
        "test_size": len(test),
    }
    (ART_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    metrics = {
        "task": "edge_prediction",
        "model": meta["model"],
        "negatives_tag": args.tag,
        "dev": {"f1_at_best_threshold": best_f1, "threshold": best_t},
        "test": test_metrics,
    }
    (ART_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    with (ART_DIR / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r, gold, p, s in zip(test, y_test, y_pred, scores_test):
            f.write(
                json.dumps(
                    {
                        "example_id": r["example_id"],
                        "gold": int(gold),
                        "pred": int(p),
                        "score": float(s),
                        "node_a": {
                            "concept_id": r["node_a"]["concept_id"],
                            "type": r["node_a"]["type"],
                            "polarity": r["node_a"]["polarity"],
                        },
                        "node_b": {
                            "concept_id": r["node_b"]["concept_id"],
                            "type": r["node_b"]["type"],
                            "polarity": r["node_b"]["polarity"],
                        },
                    }
                )
                + "\n"
            )

    print(f"[edge_baseline] artifacts written to {ART_DIR}")


if __name__ == "__main__":
    main()
