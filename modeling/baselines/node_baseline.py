"""CPU-only TF-IDF + One-vs-Rest Logistic Regression baseline for node prediction (v2).

Multi-label classification over concept ids. Fits TF-IDF (word 1-2 grams) on the
journal text, trains an OvR Logistic Regression with class_weight="balanced",
sweeps a single global decision threshold on the dev split (max micro-F1), and
evaluates on test for both the full label vocab and the "common concepts"
subset (labels with train_support >= 20 in THIS run's train subset).

v2: --language all|en|de filters the data (en/de train language-specific
models); with --language all the test metrics are additionally broken down per
language at the same threshold.

Usage:
    python modeling/baselines/node_baseline.py [--language all|en|de]
        [--data-dir ...] [--art-dir ...]

Artifacts default to modeling/baselines/artifacts_v2/node_<language>/.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
COMMON_LABEL_MIN_TRAIN = 20
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DEFAULT = REPO_ROOT / "modeling" / "data_v2" / "node_prediction"
ART_ROOT_DEFAULT = REPO_ROOT / "modeling" / "baselines" / "artifacts_v2"


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
    parser.add_argument("--language", choices=("all", "en", "de"), default="all")
    parser.add_argument("--data-dir", default=str(DATA_DEFAULT))
    parser.add_argument("--art-dir", default=None,
                        help="default: artifacts_v2/node_<language>/")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    data_dir = Path(args.data_dir)
    art_dir = Path(args.art_dir) if args.art_dir else ART_ROOT_DEFAULT / f"node_{args.language}"
    art_dir.mkdir(parents=True, exist_ok=True)
    tag = f"node_baseline/{args.language}"

    print(f"[{tag}] loading data...")
    train = read_jsonl(data_dir / "train.jsonl")
    dev = read_jsonl(data_dir / "dev.jsonl")
    test = read_jsonl(data_dir / "test.jsonl")
    if args.language != "all":
        train = [r for r in train if r.get("language") == args.language]
        dev = [r for r in dev if r.get("language") == args.language]
        test = [r for r in test if r.get("language") == args.language]

    # Label vocab + common filter derived from THIS run's train subset (same
    # logic/threshold as v1: labels with train support >= 20).
    support = Counter(c for r in train for c in r["labels"])
    label_vocab = sorted(support)
    common_labels = [l for l in label_vocab if support[l] >= COMMON_LABEL_MIN_TRAIN]
    print(f"[{tag}] sizes: train={len(train)} dev={len(dev)} test={len(test)} "
          f"labels={len(label_vocab)} common(>={COMMON_LABEL_MIN_TRAIN})={len(common_labels)}")

    mlb = MultiLabelBinarizer(classes=label_vocab)
    mlb.fit([label_vocab])
    Y_train = mlb.transform([r["labels"] for r in train])
    Y_dev = mlb.transform([r["labels"] for r in dev])
    Y_test = mlb.transform([r["labels"] for r in test])
    common_idx = np.array([label_vocab.index(l) for l in common_labels], dtype=int)

    print(f"[{tag}] fitting TF-IDF...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=True,
        strip_accents="unicode",
    )
    X_train = vectorizer.fit_transform([r["text"] for r in train])
    X_dev = vectorizer.transform([r["text"] for r in dev])
    X_test = vectorizer.transform([r["text"] for r in test])
    print(f"[{tag}] feature_dim={X_train.shape[1]}")

    print(f"[{tag}] fitting OvR LogisticRegression...")
    t0 = time.time()
    clf = OneVsRestClassifier(
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=SEED,
        ),
        n_jobs=-1,
    )
    clf.fit(X_train, Y_train)
    print(f"[{tag}] fit done in {time.time() - t0:.1f}s")

    print(f"[{tag}] sweeping threshold on dev...")
    scores_dev = clf.predict_proba(X_dev)
    best_t, best_f1 = 0.5, -1.0
    for t in THRESHOLD_GRID:
        f1 = f1_score(Y_dev, (scores_dev >= t).astype(int), average="micro", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    print(f"[{tag}] best threshold t*={best_t:.2f} dev_micro_f1={best_f1:.4f}")

    scores_test = clf.predict_proba(X_test)
    Y_pred = (scores_test >= best_t).astype(int)

    metrics_full = evaluate(Y_test, Y_pred)
    metrics_common = evaluate(Y_test[:, common_idx], Y_pred[:, common_idx])
    print(f"[{tag}] TEST full   :", json.dumps(metrics_full))
    print(f"[{tag}] TEST common :", json.dumps(metrics_common))

    # Per-language breakdown of this model at the same threshold (informative
    # mainly for --language all).
    per_language = {}
    langs = sorted({r.get("language", "?") for r in test})
    if len(langs) > 1:
        for lg in langs:
            idx = np.array([i for i, r in enumerate(test) if r.get("language") == lg], dtype=int)
            per_language[lg] = {
                "full": evaluate(Y_test[idx], Y_pred[idx]),
                "common": evaluate(Y_test[np.ix_(idx, common_idx)], Y_pred[np.ix_(idx, common_idx)]),
            }
            print(f"[{tag}] TEST common [{lg}]:", json.dumps(per_language[lg]["common"]))

    joblib.dump(vectorizer, art_dir / "vectorizer.joblib")
    joblib.dump(clf, art_dir / "model.joblib")
    joblib.dump(mlb, art_dir / "mlb.joblib")

    meta = {
        "task": "node_prediction",
        "model": "tfidf_word_1_2_+_ovr_logreg_balanced",
        "language": args.language,
        "data_dir": str(data_dir),
        "seed": SEED,
        "threshold": best_t,
        "dev_micro_f1_at_threshold": best_f1,
        "feature_dim": int(X_train.shape[1]),
        "label_vocab_size": len(label_vocab),
        "common_labels": common_labels,
        "common_label_min_train": COMMON_LABEL_MIN_TRAIN,
        "train_size": len(train),
        "dev_size": len(dev),
        "test_size": len(test),
    }
    (art_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    metrics = {
        "task": "node_prediction",
        "model": meta["model"],
        "language": args.language,
        "test": {"full": metrics_full, "common": metrics_common},
        "test_per_language": per_language,
        "dev": {"micro_f1_at_best_threshold": best_f1},
        "threshold": best_t,
        "label_counts": {"full": len(label_vocab), "common": len(common_labels)},
    }
    (art_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    with (art_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row, gold_vec, pred_vec, score_vec in zip(test, Y_test, Y_pred, scores_test):
            top_idx = np.argsort(-score_vec)[:15]
            f.write(json.dumps({
                "example_id": row["example_id"],
                "language": row.get("language"),
                "gold": [label_vocab[i] for i in np.where(gold_vec == 1)[0]],
                "pred": [label_vocab[i] for i in np.where(pred_vec == 1)[0]],
                "top15": [{"label": label_vocab[i], "score": float(score_vec[i])}
                          for i in top_idx],
            }) + "\n")

    print(f"[{tag}] artifacts written to {art_dir}")


if __name__ == "__main__":
    main()
