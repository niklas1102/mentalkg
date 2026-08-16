"""XLM-RoBERTa multi-label baseline for node prediction (v2).

Fine-tunes xlm-roberta-base with a 130-way sigmoid head (BCE) on the raw entry
text. Label vocabulary is the canonical 130-label index from
data_v2/node_prediction/labels.json (not re-derived from train). Threshold:
single global value swept on dev for max micro-F1 (same grid and procedure as
node_baseline.py), applied to test; per-language test breakdown at the same
threshold. Best checkpoint selected by dev micro-F1.

Usage:
    python modeling/baselines/node_bert.py [--language all|en|de]
        [--data-dir ...] [--art-dir ...] [--smoke-test]

Artifacts default to modeling/baselines/artifacts_v2_bert/node_xlmr_<language>/.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from bert_utils import (
    THRESHOLD_GRID,
    EncodedDataset,
    get_device,
    read_jsonl,
    score_all,
    set_seed,
    train_model,
)

COMMON_LABEL_MIN_TRAIN = 20


class FocalLoss(torch.nn.Module):
    """Multi-label sigmoid focal loss; per-label alpha from train priors."""

    def __init__(self, alpha, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)  # (C,), weight on the positive term
        self.gamma = gamma

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        pt = torch.where(targets == 1, p, 1 - p)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        return (alpha_t * (1 - pt).pow(self.gamma) * bce).mean()


def tune_per_label_thresholds(probs_dev, Y_dev, grid, fallback: float):
    """Per-label threshold maximizing that label's dev F1; no-dev-positive
    labels fall back to the global threshold."""
    thresholds = np.full(Y_dev.shape[1], fallback)
    for j in range(Y_dev.shape[1]):
        yj = Y_dev[:, j]
        n_pos = yj.sum()
        if n_pos == 0:
            continue
        pj = probs_dev[:, j]
        best_f1, best_t = -1.0, fallback
        for t in grid:
            pred = pj >= t
            tp = int((pred & (yj == 1)).sum())
            if tp == 0:
                f1 = 0.0
            else:
                prec = tp / max(int(pred.sum()), 1)
                rec = tp / int(n_pos)
                f1 = 2 * prec * rec / (prec + rec)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[j] = best_t
    return thresholds

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DEFAULT = REPO_ROOT / "modeling" / "data_v2" / "node_prediction"
ART_ROOT_DEFAULT = REPO_ROOT / "modeling" / "baselines" / "artifacts_v2_bert"


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import precision_score, recall_score

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


def encode_texts(tokenizer, texts: list[str], max_len: int) -> tuple[list[list[int]], int]:
    """Tokenize with truncation to max_len; return (ids, n_truncated)."""
    enc = tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
    n_truncated = 0
    out = []
    sep_id = tokenizer.sep_token_id
    for ids in enc:
        if len(ids) > max_len:
            n_truncated += 1
            ids = ids[: max_len - 1] + [sep_id]
        out.append(ids)
    return out, n_truncated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("all", "en", "de"), default="all")
    parser.add_argument("--data-dir", default=str(DATA_DEFAULT))
    parser.add_argument("--art-dir", default=None,
                        help="default: artifacts_v2_bert/node_xlmr_<language>/")
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos-weight", action="store_true",
                        help="per-label BCE pos_weight = neg/pos on this run's "
                             "train subset, clipped to [1, 50] (escapes the "
                             "base-rate collapse; see reports/node_collapse_diagnosis.md)")
    parser.add_argument("--head-bias-init", action="store_true",
                        help="initialize the classifier bias to log(p/(1-p)) from "
                             "train label frequencies, so training starts AT the "
                             "base-rate solution instead of descending into it")
    parser.add_argument("--loss", choices=("bce", "focal"), default="bce",
                        help="focal: gamma 2, per-label alpha = 1 - train prior")
    parser.add_argument("--per-label-thresholds", action="store_true",
                        help="ALSO tune one threshold per label on dev (grid "
                             "0.05-0.95 step 0.01) and report test metrics under "
                             "both the global and per-label modes")
    parser.add_argument("--skip-test", action="store_true",
                        help="sweep cell: train + dev selection only, no test "
                             "evaluation (protocol: test is scored once, on the "
                             "final chosen config)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="200 train / 100 dev / 100 test, 1 epoch")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    data_dir = Path(args.data_dir)
    art_dir = (
        Path(args.art_dir) if args.art_dir
        else ART_ROOT_DEFAULT / f"node_xlmr_{args.language}"
    )
    art_dir.mkdir(parents=True, exist_ok=True)
    tag = f"[node_bert/{args.language}]"
    print(f"{tag} device={device}", flush=True)

    print(f"{tag} loading data...", flush=True)
    train = read_jsonl(data_dir / "train.jsonl")
    dev = read_jsonl(data_dir / "dev.jsonl")
    test = read_jsonl(data_dir / "test.jsonl")
    if args.language != "all":
        train = [r for r in train if r.get("language") == args.language]
        dev = [r for r in dev if r.get("language") == args.language]
        test = [r for r in test if r.get("language") == args.language]
    if args.smoke_test:
        train, dev, test = train[:200], dev[:100], test[:100]
        args.epochs = 1

    # Canonical fixed 130-label vocabulary; common filter (>=20 train support)
    # computed on this run's train subset over that fixed vocab, mirroring
    # node_baseline.py (for --language all, common == full == 130 in v2).
    labels_meta = json.loads((data_dir / "labels.json").read_text())
    label_vocab = labels_meta["index_to_label"]
    support = Counter(c for r in train for c in r["labels"])
    common_labels = [l for l in label_vocab if support[l] >= COMMON_LABEL_MIN_TRAIN]
    print(f"{tag} sizes: train={len(train)} dev={len(dev)} test={len(test)} "
          f"labels={len(label_vocab)} common(>={COMMON_LABEL_MIN_TRAIN})={len(common_labels)}",
          flush=True)

    mlb = MultiLabelBinarizer(classes=label_vocab)
    mlb.fit([label_vocab])
    Y_train = mlb.transform([r["labels"] for r in train])
    Y_dev = mlb.transform([r["labels"] for r in dev])
    Y_test = mlb.transform([r["labels"] for r in test])
    common_idx = np.array([label_vocab.index(l) for l in common_labels], dtype=int)

    print(f"{tag} tokenizing...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids_train, trunc_train = encode_texts(tokenizer, [r["text"] for r in train], args.max_len)
    ids_dev, trunc_dev = encode_texts(tokenizer, [r["text"] for r in dev], args.max_len)
    ids_test, trunc_test = encode_texts(tokenizer, [r["text"] for r in test], args.max_len)
    truncation = {
        "train": {"n": trunc_train, "frac": trunc_train / max(len(train), 1)},
        "dev": {"n": trunc_dev, "frac": trunc_dev / max(len(dev), 1)},
        "test": {"n": trunc_test, "frac": trunc_test / max(len(test), 1)},
    }
    print(f"{tag} truncated@{args.max_len}: {json.dumps(truncation)}", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=len(label_vocab),
        problem_type="multi_label_classification",
    ).to(device)

    if args.head_bias_init:
        p = Y_train.mean(axis=0).clip(1e-4, 1 - 1e-4)
        prior_logits = np.log(p / (1 - p))
        with torch.no_grad():
            bias = model.classifier.out_proj.bias
            bias.copy_(torch.tensor(prior_logits, dtype=bias.dtype, device=bias.device))
        print(f"{tag} head bias init to prior logits: "
              f"min={prior_logits.min():.2f} median={np.median(prior_logits):.2f} "
              f"max={prior_logits.max():.2f}", flush=True)

    def dev_select(probs: np.ndarray) -> tuple[float, float]:
        best_t, best_f1 = 0.5, -1.0
        for t in THRESHOLD_GRID:
            f1 = f1_score(Y_dev, (probs >= t).astype(int), average="micro", zero_division=0)
            if f1 > best_f1:
                best_t, best_f1 = float(t), float(f1)
        return best_t, best_f1

    if args.loss == "focal" and args.pos_weight:
        raise SystemExit("--loss focal and --pos-weight are mutually exclusive")
    loss_fn = None
    if args.pos_weight:
        n_pos = Y_train.sum(axis=0).astype(float)
        pw = np.clip((len(Y_train) - n_pos) / np.maximum(n_pos, 1.0), 1.0, 50.0)
        print(f"{tag} pos_weight: min={pw.min():.1f} median={np.median(pw):.1f} "
              f"max={pw.max():.1f}", flush=True)
        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pw, dtype=torch.float32, device=device)
        )
    elif args.loss == "focal":
        prior = Y_train.mean(axis=0).clip(1e-4, 1 - 1e-4)
        alpha = torch.tensor(1.0 - prior, dtype=torch.float32, device=device)
        loss_fn = FocalLoss(alpha, gamma=2.0).to(device)
        print(f"{tag} focal loss: gamma=2, alpha=1-prior "
              f"(median alpha={float(alpha.median()):.3f})", flush=True)

    best = train_model(
        model,
        EncodedDataset(ids_train, Y_train),
        ids_dev,
        dev_select,
        args=args,
        device=device,
        pad_id=tokenizer.pad_token_id,
        art_dir=art_dir,
        log_prefix=tag,
        loss_fn=loss_fn,
    )
    print(f"{tag} best: {json.dumps(best)}", flush=True)

    print(f"{tag} loading best checkpoint...", flush=True)
    model.load_state_dict(torch.load(art_dir / "checkpoint" / "model.pt",
                                     map_location=device))
    best_t = best["best_threshold"]

    # Dev metrics of the BEST checkpoint, always reported (sweep transparency).
    logits_dev_best = score_all(model, ids_dev, device, tokenizer.pad_token_id,
                                batch_size=args.eval_batch_size,
                                log_prefix=f"{tag} [dev best]")
    probs_dev_best = 1.0 / (1.0 + np.exp(-logits_dev_best))
    dev_global = evaluate(Y_dev, (probs_dev_best >= best_t).astype(int))
    print(f"{tag} DEV (best ckpt, global thr {best_t:.2f}):",
          json.dumps(dev_global), flush=True)

    per_label_thr = None
    dev_per_label = None
    if args.per_label_thresholds:
        per_label_thr = tune_per_label_thresholds(
            probs_dev_best, Y_dev, THRESHOLD_GRID, best_t)
        dev_per_label = evaluate(Y_dev, (probs_dev_best >= per_label_thr).astype(int))
        print(f"{tag} DEV (per-label thresholds):",
              json.dumps(dev_per_label), flush=True)

    metrics_full = metrics_common = None
    metrics_pl = None
    per_language = {}
    scores_test = Y_pred = None
    if args.skip_test:
        print(f"{tag} --skip-test: sweep cell, test not evaluated", flush=True)
    else:
        print(f"{tag} scoring test...", flush=True)
        logits_test = score_all(model, ids_test, device, tokenizer.pad_token_id,
                                batch_size=args.eval_batch_size, log_prefix=f"{tag} [test]")
        scores_test = 1.0 / (1.0 + np.exp(-logits_test))
        Y_pred = (scores_test >= best_t).astype(int)

        metrics_full = evaluate(Y_test, Y_pred)
        metrics_common = evaluate(Y_test[:, common_idx], Y_pred[:, common_idx])
        print(f"{tag} TEST full   :", json.dumps(metrics_full), flush=True)
        print(f"{tag} TEST common :", json.dumps(metrics_common), flush=True)

        if per_label_thr is not None:
            Y_pred_pl = (scores_test >= per_label_thr).astype(int)
            metrics_pl = {
                "full": evaluate(Y_test, Y_pred_pl),
                "common": evaluate(Y_test[:, common_idx], Y_pred_pl[:, common_idx]),
            }
            print(f"{tag} TEST full (per-label thr):",
                  json.dumps(metrics_pl["full"]), flush=True)

        langs = sorted({r.get("language", "?") for r in test})
        if len(langs) > 1:
            for lg in langs:
                idx = np.array([i for i, r in enumerate(test) if r.get("language") == lg], dtype=int)
                per_language[lg] = {
                    "full": evaluate(Y_test[idx], Y_pred[idx]),
                    "common": evaluate(Y_test[np.ix_(idx, common_idx)], Y_pred[np.ix_(idx, common_idx)]),
                }
                if per_label_thr is not None:
                    per_language[lg]["full_per_label_thr"] = evaluate(
                        Y_test[idx], (scores_test[idx] >= per_label_thr).astype(int))
                print(f"{tag} TEST common [{lg}]:", json.dumps(per_language[lg]["common"]), flush=True)

    tokenizer.save_pretrained(art_dir / "checkpoint")

    meta = {
        "task": "node_prediction",
        "model": f"{args.model}_multilabel_bce",
        "language": args.language,
        "data_dir": str(data_dir),
        "seed": args.seed,
        "threshold": best_t,
        "dev_micro_f1_at_threshold": best["best_dev_score"],
        "best_epoch": best["best_epoch"],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_len": args.max_len,
        "device": str(device),
        "precision": "fp32",
        "pos_weight": args.pos_weight,
        "head_bias_init": args.head_bias_init,
        "loss": args.loss,
        "skip_test": args.skip_test,
        "per_label_thresholds": (
            [round(float(t), 2) for t in per_label_thr]
            if per_label_thr is not None else None
        ),
        "truncation": truncation,
        "label_vocab_size": len(label_vocab),
        "common_labels": common_labels,
        "common_label_min_train": COMMON_LABEL_MIN_TRAIN,
        "train_size": len(train),
        "dev_size": len(dev),
        "test_size": len(test),
        "smoke_test": args.smoke_test,
    }
    (art_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    metrics = {
        "task": "node_prediction",
        "model": meta["model"],
        "language": args.language,
        "test": {"full": metrics_full, "common": metrics_common},
        "test_per_label_thr": metrics_pl,
        "test_per_language": per_language,
        "dev": {
            "micro_f1_at_best_threshold": best["best_dev_score"],
            "best_ckpt_global": dev_global,
            "per_label": dev_per_label,
        },
        "threshold": best_t,
        "label_counts": {"full": len(label_vocab), "common": len(common_labels)},
    }
    (art_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if not args.skip_test:
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

    print(f"{tag} artifacts written to {art_dir}", flush=True)


if __name__ == "__main__":
    main()
