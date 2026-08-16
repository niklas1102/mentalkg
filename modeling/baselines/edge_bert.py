"""XLM-RoBERTa pair-aware baseline for edge prediction (v2).

Binary classification: given a journal entry and two candidate nodes, predict
whether they are connected. Unlike the TF-IDF baseline (whose text features are
identical for every pair from the same entry), this model sees a pair-specific
input, so it CAN discriminate between pairs of the same entry.

Input format (uniform for all examples — the data has no character spans and
the short node surfaces appear verbatim in the text <20% of the time, so no
inline span marking is attempted):

    <s> {entry text} </s></s> [N1] {label} ({type}, {polarity}, {time}) [/N1]
    </s> [N2] {label} ({type}, {polarity}, {time}) [/N2] </s>

[N1]/[/N1]/[N2]/[/N2] are added as special tokens (embeddings resized). The
suffix is never truncated; the entry text is right-truncated if the total
exceeds --max-len (rate logged).

Training set is subsampled BY ENTRY (--train-entry-frac, default 0.25): a
seed-42 uniform sample of entry ids, keeping ALL candidate pairs of selected
entries (preserves the per-entry pair structure of the hard negatives). Dev
and test are used in full. Threshold: swept on dev for max binary F1 (same
grid/procedure as edge_baseline.py), applied to test. Best checkpoint by
dev F1.

Ablation (--no-text): the entry text is replaced by the constant string
"entry" before tokenization; the node marker suffix is unchanged. Same
architecture, capacity and training procedure — isolates how much of the
score comes from the pair metadata alone.

Usage:
    python modeling/baselines/edge_bert.py \
        [--data-dir modeling/data_v2/edge_prediction] [--tag hard]
        [--train-entry-frac 0.25] [--no-text] [--smoke-test]

Artifacts default to modeling/baselines/artifacts_v2_bert/edge_xlmr_<tag>/
(with --no-text: edge_xlmr_<tag>_notext/).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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

MARKER_TOKENS = ["[N1]", "[/N1]", "[N2]", "[/N2]"]
NO_TEXT_PLACEHOLDER = "entry"

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DEFAULT = REPO_ROOT / "modeling" / "data_v2" / "edge_prediction"
ART_ROOT_DEFAULT = REPO_ROOT / "modeling" / "baselines" / "artifacts_v2_bert"


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


def entry_id(record: dict) -> str:
    return record["example_id"].split("__")[0]


def node_suffix(marker: str, node: dict) -> str:
    ta = (node.get("time_anchor") or {}).get("text") or "unknown"
    return f"[{marker}] {node['label']} ({node['type']}, {node['polarity']}, {ta}) [/{marker}]"


def encode_split(
    tokenizer, records: list[dict], max_len: int, no_text: bool = False
) -> tuple[list[list[int]], int]:
    """Entry text + double-sep + node suffixes; suffix never truncated.

    With no_text=True the entry text is replaced by NO_TEXT_PLACEHOLDER;
    the suffix and the special-token frame are identical.
    """
    bos, sep = tokenizer.bos_token_id, tokenizer.sep_token_id
    entry_cache: dict[str, list[int]] = {}
    ids_out: list[list[int]] = []
    n_truncated = 0
    for r in records:
        eid = entry_id(r)
        entry_ids = entry_cache.get(eid)
        if entry_ids is None:
            body_text = NO_TEXT_PLACEHOLDER if no_text else r["text"]
            entry_ids = tokenizer(body_text, add_special_tokens=False)["input_ids"]
            entry_cache[eid] = entry_ids
        suffix = (
            tokenizer(node_suffix("N1", r["node_a"]), add_special_tokens=False)["input_ids"]
            + [sep]
            + tokenizer(node_suffix("N2", r["node_b"]), add_special_tokens=False)["input_ids"]
        )
        # <s> entry </s></s> suffix </s>  -> 4 structural tokens
        budget = max_len - len(suffix) - 4
        body = entry_ids
        if len(body) > budget:
            n_truncated += 1
            body = body[:budget]
        ids_out.append([bos] + body + [sep, sep] + suffix + [sep])
    return ids_out, n_truncated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DATA_DEFAULT))
    parser.add_argument("--art-dir", default=None,
                        help="default: artifacts_v2_bert/edge_xlmr_<tag>/")
    parser.add_argument("--tag", default="hard",
                        help="label for this run (e.g. hard, random)")
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--train-entry-frac", type=float, default=0.25,
                        help="fraction of train ENTRIES kept (all their pairs)")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-text", action="store_true",
                        help="replace entry text with the constant 'entry' (C3 ablation)")
    parser.add_argument("--skip-test", action="store_true",
                        help="sweep cell: train + dev selection only, no test "
                             "evaluation (protocol: test scored once, final config only)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="200 train / 100 dev / 100 test pairs, 1 epoch")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    data_dir = Path(args.data_dir)
    run_name = f"{args.tag}_notext" if args.no_text else args.tag
    art_dir = (
        Path(args.art_dir) if args.art_dir
        else ART_ROOT_DEFAULT / f"edge_xlmr_{run_name}"
    )
    art_dir.mkdir(parents=True, exist_ok=True)
    tag = f"[edge_bert/{run_name}]"
    print(f"{tag} device={device}", flush=True)

    print(f"{tag} loading data from {data_dir}...", flush=True)
    train = read_jsonl(data_dir / "train.jsonl")
    dev = read_jsonl(data_dir / "dev.jsonl")
    test = read_jsonl(data_dir / "test.jsonl")
    n_train_full = len(train)

    # Subsample train by entry: keep all pairs of a seed-42 uniform sample of
    # entry ids. Dev/test untouched.
    n_entries_full = len({entry_id(r) for r in train})
    if args.train_entry_frac < 1.0:
        entries = sorted({entry_id(r) for r in train})
        rng = random.Random(args.seed)
        keep = set(rng.sample(entries, round(args.train_entry_frac * len(entries))))
        train = [r for r in train if entry_id(r) in keep]
        # participant-independent splits -> no leakage possible; assert anyway
        train_pids = {r["participant_id"] for r in train}
        assert not train_pids & {r["participant_id"] for r in dev}
        assert not train_pids & {r["participant_id"] for r in test}
        print(f"{tag} train subsample: {len(keep)}/{n_entries_full} entries, "
              f"{len(train)}/{n_train_full} pairs", flush=True)

    if args.smoke_test:
        train, dev, test = train[:200], dev[:100], test[:100]
        args.epochs = 1

    y_train = np.array([1 if r["connected"] else 0 for r in train], dtype=int)
    y_dev = np.array([1 if r["connected"] else 0 for r in dev], dtype=int)
    y_test = np.array([1 if r["connected"] else 0 for r in test], dtype=int)
    print(f"{tag} sizes: train={len(train)} dev={len(dev)} test={len(test)} | "
          f"connected fraction: train={y_train.mean():.3f} "
          f"dev={y_dev.mean():.3f} test={y_test.mean():.3f}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    n_added = tokenizer.add_special_tokens(
        {"additional_special_tokens": MARKER_TOKENS}
    )
    for t in MARKER_TOKENS:
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        assert len(ids) == 1, f"marker {t} not a single token: {ids}"

    print(f"{tag} tokenizing (marker tokens added: {n_added})...", flush=True)
    ids_train, trunc_train = encode_split(tokenizer, train, args.max_len, args.no_text)
    ids_dev, trunc_dev = encode_split(tokenizer, dev, args.max_len, args.no_text)
    ids_test, trunc_test = encode_split(tokenizer, test, args.max_len, args.no_text)
    truncation = {
        "train": {"n": trunc_train, "frac": trunc_train / max(len(train), 1)},
        "dev": {"n": trunc_dev, "frac": trunc_dev / max(len(dev), 1)},
        "test": {"n": trunc_test, "frac": trunc_test / max(len(test), 1)},
    }
    print(f"{tag} entry-truncated@{args.max_len}: {json.dumps(truncation)}", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1
    )
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    def dev_select(probs: np.ndarray) -> tuple[float, float]:
        p = probs[:, 0]
        best_t, best_f1 = 0.5, -1.0
        for t in THRESHOLD_GRID:
            f1 = f1_score(y_dev, (p >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_t, best_f1 = float(t), float(f1)
        return best_t, best_f1

    best = train_model(
        model,
        EncodedDataset(ids_train, y_train.reshape(-1, 1)),
        ids_dev,
        dev_select,
        args=args,
        device=device,
        pad_id=tokenizer.pad_token_id,
        art_dir=art_dir,
        log_prefix=tag,
    )
    print(f"{tag} best: {json.dumps(best)}", flush=True)

    print(f"{tag} loading best checkpoint...", flush=True)
    model.load_state_dict(torch.load(art_dir / "checkpoint" / "model.pt",
                                     map_location=device))
    best_t = best["best_threshold"]

    test_metrics = None
    per_language = {}
    scores_test = y_pred = None
    if args.skip_test:
        print(f"{tag} --skip-test: sweep cell, test not evaluated", flush=True)
    else:
        print(f"{tag} scoring test...", flush=True)
        logits_test = score_all(model, ids_test, device, tokenizer.pad_token_id,
                                batch_size=args.eval_batch_size, log_prefix=f"{tag} [test]")
        scores_test = (1.0 / (1.0 + np.exp(-logits_test)))[:, 0]
        y_pred = (scores_test >= best_t).astype(int)

        test_metrics = evaluate(y_test, y_pred, scores_test)
        print(f"{tag} TEST :", json.dumps(test_metrics), flush=True)

        langs = sorted({r.get("language", "?") for r in test})
        if len(langs) > 1:
            for lg in langs:
                idx = np.array([i for i, r in enumerate(test) if r.get("language") == lg], dtype=int)
                per_language[lg] = evaluate(y_test[idx], y_pred[idx], scores_test[idx])
                print(f"{tag} TEST [{lg}]:", json.dumps(per_language[lg]), flush=True)

    tokenizer.save_pretrained(art_dir / "checkpoint")

    meta = {
        "task": "edge_prediction",
        "model": f"{args.model}_pair_suffix_markers",
        "negatives_tag": args.tag,
        "data_dir": str(data_dir),
        "seed": args.seed,
        "threshold": best_t,
        "threshold_grid": [float(t) for t in THRESHOLD_GRID],
        "dev_f1_at_threshold": best["best_dev_score"],
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
        "no_text": args.no_text,
        "input_format": (
            f"'{NO_TEXT_PLACEHOLDER}' " if args.no_text else "entry "
        ) + "</s></s> [N1] label (type, polarity, time) [/N1] "
            "</s> [N2] label (type, polarity, time) [/N2]",
        "marker_tokens": MARKER_TOKENS,
        "truncation": truncation,
        "train_entry_frac": args.train_entry_frac,
        "train_entries_full": n_entries_full,
        "train_size_full": n_train_full,
        "train_size": len(train),
        "dev_size": len(dev),
        "test_size": len(test),
        "smoke_test": args.smoke_test,
    }
    (art_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    metrics = {
        "task": "edge_prediction",
        "model": meta["model"],
        "negatives_tag": args.tag,
        "no_text": args.no_text,
        "dev": {"f1_at_best_threshold": best["best_dev_score"], "threshold": best_t},
        "test": test_metrics,
        "test_per_language": per_language,
    }
    (art_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if args.skip_test:
        print(f"{tag} artifacts written to {art_dir}", flush=True)
        return

    with (art_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r, gold, p, s in zip(test, y_test, y_pred, scores_test):
            f.write(json.dumps({
                "example_id": r["example_id"],
                "language": r.get("language"),
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
            }) + "\n")

    print(f"{tag} artifacts written to {art_dir}", flush=True)


if __name__ == "__main__":
    main()
