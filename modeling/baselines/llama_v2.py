"""Zero-shot Llama baselines for node and edge prediction (v2, HF transformers).

Successor to modeling/llama_baseline.py (v1, Ollama): same two tasks, but on
the v2 data (modeling/data_v2) with a Hugging Face backend so the identical
script runs locally and on a GPU pod.

  node - multi-label concept prediction from journal text. The prompt lists
         the full concept_id inventory from labels.json and asks for JSON
         {"present_concepts": [...]}. Greedy decoding (max_new_tokens 400),
         lenient JSON parse (direct, then first {...} block), out-of-
         inventory ids dropped and counted.
  edge - binary "are these two nodes connected?". Scored without generation:
         one forward pass over the chat prompt (which ends at the answer
         position), P(yes) = softmax mass of the "yes"-variant first tokens
         vs the union with the "no" variants. Default eval set: 4,000 pairs,
         1,000 per (language x connected) cell, sampled deterministically
         (seed 42); chosen ids are written to sample_ids.json.

The default model id is the gated meta-llama repo; HF_TOKEN is read from the
environment by huggingface_hub. If access is not approved, use the mirror
--model unsloth/Llama-3.2-3B-Instruct. Device: cuda > mps > cpu; fp16 on
cuda, fp32 otherwise.

Usage:
    python modeling/baselines/llama_v2.py node [--split test] [--limit N]
    python modeling/baselines/llama_v2.py edge [--per-cell 1000] [--limit N]

Results default to results/llama_v2/{node,edge}_prediction/.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from bert_utils import get_device, read_jsonl, set_seed

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
MIRROR_MODEL = "unsloth/Llama-3.2-3B-Instruct"
YES_VARIANTS = ("yes", "Yes", " yes", " Yes")
NO_VARIANTS = ("no", "No", " no", " No")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "modeling" / "data_v2"
RESULTS_ROOT = REPO_ROOT / "results" / "llama_v2"


# ----- Model loading -------------------------------------------------------

def resolve_device(arg: str) -> torch.device:
    return get_device() if arg == "auto" else torch.device(arg)


def load_model(model_id: str, device: torch.device):
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except Exception as e:  # gated repo / missing token gets a clear message
        msg = str(e)
        if any(s in msg.lower() for s in ("gated", "401", "403", "authoriz")):
            raise SystemExit(
                f"Cannot download {model_id!r} ({msg.splitlines()[0]}).\n"
                "Set HF_TOKEN in the environment (repo access required) or "
                f"run with --model {MIRROR_MODEL}"
            )
        raise
    model = model.to(device)
    model.eval()
    return tokenizer, model, str(dtype).replace("torch.", "")


def chat_ids(tokenizer, prompt: str) -> torch.Tensor:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    )


# ----- Prompts (kept close to v1 for comparability) ------------------------

def build_node_prompt(text: str, label_vocab: list[str]) -> str:
    label_block = "\n".join(f"- {l}" for l in label_vocab)
    return f"""You are analyzing a short personal journal entry.

Task: identify which mental-health concepts from the controlled list below are clearly expressed in the entry.

Rules:
- Only include concepts that are clearly present in the text.
- Use the concept_id EXACTLY as listed (snake_case, e.g. emotion_anxiety).
- Do NOT invent new concepts; only use IDs from the list.
- It is fine to return many concepts or just a few. Most entries have 3-8 concepts.

Allowed concept_ids:
{label_block}

Journal entry:
\"\"\"
{text}
\"\"\"

Output JSON ONLY in this exact shape:
{{"present_concepts": ["concept_id_1", "concept_id_2", ...]}}"""


def node_desc(node: dict) -> str:
    ta = (node.get("time_anchor") or {}).get("text") or "unspecified"
    return (f"\"{node['label']}\" (type: {node['type']}, "
            f"polarity: {node['polarity']}, time: {ta})")


def build_edge_prompt(text: str, node_a: dict, node_b: dict) -> str:
    return f"""You are analyzing a short personal journal entry.

Task: decide whether the journal entry describes a CONNECTION between two specific concepts. A connection means the entry implies one causes / increases / decreases / follows / is linked to the other (any direction).

Concept A: {node_desc(node_a)}
Concept B: {node_desc(node_b)}

Both concepts may appear in the entry independently — that alone is NOT enough. There must be a relational link between them in the text.

Journal entry:
\"\"\"
{text}
\"\"\"

Are these two connected in the entry? Answer yes or no."""


# ----- Parsing -------------------------------------------------------------

def parse_json_lenient(text: str):
    """Parse JSON; if strict parse fails, extract the first {...} block."""
    text = text.strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0)), None
        except json.JSONDecodeError as e:
            return None, f"could_not_parse: {e.msg}"
    return None, "no_json_object_found"


def parse_node_response(raw: str, label_vocab_set: set):
    """Returns (predicted_in_vocab, invalid, parse_error)."""
    obj, err = parse_json_lenient(raw)
    if err is not None:
        return [], [], err
    if not isinstance(obj, dict) or "present_concepts" not in obj:
        return [], [], "missing_key_present_concepts"
    raw_list = obj["present_concepts"]
    if not isinstance(raw_list, list):
        return [], [], "present_concepts_not_list"
    in_vocab, invalid = [], []
    for item in raw_list:
        if isinstance(item, str) and item in label_vocab_set:
            in_vocab.append(item)
        else:
            invalid.append(str(item))
    return sorted(set(in_vocab)), invalid, None


# ----- Metrics (same shapes as node_baseline / edge_bert) ------------------

def node_metrics(gold_sets: list[set], pred_sets: list[set],
                 vocab: list[str]) -> dict:
    idx = {c: i for i, c in enumerate(vocab)}
    Y = np.zeros((len(gold_sets), len(vocab)), dtype=int)
    P = np.zeros_like(Y)
    for i, (g, p) in enumerate(zip(gold_sets, pred_sets)):
        for c in g:
            Y[i, idx[c]] = 1
        for c in p:
            P[i, idx[c]] = 1
    return {
        "micro_f1": float(f1_score(Y, P, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(Y, P, average="macro", zero_division=0)),
        "micro_precision": float(
            precision_score(Y, P, average="micro", zero_division=0)
        ),
        "micro_recall": float(
            recall_score(Y, P, average="micro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(Y, P, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(Y, P, average="macro", zero_division=0)
        ),
        "n_examples": int(len(gold_sets)),
    }


def edge_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                 y_score: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:  # single-class subset (e.g. tiny --limit)
        auc = None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "n_pos_pred": int(y_pred.sum()),
        "n_pos_true": int(y_true.sum()),
        "n_total": int(len(y_true)),
    }


# ----- node subcommand -----------------------------------------------------

def run_node(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "[llama_v2/node]"
    print(f"{tag} device={device} model={args.model}", flush=True)

    records = read_jsonl(data_dir / f"{args.split}.jsonl")
    if args.limit is not None:
        records = records[: args.limit]
    labels = json.loads((data_dir / "labels.json").read_text())
    vocab = labels["index_to_label"]
    vocab_set = set(vocab)
    print(f"{tag} {len(records)} examples ({args.split}), "
          f"{len(vocab)} concept ids", flush=True)

    # Incremental + resumable: each finished example is appended to
    # predictions.jsonl immediately; on restart, rows already present are kept
    # and their example_ids skipped (a torn last line from an interrupted run
    # is discarded). Final metrics are computed over the complete file.
    pred_path = out_dir / "predictions.jsonl"
    prev_rows: list[dict] = []
    if pred_path.exists():
        for line in pred_path.read_text(encoding="utf-8").splitlines():
            try:
                prev_rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
        with pred_path.open("w", encoding="utf-8") as f:
            for row in prev_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if prev_rows:
            done_ids = {row["example_id"] for row in prev_rows}
            records = [r for r in records if r["example_id"] not in done_ids]
            print(f"{tag} resume: {len(prev_rows)} examples already done, "
                  f"{len(records)} to go", flush=True)

    if args.backend == "ollama":
        # Mirrors v1 (modeling/llama_baseline.py::call_llm): raw prompt, JSON
        # mode, temperature 0. Constrained decoding terminates the JSON, so
        # runaway-enumeration parse failures of the HF greedy path do not occur.
        import ollama
        tokenizer = model = None
        dtype = "ollama_json"
        print(f"{tag} backend=ollama model={args.ollama_model}", flush=True)
    else:
        tokenizer, model, dtype = load_model(args.model, device)
    t0 = time.time()
    pred_file = pred_path.open("a", encoding="utf-8")
    for i, r in enumerate(records):
        if args.backend == "ollama":
            raw = ""
            for attempt in (1, 2):
                try:
                    resp = ollama.generate(
                        model=args.ollama_model,
                        prompt=build_node_prompt(r["text"], vocab),
                        format="json",
                        options={"temperature": 0.0,
                                 "num_predict": args.max_new_tokens},
                    )
                    raw = resp.get("response", "")
                    break
                except Exception as exc:
                    if attempt == 2:
                        raw = ""
                        print(f"{tag} ollama error on {r['example_id']}: "
                              f"{exc}", flush=True)
                    else:
                        time.sleep(2)
        else:
            ids = chat_ids(tokenizer, build_node_prompt(r["text"], vocab)).to(device)
            with torch.no_grad():
                out = model.generate(
                    ids,
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
            raw = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        pred, invalid, err = parse_node_response(raw, vocab_set)
        pred_file.write(json.dumps({
            "example_id": r["example_id"],
            "language": r.get("language"),
            "gold": sorted(r["labels"]),
            "pred": pred,
            "invalid": invalid,
            "parse_error": err,
        }, ensure_ascii=False) + "\n")
        pred_file.flush()
        if (i + 1) % 50 == 0 or i + 1 == len(records):
            print(f"{tag} {i + 1}/{len(records)} "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)
    pred_file.close()
    wall = time.time() - t0

    # Metrics over the complete file: earlier passes' rows plus this one's.
    all_rows = [json.loads(line)
                for line in pred_path.read_text(encoding="utf-8").splitlines()]
    gold_sets = [set(row["gold"]) for row in all_rows]
    pred_sets = [set(row["pred"]) for row in all_rows]
    parse_errors = sum(1 for row in all_rows if row["parse_error"] is not None)
    invalid_total = sum(len(row["invalid"]) for row in all_rows)

    overall = node_metrics(gold_sets, pred_sets, vocab)
    per_language = {}
    for lg in sorted({row.get("language", "?") for row in all_rows}):
        sel = [i for i, row in enumerate(all_rows) if row.get("language") == lg]
        per_language[lg] = node_metrics(
            [gold_sets[i] for i in sel], [pred_sets[i] for i in sel], vocab
        )
    print(f"{tag} OVERALL:", json.dumps(overall), flush=True)
    for lg, m in per_language.items():
        print(f"{tag} [{lg}]:", json.dumps(m), flush=True)

    metrics = {
        "task": "node_prediction",
        "model": args.ollama_model if args.backend == "ollama" else args.model,
        "backend": ("ollama_json_constrained" if args.backend == "ollama"
                    else "transformers_greedy"),
        "split": args.split,
        "limit": args.limit,
        "seed": args.seed,
        "device": str(device),
        "dtype": dtype,
        "max_new_tokens": args.max_new_tokens,
        "n_examples": len(all_rows),
        "resumed_examples": len(prev_rows),
        "parse_errors": parse_errors,
        "invalid_ids_dropped": invalid_total,
        "wall_time_s": round(wall, 1),
        "overall": overall,
        "per_language": per_language,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"{tag} results written to {out_dir}", flush=True)


# ----- edge subcommand -----------------------------------------------------

def first_token_ids(tokenizer, variants: tuple) -> list[int]:
    ids = set()
    for v in variants:
        toks = tokenizer(v, add_special_tokens=False)["input_ids"]
        if toks:
            ids.add(toks[0])
    return sorted(ids)


def subsample_edges(records: list[dict], per_cell: int, seed: int) -> set:
    by_cell: dict[tuple, list[str]] = {}
    for r in records:
        key = (r["language"], bool(r["connected"]))
        by_cell.setdefault(key, []).append(r["example_id"])
    chosen = set()
    for key in sorted(by_cell):
        ids = sorted(by_cell[key])
        rng = random.Random(f"{seed}:{key[0]}:{key[1]}")
        chosen.update(rng.sample(ids, min(per_cell, len(ids))))
    return chosen


def run_edge(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = resolve_device(args.device)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "[llama_v2/edge]"
    print(f"{tag} device={device} model={args.model}", flush=True)

    records = read_jsonl(data_dir / f"{args.split}.jsonl")
    n_full = len(records)
    if args.per_cell > 0:
        keep = subsample_edges(records, args.per_cell, args.seed)
        records = [r for r in records if r["example_id"] in keep]
        (out_dir / "sample_ids.json").write_text(
            json.dumps(sorted(keep), indent=2)
        )
        print(f"{tag} subsample: {len(records)}/{n_full} pairs "
              f"({args.per_cell} per language x connected cell)", flush=True)
    if args.limit is not None:
        records = records[: args.limit]

    tokenizer, model, dtype = load_model(args.model, device)
    yes_ids = first_token_ids(tokenizer, YES_VARIANTS)
    no_ids = first_token_ids(tokenizer, NO_VARIANTS)
    assert yes_ids and no_ids and not set(yes_ids) & set(no_ids), \
        f"bad yes/no token ids: {yes_ids} vs {no_ids}"
    print(f"{tag} yes_ids={yes_ids} no_ids={no_ids}", flush=True)

    t0 = time.time()
    scores = np.zeros(len(records))
    for i, r in enumerate(records):
        ids = chat_ids(
            tokenizer, build_edge_prompt(r["text"], r["node_a"], r["node_b"])
        ).to(device)
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        p_yes = float(probs[yes_ids].sum())
        p_no = float(probs[no_ids].sum())
        scores[i] = p_yes / (p_yes + p_no) if p_yes + p_no > 0 else 0.5
        if (i + 1) % 50 == 0 or i + 1 == len(records):
            print(f"{tag} {i + 1}/{len(records)} "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)
    wall = time.time() - t0

    y_true = np.array([1 if r["connected"] else 0 for r in records], dtype=int)
    y_pred = (scores >= 0.5).astype(int)
    overall = edge_metrics(y_true, y_pred, scores)
    per_language = {}
    for lg in sorted({r.get("language", "?") for r in records}):
        idx = np.array(
            [i for i, r in enumerate(records) if r.get("language") == lg],
            dtype=int,
        )
        per_language[lg] = edge_metrics(y_true[idx], y_pred[idx], scores[idx])
    print(f"{tag} OVERALL:", json.dumps(overall), flush=True)
    for lg, m in per_language.items():
        print(f"{tag} [{lg}]:", json.dumps(m), flush=True)

    metrics = {
        "task": "edge_prediction",
        "model": args.model,
        "backend": "transformers_yes_no_scoring",
        "split": args.split,
        "per_cell": args.per_cell,
        "limit": args.limit,
        "seed": args.seed,
        "device": str(device),
        "dtype": dtype,
        "yes_token_ids": yes_ids,
        "no_token_ids": no_ids,
        "n_examples": len(records),
        "n_split_full": n_full,
        "wall_time_s": round(wall, 1),
        "overall": overall,
        "per_language": per_language,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r, gold, p, s in zip(records, y_true, y_pred, scores):
            f.write(json.dumps({
                "example_id": r["example_id"],
                "language": r.get("language"),
                "gold": int(gold),
                "pred": int(p),
                "p_yes": float(s),
            }) + "\n")
    print(f"{tag} results written to {out_dir}", flush=True)


# ----- CLI -----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"HF model id (mirror: {MIRROR_MODEL})")
    common.add_argument("--split", default="test")
    common.add_argument("--limit", type=int, default=None)
    common.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"),
                        default="auto")
    common.add_argument("--seed", type=int, default=42)

    p_node = sub.add_parser("node", parents=[common],
                            help="multi-label concept prediction")
    p_node.add_argument("--data-dir",
                        default=str(DATA_ROOT / "node_prediction"))
    p_node.add_argument("--out-dir",
                        default=str(RESULTS_ROOT / "node_prediction"))
    p_node.add_argument("--max-new-tokens", type=int, default=400)
    p_node.add_argument("--backend", choices=("hf", "ollama"), default="hf",
                        help="ollama mirrors v1: raw prompt, format=json, temp 0")
    p_node.add_argument("--ollama-model", default="llama3.2:3b")
    p_node.set_defaults(func=run_node)

    p_edge = sub.add_parser("edge", parents=[common],
                            help="binary connected-pair prediction")
    p_edge.add_argument("--data-dir",
                        default=str(DATA_ROOT / "edge_prediction"))
    p_edge.add_argument("--out-dir",
                        default=str(RESULTS_ROOT / "edge_prediction"))
    p_edge.add_argument("--per-cell", type=int, default=1000,
                        help="pairs per (language x connected) cell; 0 = full split")
    p_edge.set_defaults(func=run_edge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
