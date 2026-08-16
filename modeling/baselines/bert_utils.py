"""Shared utilities for the XLM-R transformer baselines (node_bert / edge_bert).

Pre-tokenized examples, dynamic padding, a generic fine-tuning loop with
per-epoch dev evaluation and best-checkpoint selection, and length-sorted
batched scoring. Device: cuda > mps > cpu; fp32 (MPS half-precision support
for XLM-R is unreliable).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # seeds CPU, CUDA and MPS generators


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EncodedDataset(Dataset):
    """Pre-tokenized examples: parallel lists of token ids and target vectors."""

    def __init__(self, ids_list: list[list[int]], targets: np.ndarray):
        assert len(ids_list) == len(targets)
        self.ids_list = ids_list
        self.targets = targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.ids_list)

    def __getitem__(self, i: int):
        return self.ids_list[i], self.targets[i]


def make_collate(pad_id: int):
    def collate(batch):
        ids_list, targets = zip(*batch)
        max_len = max(len(ids) for ids in ids_list)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, ids in enumerate(ids_list):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
        return input_ids, attention_mask, torch.tensor(np.stack(targets))

    return collate


@torch.no_grad()
def score_all(
    model,
    ids_list: list[list[int]],
    device: torch.device,
    pad_id: int,
    batch_size: int = 64,
    log_prefix: str | None = None,
) -> np.ndarray:
    """Return logits (N, C) in original order; batches sorted by length."""
    model.eval()
    order = sorted(range(len(ids_list)), key=lambda i: len(ids_list[i]))
    out = [None] * len(ids_list)
    t0 = time.time()
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        max_len = max(len(ids_list[i]) for i in chunk)
        input_ids = torch.full((len(chunk), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(chunk), max_len), dtype=torch.long)
        for j, i in enumerate(chunk):
            ids = ids_list[i]
            input_ids[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[j, : len(ids)] = 1
        logits = model(
            input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)
        ).logits
        logits = logits.float().cpu().numpy()
        for j, i in enumerate(chunk):
            out[i] = logits[j]
        if log_prefix and (start // batch_size) % 200 == 0:
            done = min(start + batch_size, len(order))
            print(
                f"{log_prefix} scored {done}/{len(order)} "
                f"({done / max(time.time() - t0, 1e-9):.0f} ex/s)",
                flush=True,
            )
    return np.stack(out)


def build_optimizer(model, lr: float, weight_decay: float):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(groups, lr=lr)


def train_model(
    model,
    train_ds: EncodedDataset,
    dev_ids: list[list[int]],
    dev_select_fn,
    *,
    args,
    device: torch.device,
    pad_id: int,
    art_dir: Path,
    log_prefix: str,
    loss_fn=None,
) -> dict:
    """Fine-tune with BCE loss; keep the checkpoint with the best dev score.

    dev_select_fn(probs_dev) -> (threshold, score); higher score = better.
    loss_fn overrides the default BCEWithLogitsLoss (e.g. with pos_weight).
    Returns {"best_epoch", "best_threshold", "best_dev_score"}; the best
    state_dict is at art_dir/checkpoint/model.pt.
    """
    ckpt_dir = art_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_log = (art_dir / "train_log.jsonl").open("w", encoding="utf-8")

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=make_collate(pad_id),
        num_workers=0,
    )
    steps_per_epoch = len(loader)
    optim_steps = -(-steps_per_epoch // args.grad_accum) * args.epochs
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup_ratio * optim_steps), optim_steps
    )
    if loss_fn is None:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    best = {"best_epoch": -1, "best_threshold": 0.5, "best_dev_score": -1.0}
    global_step = 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        running_loss, running_n = 0.0, 0
        t_epoch = time.time()
        for step, (input_ids, attention_mask, targets) in enumerate(loader, start=1):
            logits = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            ).logits
            loss = loss_fn(logits, targets.to(device)) / args.grad_accum
            loss.backward()
            running_loss += loss.item() * args.grad_accum
            running_n += 1
            if step % args.grad_accum == 0 or step == steps_per_epoch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            if step % 100 == 0:
                rate = step / (time.time() - t_epoch)
                rec = {
                    "epoch": epoch,
                    "step": step,
                    "of": steps_per_epoch,
                    "loss": round(running_loss / running_n, 5),
                    "lr": scheduler.get_last_lr()[0],
                    "steps_per_sec": round(rate, 3),
                    "elapsed_min": round((time.time() - t_start) / 60, 1),
                }
                print(f"{log_prefix} {json.dumps(rec)}", flush=True)
                train_log.write(json.dumps(rec) + "\n")
                train_log.flush()
                running_loss, running_n = 0.0, 0

        print(f"{log_prefix} epoch {epoch} done, scoring dev...", flush=True)
        probs_dev = 1.0 / (
            1.0
            + np.exp(
                -score_all(
                    model, dev_ids, device, pad_id,
                    batch_size=args.eval_batch_size,
                    log_prefix=f"{log_prefix} [dev e{epoch}]",
                )
            )
        )
        threshold, dev_score = dev_select_fn(probs_dev)
        rec = {
            "epoch": epoch,
            "dev_score": round(float(dev_score), 5),
            "dev_threshold": round(float(threshold), 2),
        }
        print(f"{log_prefix} {json.dumps(rec)}", flush=True)
        train_log.write(json.dumps(rec) + "\n")
        train_log.flush()
        if dev_score > best["best_dev_score"]:
            best = {
                "best_epoch": epoch,
                "best_threshold": float(threshold),
                "best_dev_score": float(dev_score),
            }
            torch.save(model.state_dict(), ckpt_dir / "model.pt")
            print(f"{log_prefix} new best (epoch {epoch}), checkpoint saved", flush=True)

    train_log.close()
    return best
