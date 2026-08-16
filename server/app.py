"""FastAPI server for live graph extraction from the trained XLM-R checkpoints.

Two models load from models/mentalkg-xlmr-node and models/mentalkg-xlmr-edge:
node predicts concept ids (multi-label sigmoid, threshold from meta.json),
edge predicts connectivity over all candidate pairs with [N1]/[N2] marker
tokens, scores each ordering and averages. Edge relation TYPES come from a
majority lookup over gold-graph type-pair statistics bundled with the edge
model as relation_map.json; each such edge carries "source": "heuristic_type".

Environment overrides: NODE_CKPT_DIR, EDGE_CKPT_DIR, NODE_META, EDGE_META,
SERVER_DEVICE.

Run:
    cd server
    uvicorn app:app --port 8000
"""
from __future__ import annotations

import itertools
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = REPO_ROOT / "data" / "variable_pool.json"
DATASET_PATH = REPO_ROOT / "pipeline" / "output" / "dataset.jsonl"
MODELS_ROOT = REPO_ROOT / "models"

NODE_CKPT_DIR = Path(os.environ.get(
    "NODE_CKPT_DIR", MODELS_ROOT / "mentalkg-xlmr-node"))
EDGE_CKPT_DIR = Path(os.environ.get(
    "EDGE_CKPT_DIR", MODELS_ROOT / "mentalkg-xlmr-edge"))
# Thresholds and per-split metrics live next to the weights inside each model dir.
NODE_META = Path(os.environ.get("NODE_META", NODE_CKPT_DIR / "meta.json"))
EDGE_META = Path(os.environ.get("EDGE_META", EDGE_CKPT_DIR / "meta.json"))

MAX_LEN = 256           # matches both training runs
DEFAULT_RELATION = "linked_to"
FALLBACK_TOP_K = 3      # if nothing clears the node threshold


def _pick_device() -> torch.device:
    name = os.environ.get("SERVER_DEVICE")
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _build_relation_map(records):
    """Per-ordered-type-pair majority relation (verbatim logic from server v1)."""
    ordered: dict[tuple, Counter] = defaultdict(Counter)
    for r in records:
        graph = r.get("graph") or {}
        type_by = {n["node_id"]: n["type"] for n in graph.get("nodes") or []}
        for e in graph.get("edges") or []:
            ts, td = type_by.get(e.get("source_node_id")), type_by.get(e.get("target_node_id"))
            if ts and td:
                ordered[(ts, td)][e.get("type") or DEFAULT_RELATION] += 1
    rel = {p: c.most_common(1)[0][0] for p, c in ordered.items()}
    cnt = {p: c.most_common(1)[0][1] for p, c in ordered.items()}
    return rel, cnt


def _node_suffix(marker: str, node: dict) -> str:
    """Verbatim format from edge_bert.py::node_suffix."""
    ta = (node.get("time_anchor") or {}).get("text") or "unknown"
    return f"[{marker}] {node['label']} ({node['type']}, {node['polarity']}, {ta}) [/{marker}]"


class _Models:
    """Lazy holder: everything loads on the first /extract call."""

    def __init__(self):
        self.ready = False

    def load(self):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = _pick_device()
        print(f"[server] device={self.device}")

        node_meta = json.loads(NODE_META.read_text())
        self.node_threshold = float(node_meta["threshold"])
        edge_meta = json.loads(EDGE_META.read_text())
        self.edge_threshold = float(edge_meta["threshold"])

        pool = json.loads(POOL_PATH.read_text())
        self.concepts = {c["concept_id"]: c for c in pool["node_concepts"]}

        self.label_vocab = json.loads((NODE_CKPT_DIR / "labels.json").read_text())["index_to_label"]

        print(f"[server] loading node checkpoint from {NODE_CKPT_DIR} ...")
        self.node_tok = AutoTokenizer.from_pretrained(str(NODE_CKPT_DIR))
        self.node_model = AutoModelForSequenceClassification.from_pretrained(str(NODE_CKPT_DIR))
        self.node_model.to(self.device).eval()

        print(f"[server] loading edge checkpoint from {EDGE_CKPT_DIR} ...")
        self.edge_tok = AutoTokenizer.from_pretrained(str(EDGE_CKPT_DIR))
        self.edge_model = AutoModelForSequenceClassification.from_pretrained(str(EDGE_CKPT_DIR))
        self.edge_model.to(self.device).eval()

        # Relation map: prefer the precomputed table bundled with the edge model.
        # Fall back to building it from a local dataset.jsonl (useful when the
        # local pipeline has been re-run and the relation stats should update).
        rel_path = EDGE_CKPT_DIR / "relation_map.json"
        if rel_path.exists():
            data = json.loads(rel_path.read_text())
            self.relation_map = {tuple(k.split("|", 1)): v for k, v in data["relation"].items()}
            self.relation_count = {tuple(k.split("|", 1)): v for k, v in data["count"].items()}
            print(f"[server] loaded {len(self.relation_map)} type-pairs from {rel_path}")
        elif DATASET_PATH.exists():
            print("[server] building relation map from local dataset.jsonl ...")
            self.relation_map, self.relation_count = _build_relation_map(
                _read_jsonl(DATASET_PATH))
        else:
            raise RuntimeError(
                f"no relation map: neither {rel_path} nor {DATASET_PATH} exists"
            )
        self.ready = True
        print(f"[server] ready: node_threshold={self.node_threshold}, "
              f"edge_threshold={self.edge_threshold}, "
              f"{len(self.relation_map)} type-pairs in relation map")

    # ---- node ----
    @torch.no_grad()
    def predict_concepts(self, text: str) -> list[tuple[str, float]]:
        enc = self.node_tok(text, truncation=True, max_length=MAX_LEN,
                            return_tensors="pt").to(self.device)
        logits = self.node_model(**enc).logits[0].float().cpu()
        probs = torch.sigmoid(logits)
        pairs = [(self.label_vocab[i], float(probs[i])) for i in range(len(self.label_vocab))]
        above = sorted([p for p in pairs if p[1] >= self.node_threshold],
                       key=lambda x: -x[1])
        if above:
            return above
        return sorted(pairs, key=lambda x: -x[1])[:FALLBACK_TOP_K]

    # ---- edge ----
    def _encode_pair(self, text: str, node_a: dict, node_b: dict) -> list[int]:
        """Verbatim input format from edge_bert.py::encode_split."""
        tok = self.edge_tok
        bos, sep = tok.bos_token_id, tok.sep_token_id
        suffix = (tok(_node_suffix("N1", node_a), add_special_tokens=False)["input_ids"]
                  + [sep]
                  + tok(_node_suffix("N2", node_b), add_special_tokens=False)["input_ids"])
        body = tok(text, add_special_tokens=False)["input_ids"]
        budget = MAX_LEN - len(suffix) - 4
        if len(body) > budget:
            body = body[:budget]
        return [bos] + body + [sep, sep] + suffix + [sep]

    @torch.no_grad()
    def score_pairs(self, text: str, nodes: list[dict]) -> list[tuple[int, int, float]]:
        if len(nodes) < 2:
            return []
        pairs = list(itertools.combinations(range(len(nodes)), 2))
        ids_list = []
        for i, j in pairs:  # both orderings, averaged for symmetry
            ids_list.append(self._encode_pair(text, nodes[i], nodes[j]))
            ids_list.append(self._encode_pair(text, nodes[j], nodes[i]))
        pad = self.edge_tok.pad_token_id
        max_len = max(len(x) for x in ids_list)
        input_ids = torch.full((len(ids_list), max_len), pad, dtype=torch.long)
        mask = torch.zeros((len(ids_list), max_len), dtype=torch.long)
        for k, ids in enumerate(ids_list):
            input_ids[k, :len(ids)] = torch.tensor(ids)
            mask[k, :len(ids)] = 1
        logits = self.edge_model(
            input_ids=input_ids.to(self.device),
            attention_mask=mask.to(self.device)).logits[:, 0].float().cpu()
        probs = torch.sigmoid(logits)
        out = []
        for k, (i, j) in enumerate(pairs):
            p = float((probs[2 * k] + probs[2 * k + 1]) / 2.0)
            out.append((i, j, p))
        return out

    def pick_relation(self, t_a: str, t_b: str):
        fwd = self.relation_count.get((t_a, t_b), 0)
        rev = self.relation_count.get((t_b, t_a), 0)
        if fwd == 0 and rev == 0:
            return t_a, t_b, DEFAULT_RELATION
        if rev > fwd:
            return t_b, t_a, self.relation_map[(t_b, t_a)]
        return t_a, t_b, self.relation_map[(t_a, t_b)]


MODELS = _Models()

app = FastAPI(title="MentalKG graph extractor", version="2.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"],
)


class ExtractRequest(BaseModel):
    text: str


def _concept_meta(m: _Models, concept_id: str) -> dict:
    c = m.concepts.get(concept_id)
    if c is None:
        return {"type": "unknown", "label": concept_id, "polarity": "neutral"}
    return {"type": c["type"], "label": c["label"],
            "polarity": c.get("polarity", "neutral")}


@app.post("/extract")
def extract(req: ExtractRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")
    if not MODELS.ready:
        MODELS.load()

    predicted = MODELS.predict_concepts(text)
    nodes = []
    for i, (cid, score) in enumerate(predicted):
        meta = _concept_meta(MODELS, cid)
        nodes.append({
            "node_id": f"n{i + 1}",
            "concept_id": cid,
            "type": meta["type"],
            "label": meta["label"],
            "polarity": meta["polarity"],
            "time_anchor": {"text": "now", "day_offset": 0},
            "temporal_status": "current",
            "confidence": float(score),
            "source_entry_id": "live",
        })

    edges = []
    e_idx = 0
    for i, j, p in MODELS.score_pairs(text, nodes):
        if p < MODELS.edge_threshold:
            continue
        a, b = nodes[i], nodes[j]
        src_t, dst_t, rel = MODELS.pick_relation(a["type"], b["type"])
        src, dst = (a, b) if (src_t, dst_t) == (a["type"], b["type"]) else (b, a)
        e_idx += 1
        edges.append({
            "edge_id": f"e{e_idx}",
            "source_node_id": src["node_id"],
            "target_node_id": dst["node_id"],
            "type": rel,
            "weight": float(p),
            "confidence": float(p),
            "source": "heuristic_type",
        })

    return {
        "nodes": nodes,
        "edges": edges,
        "predicted_concepts": [
            {"concept_id": cid, "score": score} for cid, score in predicted
        ],
        "entry_text": text,
        "node_threshold": MODELS.node_threshold,
        "edge_threshold": MODELS.edge_threshold,
        "note": (
            "Node concepts and edge connectivity come from fine-tuned XLM-R "
            "models. Edge relation types come from a majority lookup over "
            "gold-graph type-pair statistics, and those edges carry "
            "source='heuristic_type'."
        ),
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "models_loaded": MODELS.ready,
        "node_ckpt": str(NODE_CKPT_DIR),
        "edge_ckpt": str(EDGE_CKPT_DIR),
    }


@app.get("/")
def root():
    return {
        "service": "mentalkg graph extractor",
        "endpoint": "POST /extract {text}",
        "learned": ["node concepts (XLM-R multi-label)",
                    "edge connectivity (XLM-R pair-aware)"],
        "heuristic": ["edge relation types (gold-graph majority lookup)",
                      "time anchors (placeholder 'now')"],
    }
