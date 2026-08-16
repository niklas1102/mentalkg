# MentalKG

Bilingual (EN/DE) mental-health journal to knowledge-graph pipeline. Generates a synthetic corpus, trains XLM-R node and edge extractors, serves them via a local HTTP endpoint, and ships an Obsidian plugin that renders the extracted graph.

## What is in this repo

| dir | contents |
|---|---|
| `data/` | concept vocabulary (`variable_pool.json`), dataset schema doc |
| `pipeline/` | dataset generator (`run.py`, `scenarios.py`, `llm.py`, `graph_stats.py`, `config.example.json`) |
| `modeling/` | data prep (`prepare_data.py`), baselines (TF-IDF, XLM-R, Llama, string-match floor) |
| `results/` | final measured metrics table (`RESULTS.md`, `metrics.json`) |
| `models/` | landing directory for downloaded HF checkpoints (populated by a script) |
| `server/` | FastAPI extractor that serves the two XLM-R checkpoints |
| `plugin/` | Obsidian plugin source + demo vault |
| `scripts/` | 12 wrappers: validate, verify, reproduce, download, eval, run |
| `docs/` | `REPRODUCING.md`, `MODELS.md`, `DATASET.md` |

## Quickstart

```bash
pip install -r requirements.txt
python scripts/download_models.py                                  # ~2 GB, one-time
python scripts/eval_text.py "I could not sleep and the deadline is tomorrow."
bash scripts/run_server.sh                                          # localhost:8000/extract
```

For a graph rendered in Obsidian, build the plugin and open the demo vault:

```bash
bash scripts/build_plugin.sh
# then open plugin/demo-vault/ in Obsidian, enable "Journal Graph" plugin,
# open a journal entry, and use the Extract command.
```

## Dataset

47,714 samples across 3,410 participants (14 days each). 41,315 accepted after LLM-based verification. Bilingual: 24,936 EN / 22,778 DE (all); 20,982 EN / 20,333 DE (accepted).

| slice | rows |
|---|---:|
| dataset.jsonl (accept + review + reject) | 47,714 |
| accepted | 41,315 |
| node splits (train / dev / test) | 20,679 / 12,390 / 8,246 |
| edge splits (train / dev / test) | 261,110 / 156,064 / 103,798 |

Hosted on Hugging Face: [`Niklas1102/mentalkg`](https://huggingface.co/datasets/Niklas1102/mentalkg). Full schema in [`data/README.md`](data/README.md) and [`docs/DATASET.md`](docs/DATASET.md).

## Results

Measured on the 8,246-example / 103,798-pair test splits (participant-independent 50/30/20, stratified by language, seed 42).

**Node prediction, micro/macro-F1, 130-label vocab**

| model | micro-F1 | macro-F1 |
|---|---:|---:|
| XLM-R combined EN+DE (this repo) | **0.911** | **0.849** |
| TF-IDF + OvR logreg | 0.813 | 0.801 |
| Llama-3.2-3B zero-shot (ollama, JSON) | 0.279 | 0.242 |
| string-match floor | 0.364 | 0.289 |

**Edge prediction, binary connected, hard negatives, 3-seed F1**

| arm | F1 (mean ± range) |
|---|---|
| XLM-R with entry text (this repo) | **0.7488 [0.7483, 0.7492]** |
| XLM-R no-text ablation | 0.7408 [0.7407, 0.7410] |
| TF-IDF + pair features + text | 0.7125 |
| Llama-3.2-3B zero-shot (4k pairs) | 0.667 |

Full tables and provenance: [`results/RESULTS.md`](results/RESULTS.md).

## Reproduction

| what | script | needs |
|---|---|---|
| dataset sanity | `scripts/validate_dataset.py` | CPU |
| split checks | `scripts/verify_splits.py` | CPU |
| inspect one participant-day | `scripts/inspect_sample.py` | CPU |
| corpus stats | `scripts/stats.py` | CPU |
| download models | `scripts/download_models.py` | network |
| extract one text | `scripts/eval_text.py` | CPU or GPU |
| run server | `scripts/run_server.sh` | CPU or GPU |
| build plugin | `scripts/build_plugin.sh` | npm |
| rebuild edge splits | `scripts/build_edge_split.py` | CPU |
| TF-IDF baselines | `scripts/reproduce_node_tfidf.sh` | CPU |
| XLM-R node retrain | `scripts/reproduce_node_xlmr.sh` | CUDA GPU |
| XLM-R edge 3-seed sweep | `scripts/reproduce_edge.sh` | CUDA GPU |
| Llama zero-shot | `scripts/reproduce_llama.sh` | ollama + local GPU |

## Models

- Node: [`Niklas1102/mentalkg-xlmr-node`](https://huggingface.co/Niklas1102/mentalkg-xlmr-node)
- Edge: [`Niklas1102/mentalkg-xlmr-edge`](https://huggingface.co/Niklas1102/mentalkg-xlmr-edge)

## License

MIT for code and model weights. CC BY 4.0 for the dataset.
