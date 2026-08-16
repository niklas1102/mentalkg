# Reproducing MentalKG

The `scripts/` directory is the reproduction interface. Every script also runs standalone; the shell wrappers exist so the exact flags used for the reported numbers are self-documenting.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data path A: use the released dataset (fast)

```bash
# node splits and metadata download directly from HF
huggingface-cli download --repo-type dataset Niklas1102/mentalkg \
    --local-dir modeling/data_v2
# edge splits are rebuilt from dataset.jsonl + splits.json
python scripts/build_edge_split.py --data-dir modeling/data_v2
```

## Data path B: regenerate the dataset (slow, needs OpenRouter credit)

```bash
export OPENROUTER_API_KEY=sk-or-v1-…
cp pipeline/config.example.json pipeline/config.json
python pipeline/run.py                                # ~$45 at gpt-4o-mini rates
python modeling/prepare_data.py --negatives hard --output modeling/data_v2
```

## Validate

```bash
python scripts/validate_dataset.py                    # PASS/FAIL table
python scripts/verify_splits.py                       # disjointness + stratification
python scripts/stats.py                               # corpus stats table
```

## Model path A: use the released checkpoints

```bash
python scripts/download_models.py                     # ~2 GB into models/
python scripts/eval_text.py "..."                     # smoke test
bash   scripts/run_server.sh                          # localhost:8000/extract
```

## Model path B: retrain from scratch

**TF-IDF baselines (CPU, minutes)**

```bash
bash scripts/reproduce_node_tfidf.sh
python modeling/baselines/edge_baseline.py --data-dir modeling/data_v2/edge_prediction --tag hard
python modeling/baselines/edge_shortcut.py
python modeling/baselines/string_match_floor.py
```

**XLM-R node checkpoint (CUDA GPU)**

Winner config: `--head-bias-init`, lr 3e-5, 9 epochs, plain BCE, seed 42.

```bash
bash scripts/reproduce_node_xlmr.sh                   # writes to artifacts_v2_bert/node_xlmr_all_rerun/
```

Without `--head-bias-init` the classifier collapses to the base-rate solution (0.187 micro-F1).

**XLM-R edge 3-seed sweep (CUDA GPU)**

```bash
bash scripts/reproduce_edge.sh                        # seeds 42/43/44 × {with-text, no-text}
```

**Llama-3.2-3B zero-shot (needs local ollama)**

```bash
ollama pull llama3.2:3b
bash scripts/reproduce_llama.sh
```

## Compare against the reported numbers

```bash
python modeling/baselines/make_results_v2.py         # rewrites results/RESULTS.md and results/metrics.json
```

The script overwrites both committed files in place. Run `git diff results/` to see whether your rerun reproduces the published numbers.

## Verification summary (what to expect)

| slice | metric | expected |
|---|---|---:|
| node XLM-R all | micro-F1 | 0.911 |
| node XLM-R all | macro-F1 | 0.849 |
| edge XLM-R hard, with text | F1 (3-seed mean) | 0.749 ± 0.001 |
| edge XLM-R hard, no text | F1 (3-seed mean) | 0.741 ± 0.001 |
| node TF-IDF | micro-F1 | 0.813 |
| node string-match floor | micro-F1 | 0.364 |
| node Llama-3.2-3B, ollama | micro-F1 | 0.279 |

Deltas ≤ 0.01 count as MATCHES.
