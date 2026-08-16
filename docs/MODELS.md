# Models

Two fine-tuned XLM-RoBERTa checkpoints, hosted on Hugging Face and downloaded on demand via `python scripts/download_models.py`.

## `mentalkg-xlmr-node`

- **base**: `xlm-roberta-base` (`FacebookAI/xlm-roberta-base`)
- **head**: 130-way sigmoid, multi-label
- **loss**: binary cross entropy with per-label prior-logit bias init
- **input**: raw journal text, right-truncated at 256 tokens
- **training**: lr 3e-5, 9 epochs, seed 42, batch size 32, single RTX 4090
- **dev-tuned threshold**: 0.28 (single global value)
- **test slice**: micro-F1 0.911, macro-F1 0.849, precision 0.920, recall 0.902 (n=8,246)
- **files**: `config.json`, `model.safetensors` (~1.1 GB), `tokenizer.json`, `special_tokens_map.json`, `tokenizer_config.json`, `meta.json`, `labels.json`
- **HF**: [`Niklas1102/mentalkg-xlmr-node`](https://huggingface.co/Niklas1102/mentalkg-xlmr-node)

## `mentalkg-xlmr-edge`

- **base**: `xlm-roberta-base` with four added marker tokens (`[N1]`, `[/N1]`, `[N2]`, `[/N2]`); embeddings resized to `len(tokenizer)`
- **head**: single logit, binary
- **input format**:

  ```
  <s> {entry text} </s></s>
  [N1] {label} ({type}, {polarity}, {time}) [/N1] </s>
  [N2] {label} ({type}, {polarity}, {time}) [/N2] </s>
  ```

  Suffix never truncated; entry text right-truncated if the total exceeds 256 tokens.

- **training**: full data (261,110 pairs, entry_frac 1.0), lr 2e-5, 5 epochs, seed 42, RTX 4090. Two more seeds (43, 44) used for the variance table.
- **dev-tuned threshold**: 0.39
- **test slice** (seed 42): F1 0.749, ROC-AUC 0.812, accuracy 0.765 (n=103,798)
- **3-seed variance** (seeds 42/43/44):
  - with text: F1 0.7488 [0.7483, 0.7492]
  - no text (ablation, `entry_text = "entry"`): F1 0.7408 [0.7407, 0.7410]
- **files**: `config.json`, `model.safetensors` (~1.1 GB), `tokenizer.json`, `added_tokens.json`, `special_tokens_map.json`, `tokenizer_config.json`, `meta.json`, `relation_map.json`
- **HF**: [`Niklas1102/mentalkg-xlmr-edge`](https://huggingface.co/Niklas1102/mentalkg-xlmr-edge)

## `meta.json` schema (both models)

```json
{
  "task": "node_prediction" | "edge_prediction",
  "base_model": "xlm-roberta-base",
  "threshold": 0.28,          // dev-tuned decision threshold
  "test": { ...metrics... },
  "test_per_language": { "en": {...}, "de": {...} },
  "training": { "lr": ..., "epochs": ..., "seed": ..., "max_len": 256 }
}
```

## `relation_map.json` (edge model only)

The connectivity model predicts *whether* two nodes are connected, not what the relation type is. `relation_map.json` records the majority relation type per ordered node-type pair, computed from the 47,714 gold graphs. 35 ordered type pairs. Unknown pairs fall back to `linked_to`.

```json
{
  "n_source_records": 47714,
  "n_type_pairs": 35,
  "relation": {"stressor|emotion": "causes", "thought|emotion": "increases", ...},
  "count":    {"stressor|emotion": 29393, "thought|emotion": 32245, ...}
}
```

The server exposes each edge with `"source": "heuristic_type"` to make this explicit to downstream consumers.

## Loading

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("Niklas1102/mentalkg-xlmr-node")
model = AutoModelForSequenceClassification.from_pretrained("Niklas1102/mentalkg-xlmr-node")
```

Or via the CLI wrapper for end-to-end extraction:

```bash
python scripts/eval_text.py "..."
```

## Conversion provenance

The published safetensors bundles were converted from raw PyTorch `state_dict` files (`model.pt`) using `torch.load` → `AutoModelForSequenceClassification` init → `save_pretrained(safe_serialization=True)`. The conversion helper is one-shot and lives in this repo's history; the SHA256 of each safetensors bundle is recorded below under Released artifacts.

## Intended use

Research on structured extraction from mental-health narrative text. Not clinical, not diagnostic, not for triage. Outputs describe narrated content, not the writer's clinical state.

## Released artifacts

Both bundles were converted from raw `state_dict` (`model.pt`) to safetensors, then uploaded with `HF_TOKEN` read from the environment. Every landed file was byte-compared against its local copy, and both weight files were re-verified by SHA256 after a cold-cache download into an empty directory.

| file | bytes | SHA256 |
|---|---:|---|
| `models/mentalkg-xlmr-node/model.safetensors` | 1,112,598,744 | `f9b69a01f215db8be7804899abdab9cb7b3e5b45b30b975c7f77455b27a9f4ab` |
| `models/mentalkg-xlmr-edge/model.safetensors` | 1,112,214,220 | `55644d6b4d907ecfae7549893278aa2e31a85c4c6edf6bdcc8cb9620320b5110` |

### Repository inventories

[`Niklas1102/mentalkg-xlmr-node`](https://huggingface.co/Niklas1102/mentalkg-xlmr-node), 8 files: `config.json`, `model.safetensors`, `tokenizer.json`, `special_tokens_map.json`, `tokenizer_config.json`, `meta.json`, `labels.json`, `README.md`.

[`Niklas1102/mentalkg-xlmr-edge`](https://huggingface.co/Niklas1102/mentalkg-xlmr-edge), 8 files: `config.json`, `model.safetensors`, `tokenizer.json`, `special_tokens_map.json`, `tokenizer_config.json`, `meta.json`, `relation_map.json`, `README.md`. The four marker tokens live inside `tokenizer.json`.

[`Niklas1102/mentalkg`](https://huggingface.co/datasets/Niklas1102/mentalkg), 12 files: `dataset.jsonl`, `graphs.jsonl`, `variable_pool.json`, `splits.json`, `node_prediction/{train,dev,test}.jsonl`, `node_prediction/labels.json`, `edge_prediction/negative_match.json`, `build_edge_split.py`, `README.md`.

The prepared edge splits stay off the Hub because the repeated entry text inflates them past 5 GB. `scripts/build_edge_split.py` regenerates them from `dataset.jsonl` + `splits.json`. The dataset card documents this.
