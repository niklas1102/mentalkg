# server/

FastAPI server that loads the two XLM-R checkpoints from `../models/` and exposes a `/extract` endpoint.

## Endpoints

- `GET /`: service info
- `GET /health`: `{ok, models_loaded, node_ckpt, edge_ckpt}`
- `POST /extract` `{text: str}`: returns `{nodes, edges, predicted_concepts, entry_text, node_threshold, edge_threshold, note}`

## What it predicts

- **Nodes**: XLM-R multi-label sigmoid over 130 concept ids. Kept above `meta.threshold` (0.28), or top-3 fallback if nothing clears. Each node comes back with concept_id, type, label, polarity, and a placeholder `time_anchor.text = "now"` (time anchors are placeholder, the model predicts concept identity only).
- **Edge connectivity**: XLM-R pair-aware binary head with marker tokens, scored on all `C(nodes, 2)` pairs. Both orderings are scored and averaged. Above `meta.threshold` (0.39).
- **Edge relation type**: majority lookup in `relation_map.json` bundled with the edge model. Each such edge carries `"source": "heuristic_type"` to make the source explicit.

## Run

```bash
# after models/ is populated (scripts/download_models.py)
bash ../scripts/run_server.sh                # localhost:8000
# or directly:
python -m uvicorn app:app --port 8000
```

## Environment overrides

- `NODE_CKPT_DIR`, `EDGE_CKPT_DIR`: directories with `config.json + model.safetensors + tokenizer + meta.json`
- `NODE_META`, `EDGE_META`: override the meta.json paths (defaults: alongside checkpoints)
- `SERVER_DEVICE`: one of `cuda`, `mps`, `cpu`. Default: `cuda` if available, then `mps`, then `cpu`.
- `PORT`: port for `scripts/run_server.sh` (default 8000)

## Example

```bash
curl -s -X POST localhost:8000/extract \
     -H 'content-type: application/json' \
     -d '{"text":"I could not sleep and the deadline is tomorrow."}' | jq
```

Returns a JSON graph with 4 nodes (stressor_sleep_schedule_disruption, stressor_deadline, coping_sleep_routine, activity_working) and 0 edges (this short a text stays below the 0.39 threshold on every pair).
