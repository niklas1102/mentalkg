# Journal Graph (Obsidian plugin)

Send the active note to a local extractor and view its knowledge graph. Requires the MentalKG server running locally.

## Install (from source)

```bash
cd plugin
npm install
node esbuild.config.mjs production
```

That writes `plugin/main.js` and copies `main.js + manifest.json + styles.css` into `plugin/demo-vault/.obsidian/plugins/journal-graph/`. To install into a real vault, copy those three files into `<vault>/.obsidian/plugins/journal-graph/`.

Or use the wrapper:

```bash
bash ../scripts/build_plugin.sh
```

## Enable

Open your vault in Obsidian → Settings → Community plugins → enable **Journal Graph**. Set the server URL if it differs from the default.

## Settings

| setting | default | what |
|---|---|---|
| Journal folder | `journal` | vault folder that Timeline and export read from |
| Extraction server URL | `http://localhost:8000/extract` | server endpoint |
| Treat all notes as journal entries | off | if on, Timeline walks every markdown note |
| Auto-refresh current note | on | re-extract a few seconds after typing stops |
| Auto-refresh delay | 1.5 s | debounce before re-extract |

## Commands

- **Extract graph for current note**: runs the extractor on the active note, renders the graph in the side panel
- **Open journal timeline**: timeline view across the journal folder
- **New journal entry for today**: creates and opens `journal/YYYY-MM-DD.md`
- **Paste clipboard as live typing (demo)**: types the clipboard into the note over ~5 s (Ctrl/Cmd + Alt + J on the demo hotkey)

## What the graph shows

- **Nodes** and **edges** come from the two XLM-R models served at the configured URL.
- **Edge relation types** (causes, increases, decreases, follows, linked_to) are a majority lookup over gold-graph type-pair statistics bundled with the edge model. This is a heuristic, not a model prediction; edges carry `source: "heuristic_type"` and the panel labels the source.
- **Time anchors** are placeholder: the server attaches `time_anchor.text = "now"` to every extracted node.

## Therapist export

The panel has an **Export therapist summary** action that writes a markdown summary of the last N days of the journal folder. Content: overview stats, recurring problem patterns (concepts seen on ≥3 of N days), mood trend, best and worst days.

## Demo vault

`plugin/demo-vault/` is a small self-contained Obsidian vault with 14 example journal entries (2026-06-01 → 2026-06-14) and one quick note. The entries were originally tuned against a llama3.2:3b prompt server; extraction under the XLM-R server will produce different (typically more precise) graphs, which is expected.
