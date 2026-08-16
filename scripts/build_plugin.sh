#!/usr/bin/env bash
# Build the Obsidian plugin and copy the artifacts into demo-vault.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE/plugin"

if [ ! -d node_modules ]; then
  npm install
fi

node esbuild.config.mjs production

DEST="$HERE/plugin/demo-vault/.obsidian/plugins/journal-graph"
mkdir -p "$DEST"
cp -f main.js manifest.json styles.css "$DEST/"

echo "built plugin → $HERE/plugin/main.js"
echo "installed into demo vault → $DEST"
