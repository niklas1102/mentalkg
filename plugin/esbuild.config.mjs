import esbuild from "esbuild";
import builtins from "builtin-modules";
import { copyFileSync, mkdirSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const isProduction = (process.argv[2] || "").toLowerCase() === "production";

const VAULT_PLUGIN_DIR = resolve(
  __dirname,
  "demo-vault/.obsidian/plugins/journal-graph",
);

const copyToVault = {
  name: "copy-to-demo-vault",
  setup(build) {
    build.onEnd((result) => {
      if (result.errors.length) return;
      if (!existsSync(VAULT_PLUGIN_DIR)) mkdirSync(VAULT_PLUGIN_DIR, { recursive: true });
      for (const f of ["main.js", "manifest.json", "styles.css"]) {
        const src = resolve(__dirname, f);
        if (existsSync(src)) {
          copyFileSync(src, resolve(VAULT_PLUGIN_DIR, f));
        }
      }
      console.log(`[copy] wrote main.js + manifest.json + styles.css → ${VAULT_PLUGIN_DIR}`);
    });
  },
};

const opts = {
  entryPoints: ["main.ts"],
  bundle: true,
  outfile: "main.js",
  format: "cjs",
  target: "es2018",
  platform: "browser",
  mainFields: ["browser", "module", "main"],
  conditions: ["browser"],
  external: [
    "obsidian",
    "electron",
    "@codemirror/autocomplete",
    "@codemirror/collab",
    "@codemirror/commands",
    "@codemirror/language",
    "@codemirror/lint",
    "@codemirror/search",
    "@codemirror/state",
    "@codemirror/view",
    "@lezer/common",
    "@lezer/highlight",
    "@lezer/lr",
    ...builtins,
  ],
  logLevel: "info",
  sourcemap: isProduction ? false : "inline",
  treeShaking: true,
  minify: isProduction,
  plugins: [copyToVault],
};

if (isProduction) {
  await esbuild.build(opts);
} else {
  const ctx = await esbuild.context(opts);
  await ctx.watch();
  console.log("[esbuild] watching for changes…");
}
