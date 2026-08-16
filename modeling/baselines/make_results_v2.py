"""Assemble results/RESULTS.md from the v2 baseline artifacts.

Reads:
  modeling/baselines/artifacts_v2/node_{all,en,de}/metrics.json
  modeling/baselines/artifacts_v2/edge_{hard,random}/metrics.json
  modeling/baselines/artifacts_v2/edge_{pair_only,type_pol_only}/metrics.json
  modeling/baselines/artifacts_v2/edge_shortcut_summary.json
  modeling/data_v2/edge_prediction/negative_match.json
  results/metrics.json                       (v1 numbers, for deltas)

Writes results/RESULTS.md and results/metrics.json,
and prints the consolidated tables.

Usage:
    python modeling/baselines/make_results_v2.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ART = REPO_ROOT / "modeling" / "baselines" / "artifacts_v2"
ART_BERT = REPO_ROOT / "modeling" / "baselines" / "artifacts_v2_bert"
LLAMA_DIR = REPO_ROOT / "results" / "llama_v2"
RESULTS_DIR = REPO_ROOT / "results"
V1_METRICS = RESULTS_DIR / "metrics.json"
NEG_MATCH = REPO_ROOT / "modeling" / "data_v2" / "edge_prediction" / "negative_match.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def load_opt(path: Path) -> dict | None:
    """Load metrics that may not exist yet (GPU runs land asynchronously)."""
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def f(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def main() -> None:
    node_all = load(ART / "node_all" / "metrics.json")
    node_en = load(ART / "node_en" / "metrics.json")
    node_de = load(ART / "node_de" / "metrics.json")
    edge_hard = load(ART / "edge_hard" / "metrics.json")
    edge_random = load(ART / "edge_random" / "metrics.json")
    c1 = load(ART / "edge_pair_only" / "metrics.json")["test"]
    c2 = load(ART / "edge_type_pol_only" / "metrics.json")["test"]
    v1 = load(V1_METRICS) if V1_METRICS.exists() else None
    neg_match = load(NEG_MATCH) if NEG_MATCH.exists() else None

    # GPU-run artifacts (XLM-R fine-tunes, llama zero-shot, string-match floor).
    # v3 = the final optimization session's runs; fall back to earlier dirs.
    def load_best(*names):
        for n in names:
            m = load_opt(ART_BERT / n / "metrics.json")
            if m:
                return m
        return None

    bx_all = load_best("node_xlmr_all_v3", "node_xlmr_all")
    bx_en = load_best("node_xlmr_en_v3", "node_xlmr_en")
    bx_de = load_best("node_xlmr_de_v3", "node_xlmr_de")
    bx_edge = load_best("edge_xlmr_hard_v3", "edge_xlmr_hard")
    bx_edge_c3 = load_best("edge_xlmr_hard_notext_v3", "edge_xlmr_hard_notext")
    ll_node = (load_opt(RESULTS_DIR / "llama_v2_ollama" / "node_prediction" / "metrics.json")
               or load_opt(LLAMA_DIR / "node_prediction" / "metrics.json"))
    ll_edge = load_opt(LLAMA_DIR / "edge_prediction" / "metrics.json")
    smatch = load_opt(ART / "string_match" / "metrics.json")

    na_c = node_all["test"]["common"]
    na_full = node_all["test"]["full"]
    pl = node_all.get("test_per_language", {})
    en_sub = pl.get("en", {}).get("common")
    de_sub = pl.get("de", {}).get("common")
    ne_c = node_en["test"]["common"]
    nd_c = node_de["test"]["common"]
    eh, er = edge_hard["test"], edge_random["test"]

    v1_node_c = v1["node_prediction"]["tfidf_ovr_logreg"]["test_common"] if v1 else None
    v1_edge = v1["edge_prediction"]["tfidf_text_plus_pair_feats_logreg"]["test"] if v1 else None
    v1_c1 = v1["edge_prediction"]["pair_features_only"]["test"] if v1 else None
    v1_c2 = v1["edge_prediction"]["type_polarity_only"]["test"] if v1 else None

    node_rows = [
        ("v1 combined (1.2k entries, 71 common labels)",
         v1_node_c and v1_node_c["micro_f1"], v1_node_c and v1_node_c["macro_f1"],
         v1_node_c and v1_node_c["micro_precision"], v1_node_c and v1_node_c["micro_recall"], 227),
        (f"v2 combined EN+DE, full ({node_all['label_counts']['full']} labels)",
         na_full["micro_f1"], na_full["macro_f1"],
         na_full["micro_precision"], na_full["micro_recall"], na_full["n_examples"]),
        (f"v2 combined EN+DE, common ({node_all['label_counts']['common']})",
         na_c["micro_f1"], na_c["macro_f1"],
         na_c["micro_precision"], na_c["micro_recall"], na_c["n_examples"]),
        ("v2 combined, scored on EN test only",
         en_sub and en_sub["micro_f1"], en_sub and en_sub["macro_f1"],
         en_sub and en_sub["micro_precision"], en_sub and en_sub["micro_recall"],
         en_sub and en_sub["n_examples"]),
        ("v2 combined, scored on DE test only",
         de_sub and de_sub["micro_f1"], de_sub and de_sub["macro_f1"],
         de_sub and de_sub["micro_precision"], de_sub and de_sub["micro_recall"],
         de_sub and de_sub["n_examples"]),
        ("v2 EN-only model on EN test",
         ne_c["micro_f1"], ne_c["macro_f1"],
         ne_c["micro_precision"], ne_c["micro_recall"], ne_c["n_examples"]),
        ("v2 DE-only model on DE test",
         nd_c["micro_f1"], nd_c["macro_f1"],
         nd_c["micro_precision"], nd_c["micro_recall"], nd_c["n_examples"]),
    ]

    edge_rows = [
        ("v1 random negatives, full model", v1_edge and v1_edge["f1"],
         v1_edge and v1_edge["roc_auc"], v1_edge and v1_edge["accuracy"]),
        ("v1 random negatives, C1 pair-only (no text)", v1_c1 and v1_c1["f1"],
         v1_c1 and v1_c1["roc_auc"], v1_c1 and v1_c1["accuracy"]),
        ("v1 random negatives, C2 type+polarity", v1_c2 and v1_c2["f1"],
         v1_c2 and v1_c2["roc_auc"], v1_c2 and v1_c2["accuracy"]),
        ("v2 random negatives, full model", er["f1"], er["roc_auc"], er["accuracy"]),
        ("**v2 HARD negatives, full model**", eh["f1"], eh["roc_auc"], eh["accuracy"]),
        ("v2 hard, C1 pair-only (no text)", c1["f1"], c1["roc_auc"], c1["accuracy"]),
        ("v2 hard, C2 type+polarity only", c2["f1"], c2["roc_auc"], c2["accuracy"]),
    ]

    # Takeaway numbers.
    d_node = (na_c["micro_f1"] - v1_node_c["micro_f1"]) if v1_node_c else None
    text_delta_v1 = (v1_edge["f1"] - v1_c1["f1"]) if v1_edge and v1_c1 else None
    text_delta_v2 = eh["f1"] - c1["f1"]
    text_delta_v2_auc = eh["roc_auc"] - c1["roc_auc"]
    text_delta_v2_c2 = eh["f1"] - c2["f1"]
    tv = {s: neg_match[s]["total_variation_distance"] for s in neg_match} if neg_match else {}

    lines = []
    lines.append("# Results v2, full 41k-sample bilingual run")
    lines.append("")
    lines.append("Data: `modeling/data_v2/` (hard edge negatives) and "
                 "`modeling/data_v2_random/` (v1-style random negatives), both from "
                 "the 41,315 accepted samples of the full pipeline run "
                 "(20,982 EN / 20,333 DE), participant-independent 50/30/20 split "
                 "stratified by language, seed 42. Models: TF-IDF (word 1 to 2 grams) "
                 "with (OvR) logistic regression, class-weight balanced, dev-tuned "
                 "threshold, matching the v1 setup.")
    lines.append("")
    lines.append("## Node prediction (micro/macro-F1, dev-tuned threshold, common concepts)")
    lines.append("")
    lines.append("| Model / eval | Micro-F1 | Macro-F1 | Micro-P | Micro-R | n test |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, mi, ma, p, r, n in node_rows:
        lines.append(f"| {name} | {f(mi)} | {f(ma)} | {f(p)} | {f(r)} | {n if n else '—'} |")
    lines.append("")
    lines.append("## Edge prediction (binary connected, dev-tuned threshold)")
    lines.append("")
    lines.append("| Model / negatives | F1 | ROC-AUC | Accuracy |")
    lines.append("|---|---:|---:|---:|")
    for name, f1v, auc, acc in edge_rows:
        lines.append(f"| {name} | {f(f1v)} | {f(auc)} | {f(acc)} |")
    lines.append("")
    # ---- GPU-run models: XLM-R fine-tunes, llama zero-shot, string floor ----
    lines.append("## Node prediction, transformer + LLM + floor (full 130-label vocab)")
    lines.append("")
    lines.append("| Model / eval | Micro-F1 | Macro-F1 | Micro-P | Micro-R | n test |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    def node_full_row(name, m):
        t = m and m.get("test", {}).get("full")
        lines.append(f"| {name} | {f(t and t['micro_f1'])} | {f(t and t['macro_f1'])} "
                     f"| {f(t and t['micro_precision'])} | {f(t and t['micro_recall'])} "
                     f"| {t['n_examples'] if t else '—'} |")

    node_full_row("XLM-R combined EN+DE", bx_all)
    node_full_row("XLM-R EN-only on EN test", bx_en)
    node_full_row("XLM-R DE-only on DE test", bx_de)
    if smatch:
        st = smatch["test"]
        lines.append(f"| string-match floor | {f(st['micro_f1'])} | {f(st['macro_f1'])} "
                     f"| {f(st['micro_precision'])} | {f(st['micro_recall'])} "
                     f"| {st['n_examples']} |")
    if ll_node:
        lo = ll_node["overall"]
        lines.append(f"| Llama-3.2-3B zero-shot ({ll_node['n_examples']} ex, "
                     f"{ll_node.get('parse_errors', '—')} parse errors, "
                     f"max_new_tokens {ll_node.get('max_new_tokens', '—')}) "
                     f"| {f(lo['micro_f1'])} | {f(lo['macro_f1'])} "
                     f"| {f(lo['micro_precision'])} | {f(lo['micro_recall'])} "
                     f"| {lo['n_examples']} |")
    else:
        lines.append("| Llama-3.2-3B zero-shot |, |, |, |, | pending |")
    lines.append("")

    lines.append("## Edge prediction, transformer + LLM (hard negatives)")
    lines.append("")
    lines.append("| Model | F1 | ROC-AUC | Accuracy | n test |")
    lines.append("|---|---:|---:|---:|---:|")

    def edge_row(name, t, n=None):
        lines.append(f"| {name} | {f(t and t.get('f1'))} | {f(t and t.get('roc_auc'))} "
                     f"| {f(t and t.get('accuracy'))} | {n or (t and t.get('n_total')) or '—'} |")

    edge_row("XLM-R pair-aware (marker tokens)", bx_edge and bx_edge["test"])
    edge_row("XLM-R C3 no-text ablation", bx_edge_c3 and bx_edge_c3["test"])
    if ll_edge:
        edge_row(f"Llama-3.2-3B zero-shot yes/no ({ll_edge['n_examples']} of "
                 f"{ll_edge['n_split_full']} pairs)", ll_edge["overall"])
    else:
        edge_row("Llama-3.2-3B zero-shot yes/no (4k subsample)", None, "pending")
    lines.append("")

    if tv:
        lines.append(f"Hard-negative type-pair match (total variation distance vs positives): "
                     + ", ".join(f"{s} {tv[s]:.3f}" for s in ("train", "dev", "test") if s in tv)
                     + ". Residual mismatch is structural: type pairs with "
                     "P(connected) > 0.5 lack enough unconnected co-occurrences at "
                     "neg_ratio 1.0; details in `modeling/data_v2/report.md`.")
        lines.append("")

    lines.append("## Takeaways")
    lines.append("")
    if v1_node_c:
        lines.append(
            f"- **(a) Scale.** 34× more training data (602 → 20,679 train entries) moved "
            f"node micro-F1 from {v1_node_c['micro_f1']:.3f} to {na_c['micro_f1']:.3f} "
            f"({d_node:+.3f}) and macro-F1 from {v1_node_c['macro_f1']:.3f} to "
            f"{na_c['macro_f1']:.3f}, with the label space growing 93 → "
            f"{node_all['label_counts']['full']} concepts (all now ≥ 20 train support)."
        )
    lines.append(
        f"- **(b) Language.** The combined EN+DE model scores micro-F1 "
        f"{en_sub['micro_f1']:.3f} on EN and {de_sub['micro_f1']:.3f} on DE; "
        f"monolingual models reach {ne_c['micro_f1']:.3f} (EN-only) and "
        f"{nd_c['micro_f1']:.3f} (DE-only). Combined vs monolingual differs by "
        f"{en_sub['micro_f1'] - ne_c['micro_f1']:+.3f} (EN) and "
        f"{de_sub['micro_f1'] - nd_c['micro_f1']:+.3f} (DE), so one bilingual model "
        f"delivers {'the same accuracy as two monolingual ones' if abs(en_sub['micro_f1'] - ne_c['micro_f1']) < 0.01 and abs(de_sub['micro_f1'] - nd_c['micro_f1']) < 0.01 else 'measurably different accuracy'}."
    )
    if text_delta_v1 is not None:
        v1_c2_auc = v1_c2["roc_auc"]
        lines.append(
            f"- **(c) Shortcut.** Hard negatives removed most of the structural shortcut: "
            f"the pure type+polarity model (C2) fell from AUC {v1_c2_auc:.3f} (v1 random "
            f"negatives) to {c2['roc_auc']:.3f}, and every model got much harder. The full "
            f"model dropped from F1 {v1_edge['f1']:.3f}/AUC {v1_edge['roc_auc']:.3f} to "
            f"{eh['f1']:.3f}/{eh['roc_auc']:.3f}. **The entry text stays inert for this "
            f"baseline family**: pair features alone (C1, with concept ids) already match "
            f"the full text+pair model ({text_delta_v2:+.3f} F1, {text_delta_v2_auc:+.3f} "
            f"AUC for adding text). Two reasons: (i) the residual signal lives in "
            f"concept-PAIR statistics, which C1 sees directly; (ii) TF-IDF text is "
            f"identical for every candidate pair from the same entry, so a bag-of-words "
            f"model only sees pair features vary within an entry. C2 also stays above "
            f"chance (AUC {c2['roc_auc']:.3f} vs 0.5) because the type-pair match is "
            f"capped by candidate availability (TV ≈ {tv.get('train', 0):.2f}), which "
            f"forbids exact matching at neg_ratio 1.0 for pairs with P(connected) > 0.5. "
            f"Verdict: the type-pair shortcut is largely closed; showing that *text* "
            f"matters for edges needs a pair-aware text model (e.g. BERT with node-span "
            f"markers), a class that entry-level TF-IDF sits outside of."
        )
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("python modeling/prepare_data.py --negatives hard   --output modeling/data_v2")
    lines.append("python modeling/prepare_data.py --negatives random --output modeling/data_v2_random")
    lines.append("python modeling/baselines/node_baseline.py --language all")
    lines.append("python modeling/baselines/node_baseline.py --language en")
    lines.append("python modeling/baselines/node_baseline.py --language de")
    lines.append("python modeling/baselines/edge_baseline.py --data-dir modeling/data_v2/edge_prediction --tag hard")
    lines.append("python modeling/baselines/edge_baseline.py --data-dir modeling/data_v2_random/edge_prediction --tag random")
    lines.append("python modeling/baselines/edge_shortcut.py")
    lines.append("# GPU (RTX 4090 used for the reported numbers):")
    lines.append("python modeling/baselines/node_bert.py --language all   # + en, de")
    lines.append("python modeling/baselines/edge_bert.py --tag hard        # + --no-text")
    lines.append("python modeling/baselines/llama_v2.py node               # + edge")
    lines.append("python modeling/baselines/string_match_floor.py")
    lines.append("python modeling/baselines/make_results_v2.py")
    lines.append("```")
    lines.append("")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_md = RESULTS_DIR / "RESULTS.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")

    metrics_out = {
        "bert": {
            "node_xlmr_all": bx_all and bx_all["test"],
            "node_xlmr_en": bx_en and bx_en["test"],
            "node_xlmr_de": bx_de and bx_de["test"],
            "node_xlmr_all_per_language": bx_all and bx_all.get("test_per_language"),
            "edge_xlmr_hard": bx_edge and bx_edge["test"],
            "edge_xlmr_hard_notext": bx_edge_c3 and bx_edge_c3["test"],
        },
        "llama": {
            "node": ll_node and {**ll_node["overall"],
                                 "parse_errors": ll_node.get("parse_errors"),
                                 "per_language": ll_node.get("per_language")},
            "edge": ll_edge and {**ll_edge["overall"],
                                 "per_language": ll_edge.get("per_language")},
        },
        "string_match_floor": smatch and smatch["test"],
        "node": {
            "v1_combined_common": v1_node_c,
            "v2_combined_full": na_full,
            "v2_combined_common": na_c,
            "v2_combined_on_en": en_sub,
            "v2_combined_on_de": de_sub,
            "v2_en_only_common": ne_c,
            "v2_de_only_common": nd_c,
        },
        "edge": {
            "v1_random_full": v1_edge,
            "v1_random_c1_pair_only": v1_c1,
            "v1_random_c2_type_pol": v1_c2,
            "v2_random_full": er,
            "v2_hard_full": eh,
            "v2_hard_c1_pair_only": c1,
            "v2_hard_c2_type_pol": c2,
            "hard_negative_tv_distance": tv,
            "text_delta_f1_v1_random": text_delta_v1,
            "text_delta_f1_v2_hard": text_delta_v2,
        },
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics_out, indent=2))

    print("\n".join(lines))
    print(f"\nWrote {out_md}")
    print(f"Wrote {RESULTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
