import argparse
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import llm
from graph_stats import connectivity, format_table
from scenarios import generate_dataset

BASE = Path(__file__).resolve().parent


def load_config():
    path = BASE / "config.json"
    if not path.exists():
        path = BASE / "config.example.json"
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # Resolve ${VAR} placeholders from the environment. Fail early if a placeholder
    # is left unresolved, so the LLM call does not send a literal placeholder.
    key = cfg.get("openrouter_api_key", "")
    if isinstance(key, str) and key.startswith("${") and key.endswith("}"):
        env_name = key[2:-1]
        env_val = os.environ.get(env_name)
        if not env_val:
            raise RuntimeError(
                f"config's openrouter_api_key is placeholder '{key}' and "
                f"environment variable {env_name} is not set"
            )
        cfg["openrouter_api_key"] = env_val
    return cfg


def read_jsonl(path):
    items = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    return items


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def graph_hash(graph):
    return hashlib.sha1(
        json.dumps(graph, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


ALWAYS_EXTRA_TIME_EN = ["next week", "last week", "next month", "last month",
                        "next year", "the day after tomorrow", "in a few days",
                        "a few days from now", "weeks from now"]
# German equivalents so the temporal-hallucination backstop is symmetric across
# languages (the LLM verifier is the primary check; this heuristic must not add a
# penalty to English entries that German entries can never receive).
ALWAYS_EXTRA_TIME_DE = ["nächste woche", "letzte woche", "nächsten monat",
                        "letzten monat", "nächstes jahr", "letztes jahr",
                        "übermorgen", "vorgestern", "in ein paar tagen",
                        "in einigen tagen", "in wenigen tagen"]


def completeness(sample, verifier):
    gn = [n["node_id"] for n in sample["graph"]["nodes"]]
    ge = [e["edge_id"] for e in sample["graph"]["edges"]]
    vn = [n.get("node_id") for n in verifier.get("nodes", [])]
    ve = [e.get("edge_id") for e in verifier.get("edges", [])]
    missing_n = [x for x in gn if vn.count(x) != 1]
    missing_e = [x for x in ge if ve.count(x) != 1]
    return missing_n, missing_e


def _snippet(text, t, i, length):
    return text[max(0, i - 25): i + length + 25].replace("\n", " ").strip()


def _extra_temporal_en(sample, text):
    t = text.lower()
    offsets = {n["time_anchor"]["day_offset"] for n in sample["graph"]["nodes"]}
    found = []
    if "tomorrow" in t and "the day after tomorrow" not in t and 1 not in offsets:
        found.append({"content": "extra future reference 'tomorrow' not in graph",
                      "evidence": _snippet(text, t, t.find("tomorrow"), len("tomorrow"))})
    if -1 not in offsets:
        for m in ("yesterday", "last night"):
            if m in t:
                found.append({"content": f"extra past reference '{m}' not in graph",
                              "evidence": _snippet(text, t, t.find(m), len(m))})
    for m in ALWAYS_EXTRA_TIME_EN:
        if m in t:
            found.append({"content": f"extra time reference '{m}' not in graph",
                          "evidence": _snippet(text, t, t.find(m), len(m))})
    return found


def _extra_temporal_de(sample, text):
    t = text.lower()
    offsets = {n["time_anchor"]["day_offset"] for n in sample["graph"]["nodes"]}
    found = []
    # future adverb 'morgen' (=tomorrow). Matched case-sensitively on the ORIGINAL
    # text: written German distinguishes the noun 'Morgen' (=morning, capitalized)
    # from the adverb 'morgen' (=tomorrow, lowercase mid-sentence), so lowercase-
    # only \b-bounded matching excludes 'am Morgen', 'morgendlich', 'morgens' and
    # 'übermorgen'. A sentence-initial capitalized adverb slips through as a false
    # negative, which the LLM verifier still catches.
    if 1 not in offsets:
        for m in re.finditer(r"\bmorgen\b", text):
            i = m.start()
            if t[max(0, i - 6):i] == "heute ":  # informal lowercase 'heute morgen'
                continue
            found.append({"content": "extra future reference 'morgen' not in graph",
                          "evidence": _snippet(text, t, i, len("morgen"))})
            break
    # past 'gestern' (covers 'gestern abend/nacht') and standalone 'letzte nacht'.
    if -1 not in offsets:
        for m in ("gestern", "letzte nacht"):
            if m in t:
                found.append({"content": f"extra past reference '{m}' not in graph",
                              "evidence": _snippet(text, t, t.find(m), len(m))})
    for m in ALWAYS_EXTRA_TIME_DE:
        if m in t:
            found.append({"content": f"extra time reference '{m}' not in graph",
                          "evidence": _snippet(text, t, t.find(m), len(m))})
    return found


def extra_temporal(sample, text):
    if sample["participant"].get("language") == "de":
        return _extra_temporal_de(sample, text)
    return _extra_temporal_en(sample, text)


def score_sample(sample, verifier, text):
    nodes = sample["graph"]["nodes"]
    edges = sample["graph"]["edges"]
    node_status = {n.get("node_id"): n for n in verifier.get("nodes", [])}
    edge_status = {e.get("edge_id"): e for e in verifier.get("edges", [])}

    total_nodes = len(nodes)
    total_edges = len(edges)
    mentioned = sum(1 for n in nodes if node_status.get(n["node_id"], {}).get("mentioned") is True)
    temporal_ok = sum(1 for n in nodes if node_status.get(n["node_id"], {}).get("temporal_correct") is True)
    expressed = sum(1 for e in edges if edge_status.get(e["edge_id"], {}).get("expressed") is True)

    node_coverage = mentioned / total_nodes if total_nodes else 1.0
    edge_coverage = expressed / total_edges if total_edges else 1.0
    temporal_consistency = temporal_ok / total_nodes if total_nodes else 1.0

    extra = []
    for x in verifier.get("extra_content", []):
        if isinstance(x, dict):
            content = str(x.get("content", "")).strip()
            if content:
                extra.append({"content": content, "evidence": str(x.get("evidence", "")).strip()})
        elif str(x).strip():
            extra.append({"content": str(x).strip(), "evidence": ""})
    for te in extra_temporal(sample, text):
        if not any(te["content"].lower() in e["content"].lower()
                   or e["content"].lower() in te["content"].lower() for e in extra):
            extra.append(te)
    hallucination_count = len(extra)
    hallucination_score = round(min(hallucination_count, 3) / 3.0, 4)

    overall_confidence = round(
        0.40 * node_coverage
        + 0.25 * edge_coverage
        + 0.25 * temporal_consistency
        + 0.10 * (1.0 - hallucination_score),
        4,
    )

    return {
        "node_coverage": round(node_coverage, 4),
        "edge_coverage": round(edge_coverage, 4),
        "temporal_consistency": round(temporal_consistency, 4),
        "hallucination_count": hallucination_count,
        "hallucination_score": hallucination_score,
        "hallucinated_content": extra,
        "overall_confidence": overall_confidence,
        "nodes_total": total_nodes,
        "nodes_mentioned": mentioned,
        "edges_total": total_edges,
        "edges_expressed": expressed,
    }


def decide(scores, t):
    if (scores["node_coverage"] >= t["accept_min_node_coverage"]
            and scores["edge_coverage"] >= t["accept_min_edge_coverage"]
            and scores["temporal_consistency"] >= t["accept_min_temporal_consistency"]
            and scores["hallucination_count"] <= t["accept_max_hallucination"]):
        return "accept"
    if (scores["node_coverage"] >= t["review_min_node_coverage"]
            and scores["edge_coverage"] >= t["review_min_edge_coverage"]
            and scores["temporal_consistency"] >= t["review_min_temporal_consistency"]
            and scores["hallucination_count"] <= t["review_max_hallucination"]):
        return "review"
    return "reject"


def build_reason(sample, verifier, scores, decision):
    nodes = sample["graph"]["nodes"]
    label_by_id = {n["node_id"]: n["label"] for n in nodes}
    node_status = {n.get("node_id"): n for n in verifier.get("nodes", [])}
    edge_status = {e.get("edge_id"): e for e in verifier.get("edges", [])}

    missing = [n["label"] for n in nodes
               if node_status.get(n["node_id"], {}).get("mentioned") is not True]
    bad_time = []
    for n in nodes:
        st = node_status.get(n["node_id"], {})
        if st.get("mentioned") is True and st.get("temporal_correct") is not True:
            ev = str(st.get("temporal_evidence", "")).strip()
            bad_time.append(f"{n['label']} (expected '{n['time_anchor']['text']}'"
                            + (f"; {ev}" if ev else "") + ")")
    unexpressed = []
    for e in sample["graph"]["edges"]:
        if edge_status.get(e["edge_id"], {}).get("expressed") is not True:
            unexpressed.append(f"{label_by_id[e['source_node_id']]}->{label_by_id[e['target_node_id']]}")
    halluc = [h["content"] for h in scores.get("hallucinated_content", [])]

    if decision == "accept":
        return (f"All {scores['nodes_total']} nodes and {scores['edges_total']} "
                f"relations clearly expressed with correct timing; no extra content.")

    parts = []
    if missing:
        parts.append(f"{len(missing)} node(s) not clearly expressed: {', '.join(missing)}")
    if unexpressed:
        parts.append(f"{len(unexpressed)} relation(s) not expressed: {', '.join(unexpressed)}")
    if bad_time:
        parts.append(f"incorrect timing for: {', '.join(bad_time)}")
    if halluc:
        parts.append(f"extra content not in graph: {', '.join(halluc)}")
    if not parts:
        parts.append("borderline match flagged by scoring thresholds")
    return f"{decision.upper()} — " + "; ".join(parts) + "."


def _add_usage(acc, u):
    acc["prompt_tokens"] += u["prompt_tokens"]
    acc["completion_tokens"] += u["completion_tokens"]
    acc["calls"] += 1


def verify_sample(config, sample, text, usage_acc):
    messages = llm.build_verification_messages(sample, text)
    attempts = config.get("max_verify_attempts", 3)
    raw = ""
    parsed = None
    error = None
    for attempt in range(attempts):
        raw, u = llm.chat(config, messages, config["verification_temperature"])
        _add_usage(usage_acc, u)
        try:
            candidate = llm.parse_json_object(raw)
        except (ValueError, json.JSONDecodeError) as e:
            error = str(e)
            continue
        parsed = candidate
        miss_n, miss_e = completeness(sample, candidate)
        if not miss_n and not miss_e:
            return candidate, raw, None
        error = (f"incomplete verifier output: missing/duplicate node checks "
                 f"{miss_n}, edge checks {miss_e}")
    return parsed, raw, error


def process_one(config, sample):
    usage_acc = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    language = sample["participant"].get("language", "en")

    gen_messages = llm.build_generation_messages(sample)
    text, u = llm.chat(config, gen_messages, config["generation_temperature"])
    _add_usage(usage_acc, u)
    has_future_node = any(n["time_anchor"]["day_offset"] == 1
                          for n in sample["graph"]["nodes"])
    for _ in range(config.get("max_generation_attempts", 3)):
        banned = llm.contains_banned(text, language)
        # German models habitually close with "Ich hoffe, dass es morgen besser
        # wird" — a future reference the graph does not license. Catch the
        # lowercase adverb 'morgen' (tomorrow) at generation time and rewrite.
        de_morgen = (language == "de" and not has_future_node
                     and re.search(r"\bmorgen\b", text) is not None)
        if not banned and not de_morgen:
            break
        instructions = []
        if banned:
            instructions.append("It must not contain any of these phrases: "
                                f"{', '.join(banned)}. Express the same meaning "
                                "through natural storytelling instead.")
        if de_morgen:
            instructions.append("Streiche jeden Bezug auf 'morgen' oder die "
                                "Zukunft ersatzlos (z.B. 'Ich hoffe, dass es "
                                "morgen besser wird') — der Eintrag endet bei "
                                "der Gegenwart, ohne Ausblick.")
        retry_messages = gen_messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": "Rewrite the entry. "
             + " ".join(instructions) + " Output only the entry."},
        ]
        text, u = llm.chat(config, retry_messages, config["generation_temperature"])
        _add_usage(usage_acc, u)
    verifier, raw_verifier, verify_error = verify_sample(config, sample, text, usage_acc)

    if verifier is None:
        scores = {
            "node_coverage": 0.0, "edge_coverage": 0.0,
            "temporal_consistency": 0.0, "hallucination_count": 0,
            "hallucination_score": 1.0, "hallucinated_content": [],
            "overall_confidence": 0.0,
            "nodes_total": len(sample["graph"]["nodes"]), "nodes_mentioned": 0,
            "edges_total": len(sample["graph"]["edges"]), "edges_expressed": 0,
        }
        decision = "reject"
        decision_reason = f"REJECT — verifier output could not be parsed after retries: {verify_error}"
    elif verify_error:
        scores = score_sample(sample, verifier, text)
        decision = "reject"
        decision_reason = f"REJECT — {verify_error}"
    else:
        scores = score_sample(sample, verifier, text)
        decision = decide(scores, config["thresholds"])
        decision_reason = build_reason(sample, verifier, scores, decision)

    final_verification = {
        "nodes": (verifier or {}).get("nodes", []),
        "edges": (verifier or {}).get("edges", []),
        "extra_content": scores.get("hallucinated_content", []),
        "overall": decision,
        "reason": decision_reason,
    }
    if verifier is None:
        final_verification["error"] = verify_error

    record = {
        "schema_version": sample["schema_version"],
        "generator_version": sample.get("generator_version"),
        "sample_id": sample["sample_id"],
        "source": sample["source"],
        "scenario": sample["scenario"],
        "language": language,
        "participant": sample["participant"],
        "entry": {
            "entry_id": sample["entry"]["entry_id"],
            "day_index": sample["entry"]["day_index"],
            "text": text,
        },
        "day_index": sample["entry"]["day_index"],
        "mood_score": sample.get("mood_score"),
        "previous_day_context": sample.get("previous_day_context", ""),
        "graph": sample["graph"],
        "generated_entry": text,
        "verification": final_verification,
        "scores": scores,
        "decision": decision,
        "decision_reason": decision_reason,
        "usage": {
            "prompt_tokens": usage_acc["prompt_tokens"],
            "completion_tokens": usage_acc["completion_tokens"],
            "llm_calls": usage_acc["calls"],
        },
    }
    verification_entry = {
        "sample_id": sample["sample_id"],
        "raw_verifier_output": raw_verifier,
        "parsed": verifier,
        "error": verify_error,
    }
    return record, verification_entry, usage_acc


# ---------------------------------------------------------------------------
# Cost / usage tracking
# ---------------------------------------------------------------------------

def compute_cost(prompt_tokens, completion_tokens, pricing):
    return round(prompt_tokens / 1_000_000 * pricing["prompt_per_1m"]
                 + completion_tokens / 1_000_000 * pricing["completion_per_1m"], 6)


def write_usage(path, state, pricing, model):
    payload = {
        "model": model,
        "pricing_per_1m": pricing,
        "prompt_tokens": state["prompt_tokens"],
        "completion_tokens": state["completion_tokens"],
        "total_tokens": state["prompt_tokens"] + state["completion_tokens"],
        "llm_calls": state["calls"],
        "samples": state["samples"],
        "cost_usd": compute_cost(state["prompt_tokens"], state["completion_tokens"], pricing),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return payload


def usage_from_records(records):
    """Reconstruct cumulative token usage by summing the per-sample usage stored
    on each dataset record. This keeps usage.json exactly consistent with
    dataset.jsonl on resume (no tokens lost between flushes)."""
    state = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    for d in records:
        u = d.get("usage") or {}
        state["prompt_tokens"] += int(u.get("prompt_tokens", 0))
        state["completion_tokens"] += int(u.get("completion_tokens", 0))
        state["calls"] += int(u.get("llm_calls", 0))
    return state


def fmt_hms(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _decision_counts(items):
    c = {"accept": 0, "review": 0, "reject": 0}
    for d in items:
        c[d["decision"]] = c.get(d["decision"], 0) + 1
    return c


def _rate_line(name, c):
    n = sum(c.values())
    if not n:
        return f"| {name} | 0 | 0 | 0 | 0 | – | – | – |"
    return (f"| {name} | {n} | {c['accept']} | {c['review']} | {c['reject']} | "
            f"{c['accept']/n:.0%} | {c['review']/n:.0%} | {c['reject']/n:.0%} |")


def language_breakdown_lines(dataset):
    lines = ["## Decisions by language", "",
             "| language | n | accept | review | reject | acc% | rev% | rej% |",
             "|---|---|---|---|---|---|---|---|"]
    for lang in ("en", "de"):
        items = [d for d in dataset if d.get("language", d["participant"].get("language")) == lang]
        lines.append(_rate_line(lang, _decision_counts(items)))
    lines.append("")
    return lines


def template_breakdown_lines(dataset):
    lines = ["## Decisions by template", "",
             "| template | n | accept | review | reject | acc% | rev% | rej% |",
             "|---|---|---|---|---|---|---|---|"]
    scenarios = sorted({d.get("scenario", "?") for d in dataset})
    for sc in scenarios:
        items = [d for d in dataset if d.get("scenario") == sc]
        lines.append(_rate_line(sc, _decision_counts(items)))
    lines.append("")
    return lines


def connectivity_min_count(n):
    # small samples (pilot) use a low threshold; large runs cap at 100 so real
    # low-frequency pairs are still checked rather than silently excluded.
    return 15 if n < 2000 else 100


def connectivity_lines(dataset):
    n = len(dataset)
    min_count = connectivity_min_count(n)
    stats = connectivity(dataset)
    lines = ["## Type-pair connectivity  P(connected | type pair)", "",
             f"Anti-shortcut check ({n} graphs, common pair = co-occurrence "
             f">= {min_count}). Target: every common pair strictly in (0.15, 0.85).",
             "", "```", format_table(stats, min_count=min_count), "```", ""]
    return lines


def _entry_block(d):
    lines = []
    p = d["participant"]
    lines.append(f"### {d['sample_id']} — {d.get('language','?')} / "
                 f"{p.get('writing_style','?')} / {d.get('scenario','?')} → {d['decision']}")
    lines.append(f"_age {p.get('age')}, {p.get('gender')}, {p.get('occupation')}_")
    lines.append("")
    lines.append("Graph nodes:")
    for nd in d["graph"]["nodes"]:
        lines.append(f"- [{nd['type']}] {nd['label']} @ {nd['time_anchor']['text']} ({nd['temporal_status']})")
    lines.append("")
    lines.append("Graph relations:")
    label_by_id = {nn["node_id"]: nn["label"] for nn in d["graph"]["nodes"]}
    for e in d["graph"]["edges"]:
        lines.append(f"- {label_by_id[e['source_node_id']]} [{e['type']}] {label_by_id[e['target_node_id']]}")
    lines.append("")
    lines.append("Generated entry:")
    lines.append("")
    for para in d["generated_entry"].split("\n"):
        lines.append("> " + para if para.strip() else ">")
    lines.append("")
    lines.append(f"Scores: {json.dumps(d['scores'])}")
    lines.append("")
    lines.append(f"Decision reason: {d.get('decision_reason','')}")
    return lines


SHORT_STYLES = ("terse_fragmented", "matter_of_fact")
LONG_STYLES = ("rambling", "emotional")


def _pick_example(dataset, lang, styles):
    pool = [d for d in dataset
            if d.get("language") == lang
            and d["participant"].get("writing_style") in styles]
    for pref in ("accept", "review", "reject"):
        for d in pool:
            if d["decision"] == pref:
                return d
    return pool[0] if pool else None


def example_entries_lines(dataset):
    lines = ["## Example entries (full text, for quality judging)", ""]
    picks = []
    for lang in ("en", "de"):
        for label, styles in (("short-style", SHORT_STYLES), ("long-style", LONG_STYLES)):
            d = _pick_example(dataset, lang, styles)
            if d:
                picks.append((f"{lang} {label}", d))
    seen = set()
    for label, d in picks:
        if d["sample_id"] in seen:
            continue
        seen.add(d["sample_id"])
        lines.append(f"**{label}**")
        lines.append("")
        lines.extend(_entry_block(d))
        lines.append("")
    return lines


def cost_lines(usage_payload, pilot_samples, full_samples):
    pilot_cost = usage_payload["cost_usd"]
    per_sample = pilot_cost / pilot_samples if pilot_samples else 0.0
    extrapolated = round(per_sample * full_samples, 2)
    lines = ["## Cost", "",
             f"- Model: `{usage_payload['model']}`  (prompt "
             f"${usage_payload['pricing_per_1m']['prompt_per_1m']}/1M, completion "
             f"${usage_payload['pricing_per_1m']['completion_per_1m']}/1M)",
             f"- Prompt tokens: {usage_payload['prompt_tokens']:,}",
             f"- Completion tokens: {usage_payload['completion_tokens']:,}",
             f"- Total tokens: {usage_payload['total_tokens']:,}",
             f"- LLM calls: {usage_payload['llm_calls']:,}",
             f"- **This run cost: ${pilot_cost:.4f}** over {pilot_samples} samples "
             f"(${per_sample*1000:.4f} per 1000 samples)",
             f"- **Extrapolated full-run cost ({full_samples:,} samples): "
             f"~${extrapolated:,.2f}** (linear estimate, assumes similar retry/accept rates)",
             ""]
    return lines


def write_report(config, dataset, out_dir, usage_payload=None, full_samples=None):
    n = len(dataset)
    counts = _decision_counts(dataset)
    sums = {"node_coverage": 0.0, "edge_coverage": 0.0,
            "temporal_consistency": 0.0, "hallucination_score": 0.0,
            "overall_confidence": 0.0}
    halluc_samples = 0
    for d in dataset:
        s = d["scores"]
        for k in sums:
            sums[k] += s.get(k, 0.0)
        if s["hallucination_count"] > 0:
            halluc_samples += 1

    def avg(k):
        return round(sums[k] / n, 4) if n else 0.0

    participants = sorted({d["participant"]["participant_id"] for d in dataset})
    days_per = sorted({d["day_index"] for d in dataset})

    lines = []
    lines.append("# Longitudinal Synthetic Paired Dataset — Run Report")
    lines.append("")
    lines.append(f"- Model: `{config['model']}` via {config['provider']}")
    lines.append(f"- Generator version: {config.get('generator_version')}")
    lines.append(f"- Structure: **{len(participants)} participants × "
                 f"{len(days_per)} days = {n} samples**")
    lines.append(f"- Accepted: **{counts['accept']}**  |  Review: **{counts['review']}**  "
                 f"|  Rejected: **{counts['reject']}**")
    lines.append(f"- Samples with hallucinated content: **{halluc_samples}**")
    lines.append("")
    lines.append("## Mean scores")
    lines.append(f"- Node coverage: {avg('node_coverage')}")
    lines.append(f"- Edge coverage: {avg('edge_coverage')}")
    lines.append(f"- Temporal consistency: {avg('temporal_consistency')}")
    lines.append(f"- Hallucination score (0 best, 1 worst): {avg('hallucination_score')}")
    lines.append(f"- Overall confidence: {avg('overall_confidence')}")
    lines.append("")
    lines.extend(language_breakdown_lines(dataset))
    lines.extend(template_breakdown_lines(dataset))
    lines.extend(connectivity_lines(dataset))
    if usage_payload is not None:
        lines.extend(cost_lines(usage_payload, n, full_samples or n))
    lines.extend(example_entries_lines(dataset))

    lines.append("## Output files")
    lines.append("- `graphs.jsonl` — gold graphs only")
    lines.append("- `dataset.jsonl` — full paired + verified samples")
    lines.append("- `verification.jsonl` — raw verifier outputs")
    lines.append("- `usage.json` — token usage and cost")
    lines.append("- `report.md` — this report")

    with open(out_dir / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def print_pilot_summary(config, dataset, usage_payload, full_samples):
    """Print the required post-pilot outputs to stdout."""
    print("\n" + "=" * 70)
    print("PILOT SUMMARY")
    print("=" * 70)

    print("\n(a) Accept / review / reject by language")
    print(f"{'language':<10}{'n':>6}{'accept':>8}{'review':>8}{'reject':>8}"
          f"{'acc%':>7}{'rev%':>7}{'rej%':>7}")
    for lang in ("en", "de"):
        items = [d for d in dataset if d.get("language") == lang]
        c = _decision_counts(items)
        n = sum(c.values()) or 1
        print(f"{lang:<10}{sum(c.values()):>6}{c['accept']:>8}{c['review']:>8}{c['reject']:>8}"
              f"{c['accept']/n:>7.0%}{c['review']/n:>7.0%}{c['reject']/n:>7.0%}")

    print("\n(b) Type-pair P(connected) from pilot graphs")
    print(format_table(connectivity(dataset), min_count=connectivity_min_count(len(dataset))))

    print("\n(c) Example entries: see report.md section 'Example entries'. "
          "Sample IDs chosen:")
    for lang in ("en", "de"):
        for label, styles in (("short", SHORT_STYLES), ("long", LONG_STYLES)):
            d = _pick_example(dataset, lang, styles)
            if d:
                print(f"    {lang} {label}: {d['sample_id']} "
                      f"({d['participant'].get('writing_style')}, {d['scenario']}, {d['decision']})")

    pilot_cost = usage_payload["cost_usd"]
    per_sample = pilot_cost / len(dataset) if dataset else 0.0
    print("\n(d) Cost")
    print(f"    Pilot: ${pilot_cost:.4f} for {len(dataset)} samples "
          f"({usage_payload['total_tokens']:,} tokens, {usage_payload['llm_calls']:,} calls)")
    print(f"    Extrapolated full run ({full_samples:,} samples): "
          f"~${per_sample * full_samples:,.2f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None,
                        help="override total sample count (participants = n // days)")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run without regenerating done samples")
    parser.add_argument("--pilot", action="store_true",
                        help="run a small pilot (10 en + 10 de participants) into the pilot output dir")
    parser.add_argument("--languages", type=str, default=None,
                        help="comma-separated language filter (e.g. 'de'): only run samples "
                             "of these languages; graphs stay identical to the unfiltered run")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="override the output directory")
    args = parser.parse_args()

    config = load_config()
    n_days = config["days_per_participant"]

    lang_counts = None
    if args.pilot:
        n_participants = config.get("pilot_participants", 20)
        out_dir = BASE / config.get("pilot_output_dir", "output_pilot")
    else:
        if "n_participants_en" in config or "n_participants_de" in config:
            lang_counts = {"en": int(config.get("n_participants_en", 0)),
                           "de": int(config.get("n_participants_de", 0))}
            n_participants = lang_counts["en"] + lang_counts["de"]
        else:
            n_participants = config["n_participants"]
        out_dir = BASE / config["output_dir"]
    if args.n is not None:
        n_participants = max(1, args.n // n_days)
        lang_counts = None
    if args.out_dir:
        out_dir = BASE / args.out_dir
    out_dir.mkdir(exist_ok=True)

    graphs_path = out_dir / "graphs.jsonl"
    dataset_path = out_dir / "dataset.jsonl"
    verification_path = out_dir / "verification.jsonl"
    report_path = out_dir / "report.md"
    usage_path = out_dir / "usage.json"

    pool_path = (BASE / config["variable_pool_path"]).resolve()
    samples = generate_dataset(str(pool_path), n_participants, n_days, config["seed"],
                               language_counts=lang_counts)
    if args.languages:
        wanted = {x.strip() for x in args.languages.split(",") if x.strip()}
        samples = [s for s in samples if s["participant"]["language"] in wanted]
    for s in samples:
        s["generator_version"] = config["generator_version"]
    n = len(samples)
    if "n_participants_en" in config or "n_participants_de" in config:
        full_participants = (int(config.get("n_participants_en", 0))
                             + int(config.get("n_participants_de", 0)))
    else:
        full_participants = config["n_participants"]
    full_samples = full_participants * n_days

    workers = int(os.environ.get(config.get("concurrency_env", "PIPELINE_CONCURRENCY"),
                                 config["concurrency"]))

    # Resume safety: existing graphs must match sample_ids, generator_version, AND
    # graph content (hash). Any drift -> wipe and regenerate so old and new samples
    # never mix.
    existing_graphs = read_jsonl(graphs_path)
    version_ok = (existing_graphs
                  and existing_graphs[0].get("generator_version") == config["generator_version"])
    matches = (version_ok
               and len(existing_graphs) == n
               and all(a["sample_id"] == b["sample_id"]
                       and graph_hash(a["graph"]) == graph_hash(b["graph"])
                       for a, b in zip(existing_graphs, samples)))

    if not (args.resume and matches):
        if args.resume and not matches:
            print("Resume requested but existing graphs do not match current "
                  "generator (version/content drift) — regenerating from scratch.",
                  flush=True)
        for p in (graphs_path, dataset_path, verification_path, report_path, usage_path):
            if p.exists():
                p.unlink()
        for s in samples:
            append_jsonl(graphs_path, s)

    dataset = read_jsonl(dataset_path)
    done = {d["sample_id"] for d in dataset}
    pending = [s for s in samples if s["sample_id"] not in done]

    lock = threading.Lock()
    pricing = config["pricing"]
    # cost accounting is reconstructed from the per-sample usage stored on each
    # completed record, so a resumed run's totals stay exact regardless of when
    # the previous run was interrupted.
    base_usage = usage_from_records(dataset)
    usage_state = {
        "prompt_tokens": base_usage["prompt_tokens"],
        "completion_tokens": base_usage["completion_tokens"],
        "calls": base_usage["calls"],
        "samples": len(done),
    }
    flush_every = config.get("usage_flush_every", 5)

    total = len(samples)
    completed = len(done)
    processed_this_run = 0
    failures = 0
    run_start = time.time()

    def handle(sample):
        # never let one sample's terminal LLM error abort the whole run; it stays
        # unwritten (pending) so a later --resume retries it.
        try:
            return sample, process_one(config, sample), None
        except Exception as e:  # noqa: BLE001 - deliberately broad for run resilience
            return sample, None, f"{type(e).__name__}: {e}"

    mode = "PILOT" if args.pilot else "FULL"
    print(f"[{mode}] Running {total} samples ({n_participants} participants x {n_days} days), "
          f"{len(pending)} pending, {workers} workers", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(handle, s) for s in pending]
        for future in as_completed(futures):
            sample, result, err = future.result()
            with lock:
                processed_this_run += 1
                elapsed = time.time() - run_start
                rate = processed_this_run / elapsed if elapsed > 0 else 0.0
                remaining = len(pending) - processed_this_run
                eta = fmt_hms(remaining / rate) if rate > 0 else "?"
                if err is not None:
                    failures += 1
                    print(f"[{completed}/{total}] {sample['sample_id']} FAILED "
                          f"({err}) — left pending for --resume | {rate:.1f}/s | ETA {eta}",
                          flush=True)
                    continue
                record, verification_entry, usage_acc = result
                append_jsonl(dataset_path, record)
                append_jsonl(verification_path, verification_entry)
                dataset.append(record)
                completed += 1
                usage_state["prompt_tokens"] += usage_acc["prompt_tokens"]
                usage_state["completion_tokens"] += usage_acc["completion_tokens"]
                usage_state["calls"] += usage_acc["calls"]
                usage_state["samples"] = completed
                cost = compute_cost(usage_state["prompt_tokens"],
                                    usage_state["completion_tokens"], pricing)
                if processed_this_run % flush_every == 0 or processed_this_run == len(pending):
                    write_usage(usage_path, usage_state, pricing, config["model"])
                print(f"[{completed}/{total}] {sample['sample_id']} "
                      f"({sample['scenario']}/{record['language']}) -> {record['decision']} "
                      f"| ${cost:.4f} | {rate:.1f}/s | ETA {eta}",
                      flush=True)

    usage_payload = write_usage(usage_path, usage_state, pricing, config["model"])

    def sort_file(path, key):
        # atomic rewrite (tmp + os.replace) so an interruption during the final
        # sort can never leave a truncated .jsonl that would break resume
        items = sorted(read_jsonl(path), key=key)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
        return items

    sort_file(graphs_path, lambda x: (x["participant"]["participant_id"], x["entry"]["day_index"]))
    sort_file(verification_path, lambda x: x["sample_id"])
    dataset = sort_file(dataset_path,
                        lambda x: (x["participant"]["participant_id"], x["day_index"]))
    write_report(config, dataset, out_dir, usage_payload=usage_payload, full_samples=full_samples)

    print(f"\nDone. {len(dataset)}/{total} samples.")
    if failures:
        print(f"WARNING: {failures} sample(s) failed after retries and were left "
              f"unwritten. Re-run with --resume to retry only those.")
    print(f"Outputs in: {out_dir}")
    print(f"Report: {report_path}")
    print(f"Usage: {usage_path}  (cost ${usage_payload['cost_usd']:.4f})")

    if args.pilot:
        print_pilot_summary(config, dataset, usage_payload, full_samples)


if __name__ == "__main__":
    main()
