import { ExtractResponse, HONESTY, JournalDay, ServerEdge, ServerNode } from "./types";

type DayExtract = { day: JournalDay; payload: ExtractResponse };

export type AggregationResult = {
  markdown: string;
  succeeded: number;
  skipped: number;
};

const PROBLEM_TYPES = new Set(["stressor", "symptom", "emotion", "thought", "event"]);
const RELIEF_EDGE_TYPES = new Set(["decreases", "reduces", "relieves"]);

type ConceptMeta = { type: string; label: string; polarity: string };

// ── small numeric helpers ──────────────────────────────────────────────────
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
const stddev = (xs: number[]) => {
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
};
const signed = (x: number, digits = 1) => `${x >= 0 ? "+" : ""}${x.toFixed(digits)}`;

const dayMs = 86400000;
const toUtc = (date: string) => {
  const [y, m, d] = date.split("-").map(Number);
  return Date.UTC(y, m - 1, d);
};
const daysBetween = (a: string, b: string) => Math.round((toUtc(b) - toUtc(a)) / dayMs);
const nextDate = (date: string) => new Date(toUtc(date) + dayMs).toISOString().slice(0, 10);
const WEEKDAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
const weekday = (date: string) => WEEKDAYS[new Date(toUtc(date)).getUTCDay()];

export function aggregate(items: DayExtract[], skipped: number): AggregationResult {
  const sorted = [...items].sort((a, b) => (a.day.date < b.day.date ? -1 : 1));
  const n = sorted.length;
  const md: string[] = [];

  // ── shared indexes ───────────────────────────────────────────────────────
  const conceptMeta = new Map<string, ConceptMeta>();
  const conceptDays = new Map<string, Set<string>>(); // concept_id -> distinct dates
  const dayConcepts = new Map<string, Set<string>>(); // date -> concept_ids
  const moodByDate = new Map<string, number>();
  const dates: string[] = [];

  for (const it of sorted) {
    const date = it.day.date;
    if (!dates.includes(date)) dates.push(date);
    if (typeof it.day.mood_score === "number" && !moodByDate.has(date)) {
      moodByDate.set(date, it.day.mood_score);
    }
    for (const node of it.payload.nodes) {
      if (!conceptMeta.has(node.concept_id)) {
        conceptMeta.set(node.concept_id, {
          type: node.type,
          label: node.label || node.concept_id,
          polarity: node.polarity ?? "neutral",
        });
      }
      (conceptDays.get(node.concept_id) ?? conceptDays.set(node.concept_id, new Set()).get(node.concept_id)!).add(date);
      (dayConcepts.get(date) ?? dayConcepts.set(date, new Set()).get(date)!).add(node.concept_id);
    }
  }

  const label = (cid: string) => conceptMeta.get(cid)?.label ?? cid;
  const moodValues = [...moodByDate.values()];
  const baselineMood = moodValues.length >= 2 ? mean(moodValues) : null;
  const totalDays = dates.length;
  const halfCut = Math.floor(totalDays / 2);
  const firstHalfDates = new Set(dates.slice(0, halfCut));
  // second half = dates[halfCut:]; membership test is !firstHalfDates.has(d)

  const isProblem = (cid: string) => {
    const meta = conceptMeta.get(cid);
    return !!meta && PROBLEM_TYPES.has(meta.type) && meta.polarity !== "positive";
  };
  const isCoping = (cid: string) => {
    const meta = conceptMeta.get(cid);
    return !!meta && (meta.type === "coping_action" || (meta.type === "activity" && meta.polarity === "positive"));
  };

  // ── header ───────────────────────────────────────────────────────────────
  const first = dates[0];
  const last = dates[dates.length - 1];
  md.push(`# Therapist summary — ${first} → ${last}`);
  md.push("");
  md.push("> [!info] About this document");
  md.push(`> ${HONESTY} All figures below are associations observed in journal text, not clinical measurements.`);
  md.push("");

  // ── 1. Overview ──────────────────────────────────────────────────────────
  md.push("## 1. Overview");
  md.push("");
  const spanDays = daysBetween(first, last) + 1;
  md.push("| | |");
  md.push("| --- | --- |");
  md.push(
    `| Entries | ${n}${totalDays > 1 ? ` (${totalDays} distinct days over a ${spanDays}-day span)` : ""} |`,
  );
  md.push(`| Period | ${first} → ${last} |`);
  if (skipped) {
    md.push(`| Skipped | ${skipped} note${skipped === 1 ? "" : "s"} (errors) |`);
  }
  if (totalDays >= 2) {
    let maxGap = 0;
    let gapAfter = first;
    for (let i = 1; i < dates.length; i++) {
      const g = daysBetween(dates[i - 1], dates[i]);
      if (g > maxGap) {
        maxGap = g;
        gapAfter = dates[i - 1];
      }
    }
    md.push(
      `| Longest gap | ${
        maxGap > 1
          ? `${maxGap - 1} day${maxGap === 2 ? "" : "s"} without an entry, after ${gapAfter}`
          : "none — entries are consecutive"
      } |`,
    );
  }
  if (baselineMood !== null) {
    const moodDates = [...moodByDate.keys()];
    const firstMood = moodByDate.get(moodDates[0])!;
    const lastMood = moodByDate.get(moodDates[moodDates.length - 1])!;
    const net = lastMood - firstMood;
    const sd = stddev(moodValues);
    const volatility = sd < 0.8 ? "low" : sd < 1.6 ? "moderate" : "high";
    const arrow = net > 0.5 ? "↗" : net < -0.5 ? "↘" : "→";
    let best = moodDates[0];
    let worst = moodDates[0];
    for (const d of moodDates) {
      if (moodByDate.get(d)! > moodByDate.get(best)!) best = d;
      if (moodByDate.get(d)! < moodByDate.get(worst)!) worst = d;
    }
    md.push(
      `| Mood average | ${baselineMood.toFixed(1)} (${moodValues.length} of ${n} entries scored) |`,
    );
    md.push(
      `| Mood trend | ${arrow} ${signed(net)} (${firstMood.toFixed(1)} → ${lastMood.toFixed(1)}), ${volatility} volatility |`,
    );
    md.push(
      `| Best / worst day | ${best} (${moodByDate.get(best)!.toFixed(1)}) / ${worst} (${moodByDate.get(worst)!.toFixed(1)}) |`,
    );
  } else {
    md.push("| Mood | no usable mood data |");
  }
  md.push("");

  // ── 2. Recurring problem patterns ────────────────────────────────────────
  md.push("## 2. Recurring problem patterns");
  md.push("");
  const threshold = totalDays < 5 ? 2 : Math.max(3, Math.ceil(0.2 * totalDays));
  const recurring = [...conceptDays.entries()]
    .filter(([cid, ds]) => isProblem(cid) && ds.size >= threshold)
    .sort((a, b) => b[1].size - a[1].size);
  if (recurring.length === 0) {
    md.push(`_No stressor/symptom/emotion recurred on ${threshold}+ days — nothing crossed the recurrence threshold._`);
  } else {
    md.push(`_Threshold: seen on at least ${threshold} of ${totalDays} days._`);
    md.push("");
    for (const [cid, ds] of recurring.slice(0, 8)) {
      const meta = conceptMeta.get(cid)!;
      const dList = [...ds].sort();
      md.push(`### ${meta.label} (${meta.type})`);
      md.push(`- Appears on **${ds.size} of ${totalDays} days** (${dList[0]} → ${dList[dList.length - 1]}).`);
      // Same-day co-occurring concepts (any type except itself).
      const co = new Map<string, number>();
      for (const d of ds) {
        for (const other of dayConcepts.get(d) ?? []) {
          if (other !== cid) co.set(other, (co.get(other) ?? 0) + 1);
        }
      }
      const topCo = [...co.entries()]
        .filter(([, c]) => c >= 2)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);
      if (topCo.length) {
        md.push(
          `- On those days, also present: ${topCo.map(([o, c]) => `${label(o)} (${c} day${c === 1 ? "" : "s"})`).join(", ")}.`,
        );
      }
      // First-half vs second-half trend.
      if (totalDays >= 4) {
        const firstCount = dList.filter((d) => firstHalfDates.has(d)).length;
        const secondCount = ds.size - firstCount;
        const secondHalfSize = totalDays - halfCut;
        const trend =
          secondCount > firstCount ? "**increasing**" : secondCount < firstCount ? "**decreasing**" : "stable";
        md.push(
          `- Trend: ${firstCount}/${halfCut} days in the first half vs ${secondCount}/${secondHalfSize} in the second — ${trend} over the covered period.`,
        );
      }
      md.push("");
    }
  }
  md.push("");

  // ── 3. Recurring causal chains ───────────────────────────────────────────
  md.push("## 3. Recurring chains");
  md.push("");
  // Enumerate 3-node paths within each day's graph, keyed by concept sequence.
  const chainDays = new Map<string, Set<string>>();
  const chainRelations = new Map<string, string[]>();
  for (const it of sorted) {
    const nodeById = new Map<string, ServerNode>();
    for (const node of it.payload.nodes) nodeById.set(node.node_id, node);
    const adj = new Map<string, { edge: ServerEdge; target: ServerNode }[]>();
    for (const e of it.payload.edges) {
      const target = nodeById.get(e.target_node_id);
      if (!nodeById.has(e.source_node_id) || !target) continue;
      (adj.get(e.source_node_id) ?? adj.set(e.source_node_id, []).get(e.source_node_id)!).push({ edge: e, target });
    }
    let pathsThisDay = 0;
    for (const a of it.payload.nodes) {
      if (pathsThisDay >= 50) break;
      for (const step1 of adj.get(a.node_id) ?? []) {
        if (pathsThisDay >= 50) break;
        for (const step2 of adj.get(step1.target.node_id) ?? []) {
          const [b, c] = [step1.target, step2.target];
          if (c.concept_id === a.concept_id || c.concept_id === b.concept_id || a.concept_id === b.concept_id) continue;
          const key = [a.concept_id, b.concept_id, c.concept_id].join("→");
          (chainDays.get(key) ?? chainDays.set(key, new Set()).get(key)!).add(it.day.date);
          if (!chainRelations.has(key)) chainRelations.set(key, [step1.edge.type, step2.edge.type]);
          if (++pathsThisDay >= 50) break;
        }
      }
    }
  }
  const minChainDays = totalDays < 5 ? 1 : 2;
  const topChains = [...chainDays.entries()]
    .filter(([, ds]) => ds.size >= minChainDays)
    .sort((a, b) => b[1].size - a[1].size)
    .slice(0, 5);
  if (topChains.length === 0) {
    md.push("_No multi-step chain (A → B → C) repeated across days in the extracted graphs._");
    const examples = [...chainDays.entries()].slice(0, 3);
    if (examples.length) {
      md.push("");
      md.push("_Single-day examples (not recurring):_");
      md.push("");
      for (const [key] of examples) {
        const [a, b, c] = key.split("→");
        const rels = chainRelations.get(key) ?? [];
        md.push(`- ${label(a)} → ${label(b)} → ${label(c)} (relations: ${rels.join(", ")})`);
      }
    }
  } else {
    md.push("_Most frequent multi-step paths in the extracted day graphs (edge relation types are heuristic):_");
    md.push("");
    for (const [key, ds] of topChains) {
      const [a, b, c] = key.split("→");
      const rels = chainRelations.get(key) ?? [];
      md.push(
        `- **${label(a)} → ${label(b)} → ${label(c)}** — on ${ds.size} day${ds.size === 1 ? "" : "s"} (relations: ${rels.join(", ")})`,
      );
    }
  }
  md.push("");

  // ── 4. Coping analysis ───────────────────────────────────────────────────
  md.push("## 4. Coping analysis");
  md.push("");
  const copingConcepts = [...conceptDays.entries()]
    .filter(([cid, ds]) => isCoping(cid) && ds.size >= 2)
    .sort((a, b) => b[1].size - a[1].size);
  const anyCoping = [...conceptMeta.keys()].some(isCoping);
  if (!anyCoping) {
    md.push("_No coping actions or positive activities were detected in these entries._");
  } else if (copingConcepts.length === 0) {
    md.push("_Coping actions appear, but none on 2+ days — too little repetition to analyse. They are listed in the appendix._");
  } else {
    md.push("_All mood comparisons are associations on the days an activity was mentioned — not evidence of causation._");
    md.push("");
    const noAssociation: string[] = [];
    for (const [cid, ds] of copingConcepts) {
      const meta = conceptMeta.get(cid)!;
      md.push(`### ${meta.label}`);
      md.push(`- Mentioned on **${ds.size} of ${totalDays} days**.`);

      // What it is used against: same-day relief edges from this concept.
      const reliefTargets = new Map<string, number>();
      const anyTargets = new Map<string, number>();
      for (const it of sorted) {
        if (!ds.has(it.day.date)) continue;
        const nodeById = new Map<string, ServerNode>();
        for (const node of it.payload.nodes) nodeById.set(node.node_id, node);
        for (const e of it.payload.edges) {
          const src = nodeById.get(e.source_node_id);
          const tgt = nodeById.get(e.target_node_id);
          if (!src || !tgt || src.concept_id !== cid || tgt.concept_id === cid) continue;
          anyTargets.set(tgt.concept_id, (anyTargets.get(tgt.concept_id) ?? 0) + 1);
          if (RELIEF_EDGE_TYPES.has(e.type)) {
            reliefTargets.set(tgt.concept_id, (reliefTargets.get(tgt.concept_id) ?? 0) + 1);
          }
        }
      }
      const fmtTargets = (m: Map<string, number>) =>
        [...m.entries()]
          .sort((a, b) => b[1] - a[1])
          .slice(0, 5)
          .map(([t, c]) => `${label(t)} (${c}×)`)
          .join(", ");
      if (reliefTargets.size) {
        md.push(`- Used against (via "decreases" edges): ${fmtTargets(reliefTargets)}.`);
      } else if (anyTargets.size) {
        md.push(`- No "decreases" edges found; it co-occurs (connected in the graph) with: ${fmtTargets(anyTargets)}.`);
      } else {
        md.push("- Not connected to other concepts in the extracted graphs.");
      }

      // Same-day mood vs baseline.
      let sameDayDelta: number | null = null;
      const moodsWith = [...ds].filter((d) => moodByDate.has(d)).map((d) => moodByDate.get(d)!);
      if (baselineMood !== null && moodsWith.length >= 2) {
        const avgWith = mean(moodsWith);
        sameDayDelta = avgWith - baselineMood;
        md.push(
          `- On days with ${meta.label}, mood averaged **${avgWith.toFixed(1)}** vs a baseline of ${baselineMood.toFixed(1)} — Δ ${signed(sameDayDelta)} (n=${moodsWith.length}).`,
        );
      } else {
        md.push("- Not enough mood data on those days for a same-day comparison.");
      }

      // Next-day mood shift (only when the next calendar day has an entry with mood).
      const nextDeltas: number[] = [];
      for (const d of ds) {
        const nd = nextDate(d);
        if (moodByDate.has(d) && moodByDate.has(nd)) nextDeltas.push(moodByDate.get(nd)! - moodByDate.get(d)!);
      }
      if (nextDeltas.length >= 2) {
        md.push(`- Next-day mood shift: **${signed(mean(nextDeltas))}** on average (n=${nextDeltas.length}).`);
      }

      if (sameDayDelta !== null && Math.abs(sameDayDelta) < 0.3) {
        noAssociation.push(`${meta.label} (Δ ${signed(sameDayDelta)}, ${ds.size} days)`);
      }
      md.push("");
    }
    if (noAssociation.length) {
      md.push("**Coping attempts with no clear mood association:**");
      md.push("");
      for (const s of noAssociation) md.push(`- ${s}`);
      md.push("");
      md.push("_“No association” here only means mood on those days was close to baseline — it says nothing about subjective relief._");
    }
  }
  md.push("");

  // ── 5. Possible discussion points ────────────────────────────────────────
  md.push("## 5. Possible discussion points");
  md.push("");
  type Candidate = { strength: number; text: string };
  const candidates: Candidate[] = [];

  // (a) Increasing problem patterns.
  if (totalDays >= 4) {
    for (const [cid, ds] of recurring) {
      const firstCount = [...ds].filter((d) => firstHalfDates.has(d)).length;
      const secondCount = ds.size - firstCount;
      if (secondCount >= firstCount * 1.5 && secondCount - firstCount >= 2) {
        candidates.push({
          strength: secondCount - firstCount + ds.size / totalDays,
          text: `**${label(cid)}** appears more often in the second half of the period (${firstCount} → ${secondCount} days). Did something change around then?`,
        });
      }
    }
  }
  // (b) Problem concepts that never share a day with any coping action.
  const copingDates = new Set<string>();
  for (const [cid, ds] of conceptDays) if (isCoping(cid)) for (const d of ds) copingDates.add(d);
  if (copingDates.size > 0) {
    for (const [cid, ds] of recurring) {
      if ([...ds].every((d) => !copingDates.has(d))) {
        candidates.push({
          strength: ds.size / totalDays + 0.5,
          text: `**${label(cid)}** (${ds.size} days) never shares a day with any coping action in these entries. Is there something they have tried for it?`,
        });
      }
    }
  }
  // (c) Strongest recurring chain.
  if (topChains.length && topChains[0][1].size >= 2) {
    const [key, ds] = topChains[0];
    const [a, b, c] = key.split("→");
    candidates.push({
      strength: ds.size / totalDays + 0.4,
      text: `The sequence **${label(a)} → ${label(b)} → ${label(c)}** recurs on ${ds.size} days. Do they experience it as a chain, and where could it be interrupted?`,
    });
  }
  // (d) Weekday with notably low mood.
  if (baselineMood !== null && moodByDate.size >= 6) {
    const byWd = new Map<string, number[]>();
    for (const [d, m] of moodByDate) {
      const wd = weekday(d);
      (byWd.get(wd) ?? byWd.set(wd, []).get(wd)!).push(m);
    }
    for (const [wd, moods] of byWd) {
      if (moods.length >= 2 && baselineMood - mean(moods) >= 1.0) {
        candidates.push({
          strength: (baselineMood - mean(moods)) / 2,
          text: `**${wd}s** average mood ${mean(moods).toFixed(1)} vs ${baselineMood.toFixed(1)} overall (${moods.length} ${wd}s). Is there something specific about that day of the week?`,
        });
      }
    }
  }
  // (e) Weekday where stressors concentrate.
  const stressorDates: string[] = [];
  for (const [cid, ds] of conceptDays) {
    if (conceptMeta.get(cid)?.type === "stressor") stressorDates.push(...ds);
  }
  if (stressorDates.length >= 6 && totalDays >= 7) {
    const wdCount = new Map<string, number>();
    for (const d of stressorDates) wdCount.set(weekday(d), (wdCount.get(weekday(d)) ?? 0) + 1);
    const [topWd, topCount] = [...wdCount.entries()].sort((a, b) => b[1] - a[1])[0];
    if (topCount >= stressorDates.length * 0.4 && wdCount.size > 1) {
      candidates.push({
        strength: topCount / stressorDates.length,
        text: `Stressor mentions cluster on **${topWd}s** (${topCount} of ${stressorDates.length}). Worth asking what that day typically holds?`,
      });
    }
  }

  if (candidates.length === 0) {
    md.push("_Not enough signal in these entries to generate discussion points._");
  } else {
    md.push("_Auto-generated prompts from the data above — observations to explore, not conclusions or advice._");
    md.push("");
    let pointNo = 1;
    for (const cand of candidates.sort((a, b) => b.strength - a.strength).slice(0, 5)) {
      md.push(`> [!question] Discussion point ${pointNo}`);
      md.push(`> ${cand.text}`);
      md.push("");
      pointNo += 1;
    }
  }
  md.push("");

  // ── 6. Appendix: per-day reference ───────────────────────────────────────
  md.push("## 6. Appendix — per-day reference");
  md.push("");
  md.push("| date | mood | top concepts |");
  md.push("| ---- | ---- | ------------ |");
  for (const it of sorted) {
    const mood = typeof it.day.mood_score === "number" ? it.day.mood_score.toFixed(1) : "—";
    const seen = new Set<string>();
    const top: string[] = [];
    for (const node of [...it.payload.nodes]
      .filter((nd) => nd.type !== "unknown")
      .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0))) {
      if (seen.has(node.concept_id)) continue;
      seen.add(node.concept_id);
      top.push(node.label || node.concept_id);
      if (top.length >= 5) break;
    }
    md.push(`| ${it.day.date} | ${mood} | ${top.join(", ") || "—"} |`);
  }
  md.push("");

  return { markdown: md.join("\n"), succeeded: items.length, skipped };
}

export type { DayExtract };
