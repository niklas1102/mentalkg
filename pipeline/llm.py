import json
import re
import time

import requests


def chat(config, messages, temperature):
    """Call the chat endpoint. Returns (content, usage) where usage is
    {"prompt_tokens": int, "completion_tokens": int} (zeros if absent)."""
    payload = {
        "model": config["model"],
        "messages": messages,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {config['openrouter_api_key']}",
        "Content-Type": "application/json",
    }
    last_error = None
    for attempt in range(config["max_retries"]):
        try:
            response = requests.post(
                config["base_url"],
                headers=headers,
                json=payload,
                timeout=config["request_timeout"],
            )
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                raw_usage = data.get("usage") or {}
                usage = {
                    "prompt_tokens": int(raw_usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(raw_usage.get("completion_tokens") or 0),
                }
                return content, usage
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
        except requests.RequestException as e:
            last_error = str(e)
        time.sleep(config["retry_backoff_seconds"] * (attempt + 1))
    raise RuntimeError(f"LLM request failed after retries: {last_error}")


# Connector phrases that give away the underlying graph. Checked per language.
BANNED_PHRASES_EN = ["linked to", "caused by", "led to", "connected to",
                     "connected with", "increased my", "decreased my", "eased my"]
# German multi-word connectors, matched whole-word.
BANNED_PHRASES_DE = ["führte zu", "verursacht durch", "verbunden mit",
                     "hängt zusammen mit", "verstärkte mein", "verringerte mein"]
# German clinical / connector stems, matched allowing inflectional suffixes so
# plurals and declensions are caught too (Symptom -> Symptome/Symptomen,
# Stressor -> Stressoren, linderte -> lindert/Linderung, Auslöser -> Auslösern).
BANNED_STEMS_DE = ["symptom", "stressor", "auslöser", "linder"]
# German graph-structure nouns that are also ordinary words; matched whole-word
# only to avoid false positives on unrelated compounds.
BANNED_WORDS_DE = ["knoten", "kante"]

# German letters incl. ß (U+00DF, which sits just below à=U+00E0 and would
# otherwise be treated as a word boundary, e.g. spuriously matching 'kante' in
# 'Fußkante').
_DE_LETTER = "a-zßà-ÿ"


def contains_banned(text, language="en"):
    low = text.lower()
    found = []
    if language == "de":
        for p in BANNED_PHRASES_DE + BANNED_WORDS_DE:
            if re.search(rf"(?<![{_DE_LETTER}]){re.escape(p)}(?![{_DE_LETTER}])", low):
                found.append(p)
        for stem in BANNED_STEMS_DE:
            if re.search(rf"(?<![{_DE_LETTER}]){re.escape(stem)}[{_DE_LETTER}]*", low):
                found.append(stem)
    else:
        for p in BANNED_PHRASES_EN:
            if re.search(r"(?<![a-z])" + re.escape(p) + r"(?![a-z])", low):
                found.append(p)
    return found


# Per-writing-style target word range (English). Keys match WRITING_STYLES in
# scenarios.py. German ranges are these clamped to the denser 80-160 band.
STYLE_WORDCOUNTS = {
    "terse_fragmented": (60, 110),
    "matter_of_fact": (80, 140),
    "reflective": (110, 180),
    "emotional": (100, 180),
    "rambling": (140, 200),
}

STYLE_DESC_EN = {
    "terse_fragmented": "Write tersely, in short clipped sentences and sometimes bare "
                        "fragments. Little elaboration.",
    "matter_of_fact": "Write plainly and matter-of-factly: straightforward, unfussy, "
                      "just what happened and how it felt.",
    "reflective": "Write reflectively: thoughtful, turning things over, noticing how "
                  "one thing fed into another.",
    "emotional": "Write in an emotionally expressive way: feelings close to the "
                 "surface, vivid and heartfelt.",
    "rambling": "Write in a rambling, run-on way: thoughts spill out with tangents "
                "and long, loosely connected sentences.",
}

STYLE_DESC_DE = {
    "terse_fragmented": "Schreib knapp, in kurzen, abgehackten Sätzen, manchmal nur "
                        "Fragmente. Kaum Ausschmückung.",
    "matter_of_fact": "Schreib sachlich und schnörkellos: einfach, was passiert ist "
                      "und wie es sich angefühlt hat.",
    "reflective": "Schreib nachdenklich: du drehst die Dinge in Gedanken hin und her "
                  "und merkst, wie eins ins andere gegriffen hat.",
    "emotional": "Schreib gefühlsbetont: die Emotionen liegen offen, lebendig und "
                 "von Herzen.",
    "rambling": "Schreib weitschweifig: die Gedanken sprudeln raus, mit Abschweifungen "
                "und langen, lose verbundenen Sätzen.",
}


RELATION_WORDS = {
    "causes": "because of",
    "increases": "which made worse",
    "decreases": "which helped with",
    "follows": "after",
    "linked_to": "around the same time as",
}


def graph_for_prompt(sample):
    nodes = []
    for n in sample["graph"]["nodes"]:
        nodes.append({
            "node_id": n["node_id"],
            "type": n["type"],
            "label": n["label"],
            "time_anchor": n["time_anchor"]["text"],
            "temporal_status": n["temporal_status"],
            "polarity": n["polarity"],
        })
    edges = []
    label_by_id = {n["node_id"]: n["label"] for n in sample["graph"]["nodes"]}
    for e in sample["graph"]["edges"]:
        verb = RELATION_WORDS.get(e["type"], e["type"])
        edges.append({
            "edge_id": e["edge_id"],
            "relation": f"{label_by_id[e['source_node_id']]} {verb} {label_by_id[e['target_node_id']]}",
            "source_node_id": e["source_node_id"],
            "target_node_id": e["target_node_id"],
        })
    return {"nodes": nodes, "edges": edges}


def _style_range(participant):
    style = participant.get("writing_style", "reflective")
    lo, hi = STYLE_WORDCOUNTS.get(style, (90, 170))
    if participant.get("language") == "de":
        lo, hi = max(80, lo), min(160, hi)
    return style, lo, hi


def _generation_messages_en(g, sample, participant, style, lo, hi):
    system = (
        "You write realistic, private first-person mental-health journal entries "
        "from a structured graph for a research dataset. The graph is the ground "
        "truth. You express its meaning in natural everyday language, never as "
        "clinical or technical terms."
    )
    traits = ""
    if participant.get("recurring_stressors"):
        traits += f" Recurring stressors: {', '.join(participant['recurring_stressors'])}."
    if participant.get("coping_style"):
        traits += f" Usual coping: {', '.join(participant['coping_style'])}."
    prev_ctx = sample.get("previous_day_context", "")
    prev_line = (f"\nPrevious-day context (for natural continuity tone only; do "
                 f"NOT add anything from it that is not in today's graph): {prev_ctx}\n"
                 if prev_ctx else "")
    style_line = STYLE_DESC_EN.get(style, "")

    user = f"""Write ONE journal entry for this person and day.

Participant: age {participant['age']}, {participant['gender']}, {participant['occupation']}.{traits}
Writing style: {style_line}
This is day {sample['entry']['day_index']} of an ongoing personal journal.{prev_line}

Graph nodes (express the meaning of each one, with the given timing):
{json.dumps(g['nodes'], ensure_ascii=False, indent=2)}

Graph relations (the text must make these connections feel natural):
{json.dumps(g['edges'], ensure_ascii=False, indent=2)}

Rules:
1. Output only the journal entry text. No title, no lists, no explanation.
2. First person, private journal tone. Informal, believable, not poetic, not clinical.
3. Express every node and every relation, but PARAPHRASE naturally. Do NOT copy
   the clinical labels verbatim. Write how a real person would actually say it.
   - "perceived social rejection" -> "I kept feeling like they didn't really want me around"
   - "emotional numbness" -> "I just felt kind of flat and disconnected all day"
   - "rumination" -> "I couldn't stop going over it in my head"
   - "fear of failure" -> "I keep thinking I'm going to mess this up"
   Show connections through the story, not with connector words. NEVER write
   any of: "linked to", "connected to", "connected with", "caused by",
   "led to", "eased", "increased my", "decreased my", "increased",
   "decreased", "node", "edge", "symptom", "stressor", or the label phrases
   themselves. Instead show cause and effect naturally, e.g. "with the
   deadline coming up, I couldn't stop worrying".
4. Keep each node's timing exactly as given: "today"/"now" = happening today,
   "tomorrow"/"later today" = has not happened yet and is only anticipated,
   "yesterday"/"last night" = already happened before today, "this morning" =
   earlier today. A "tomorrow" item must read as something expected, not as a
   fact that already occurred.
5. Do NOT add any other symptoms, events, stressors, emotions, thoughts,
   coping actions, or extra time references that are not in the graph.
6. No diagnostic labels (depression, anxiety disorder, PTSD, etc.).
7. No self-harm, suicide, abuse, or violence.
8. Between {lo} and {hi} words."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _generation_messages_de(g, sample, participant, style, lo, hi):
    system = (
        "Du schreibst realistische, private Tagebucheinträge in der Ich-Perspektive "
        "zum Thema psychische Gesundheit, auf Basis eines strukturierten Graphen, für "
        "einen Forschungsdatensatz. Der Graph ist die Grundwahrheit. Du drückst seine "
        "Bedeutung in natürlicher Alltagssprache aus, niemals in klinischen oder "
        "Fachbegriffen. Schreib auf Deutsch, in der Ich-Perspektive und im informellen "
        "Du-Register (lockere Umgangssprache, kein förmliches Sie, kein Amtsdeutsch), "
        "wie ein echter Mensch in seinem privaten Tagebuch schreiben würde."
    )
    traits = ""
    if participant.get("recurring_stressors"):
        traits += f" Wiederkehrende Belastungen: {', '.join(participant['recurring_stressors'])}."
    if participant.get("coping_style"):
        traits += f" Übliche Bewältigung: {', '.join(participant['coping_style'])}."
    prev_ctx = sample.get("previous_day_context", "")
    prev_line = (f"\nKontext vom Vortag (nur für einen natürlichen Übergangston; füg "
                 f"NICHTS daraus hinzu, das nicht im heutigen Graphen steht): {prev_ctx}\n"
                 if prev_ctx else "")
    style_line = STYLE_DESC_DE.get(style, "")

    user = f"""Schreib EINEN Tagebucheintrag für diese Person und diesen Tag.

Person: {participant['age']} Jahre, {participant['gender']}, {participant['occupation']}.{traits}
Schreibstil: {style_line}
Das ist Tag {sample['entry']['day_index']} eines fortlaufenden persönlichen Tagebuchs.{prev_line}

Graph-Knoten (drück die Bedeutung von jedem aus, mit dem angegebenen Zeitbezug):
{json.dumps(g['nodes'], ensure_ascii=False, indent=2)}

Graph-Beziehungen (der Text muss diese Zusammenhänge natürlich wirken lassen):
{json.dumps(g['edges'], ensure_ascii=False, indent=2)}

Regeln:
1. Gib nur den Tagebucheintrag aus. Kein Titel, keine Aufzählung, keine Erklärung.
2. Ich-Perspektive, privater Tagebuchton, innerer Monolog. Informell und
   umgangssprachlich (nicht förmlich, nicht gestelzt), glaubwürdig, nicht poetisch,
   nicht klinisch.
3. Drück jeden Knoten und jede Beziehung aus, aber PARAPHRASIER natürlich. Übernimm
   die klinischen Begriffe NICHT wörtlich. Schreib, wie ein echter Mensch es sagen würde.
   - "perceived social rejection" -> "ich hatte das Gefühl, dass sie mich eigentlich nicht dabeihaben wollten"
   - "emotional numbness" -> "ich war den ganzen Tag irgendwie leer und wie abgeschnitten"
   - "rumination" -> "ich hab die Sache einfach nicht aus dem Kopf gekriegt"
   - "fear of failure" -> "ich denke ständig, dass ich das in den Sand setze"
   Zeig Zusammenhänge über die Erzählung, nicht mit Verbindungswörtern. Schreib
   NIEMALS eines von: "führte zu", "verursacht durch", "verbunden mit", "hängt
   zusammen mit", "verstärkte mein", "verringerte mein", "linderte", "Knoten",
   "Kante", "Symptom", "Stressor", "Auslöser", oder die Label-Begriffe selbst.
   Zeig Ursache und Wirkung stattdessen natürlich, z.B. "mit der Deadline im Nacken
   konnte ich nicht aufhören, mir Sorgen zu machen". Auch Linderung und Abfolge
   erzählerisch zeigen: "nach dem Spaziergang war die Anspannung ein Stück weit weg"
   (etwas hat geholfen), "nach der miesen Nacht war ich den ganzen Tag erschöpft"
   (eins folgte auf das andere).
4. Halt den Zeitbezug jedes Knotens genau ein und mach ihn im Text EXPLIZIT mit
   dem passenden Wort fest:
   - "today" -> "heute"; "now" -> "jetzt" / "gerade"
   - "yesterday" -> unbedingt "gestern" schreiben (z.B. "gestern hatte ich...",
     "seit gestern"); "last night" -> "letzte Nacht" / "gestern Abend"
   - "this morning" -> "heute Morgen" / "heute früh"; "this afternoon" ->
     "heute Nachmittag" / "nachmittags"
   - "tomorrow" -> "morgen", und zwar als etwas Erwartetes, nicht als schon geschehen
   Vage Zeitangaben wie "in letzter Zeit", "die letzten Tage" oder zeitloses
   Präsens ("ich merke, dass ich mehr esse") reichen NICHT — anker jedes Element
   klar an seinem Zeitpunkt.
5. Füg KEINE weiteren Symptome, Ereignisse, Belastungen, Gefühle (z.B. kein
   zusätzliches "es ist frustrierend"), Gedanken, Bewältigungsversuche oder
   zusätzlichen Zeitbezüge hinzu, die nicht im Graphen stehen.
6. WICHTIG: Beende den Eintrag NICHT mit einer Hoffnung oder einem Ausblick auf
   morgen oder die Zukunft (also NICHT "Ich hoffe, dass es morgen besser wird",
   "Hoffentlich wird morgen leichter", "mal sehen, was morgen bringt" o.ä.) —
   außer der Graph enthält einen Knoten mit Zeitbezug "tomorrow". Ohne einen
   solchen Knoten darf das Wort "morgen" (= der nächste Tag) gar nicht vorkommen.
   Hör stattdessen einfach bei der Gegenwart auf.
7. Keine Diagnosebegriffe (Depression, Angststörung, PTBS usw.).
8. Nichts über Selbstverletzung, Suizid, Missbrauch oder Gewalt.
9. Zwischen {lo} und {hi} Wörter."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_generation_messages(sample):
    g = graph_for_prompt(sample)
    participant = sample["participant"]
    style, lo, hi = _style_range(participant)
    if participant.get("language") == "de":
        return _generation_messages_de(g, sample, participant, style, lo, hi)
    return _generation_messages_en(g, sample, participant, style, lo, hi)


def build_verification_messages(sample, generated_text):
    g = graph_for_prompt(sample)
    language = sample["participant"].get("language", "en")
    lang_note = ""
    if language == "de":
        lang_note = """
NOTE: The journal entry is written in GERMAN. Judge it against the (English)
graph BY MEANING. Evidence quotes must be copied verbatim from the German text.

German coverage guidance — idiomatic or implicit German expression COUNTS as
coverage; do not require literal translations of the English labels:
- emotion: "stress" ~ "alles wächst mir über den Kopf", "loneliness" ~ "ich fühlte
  mich ganz allein gelassen", "relief" ~ "mir ist ein Stein vom Herzen gefallen"
- symptom: "racing thoughts" ~ "meine Gedanken rasen / kreisen ständig",
  "fatigue" ~ "ich bin völlig erschöpft / kaputt"
- thought: "rumination" ~ "ich krieg die Sache nicht aus dem Kopf",
  "not good enough" ~ "ich hab das Gefühl, nicht gut genug zu sein"
- stressor: "deadline pressure" ~ "die Abgabe sitzt mir im Nacken"
- event: "argument" ~ "wir sind aneinandergeraten / hatten Streit"
- activity/coping: "walk" ~ "eine Runde um den Block gedreht",
  "talking to a friend" ~ "mit einer Freundin geredet, das hat gutgetan"
- relations are usually implicit in German narration: "wegen X war ich Y",
  "X hat mich Y gemacht", "nach X ging es mir besser", "durch X wurde Y schlimmer"
  all express causes/increases/decreases without connector words — count them.

German temporal mapping (apply when judging temporal_correct):
- "heute" = today; "jetzt"/"gerade"/"im Moment" = now
- lowercase adverb "morgen" = tomorrow (anticipated); "Morgen" as a NOUN
  ("am Morgen", "heute Morgen", "der Morgen") = this morning, NOT tomorrow
- "gestern" / "von gestern" / "seit gestern" = yesterday; "letzte Nacht"/"gestern
  Abend" = last night
- "heute Morgen" / "heute früh" = this morning (earlier today)
- "heute Nachmittag" / "vorhin" / "nachmittags" (in a same-day entry) = earlier
  today / this afternoon
A node anchored 'yesterday' whose German text says "gestern ..." (in any of the
forms above) IS temporally correct. If the same concept appears at two different
times (e.g. yesterday AND today as two separate nodes), judge each node against
its own time anchor — one node's quote does not invalidate the other.
"""
    system = (
        "You are a strict, skeptical verifier for a research dataset. You judge "
        "whether a journal entry faithfully and specifically expresses a "
        "ground-truth graph. Do not give the benefit of the doubt. A vague or "
        "merely tonal match does NOT count. Respond with JSON only."
    )
    user = f"""Graph nodes:
{json.dumps(g['nodes'], ensure_ascii=False, indent=2)}

Graph relations:
{json.dumps(g['edges'], ensure_ascii=False, indent=2)}
{lang_note}
Journal entry:
\"\"\"{generated_text}\"\"\"

You MUST return exactly one node object for each of these node_ids:
{[n['node_id'] for n in g['nodes']]}
and exactly one edge object for each of these edge_ids:
{[e['edge_id'] for e in g['edges']]}

Judge the entry against the graph and return ONLY this JSON object:
{{
  "nodes": [
    {{"node_id": "n1", "mentioned": true, "evidence": "exact quote from the entry, or \\"\\" if not mentioned", "temporal_correct": true, "temporal_evidence": "quote showing the timing, or \\"\\""}}
  ],
  "edges": [
    {{"edge_id": "e1", "expressed": true, "evidence": "quote/short reason showing the relation is implied, or \\"\\""}}
  ],
  "extra_content": [
    {{"content": "mental-health-relevant symptom/event/stressor/emotion/thought/coping action/time reference in the text but NOT in the graph", "evidence": "exact quote"}}
  ],
  "overall": "accept",
  "reason": "one short sentence justifying the overall decision"
}}

Rules:
- Include EVERY listed node_id and EVERY listed edge_id, exactly once each, no
  more and no fewer. Do not skip edges even if there are several.
- "mentioned" is true ONLY if the node's specific meaning is clearly expressed
  and you can quote concrete supporting text in "evidence". Otherwise false with
  evidence "".
- "temporal_correct" is true ONLY if the text clearly places the node at the
  correct time relative to today (today/now, tomorrow=anticipated only,
  yesterday/last night=before today, this morning=earlier today). If the timing
  is wrong, ambiguous, or absent, set false and explain in "temporal_evidence".
- "extra_content": be strict and thorough. List EVERY mental-health-relevant
  detail the text adds that is not represented by a graph node, including:
  new symptoms, new emotions, new stressors, new events, new coping actions,
  new clinically relevant thoughts, and any extra time reference (e.g.
  "tomorrow", "yesterday", "last night", "next week", "tonight", "hope
  tomorrow is better") that no graph node anchors. Example: if the text says
  "I really hope tomorrow will be better" but no graph node is anchored to a
  future day, list it as extra content. Empty list only if there is genuinely
  nothing added.
- "overall": "accept" only if essentially all nodes and edges are clearly
  expressed, timing is correct, and there is no extra content. "review" for
  partial or borderline matches. "reject" if coverage is poor, timing is wrong,
  or there is clear hallucinated content.
- Be strict. Output valid JSON only, no code fences, no extra text."""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_json_object(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])
