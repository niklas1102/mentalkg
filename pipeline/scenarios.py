import json
import random
from pathlib import Path

ANCHORS = {
    "today": {"text": "today", "day_offset": 0, "temporal_status": "present"},
    "now": {"text": "now", "day_offset": 0, "temporal_status": "present"},
    "tomorrow": {"text": "tomorrow", "day_offset": 1, "temporal_status": "future"},
    "later_today": {"text": "later today", "day_offset": 0, "temporal_status": "future"},
    "yesterday": {"text": "yesterday", "day_offset": -1, "temporal_status": "past"},
    "last_night": {"text": "last night", "day_offset": -1, "temporal_status": "past"},
    "this_morning": {"text": "this morning", "day_offset": 0, "temporal_status": "past"},
    "this_afternoon": {"text": "this afternoon", "day_offset": 0, "temporal_status": "past"},
    "after_that": {"text": "after that", "day_offset": 0, "temporal_status": "present"},
}

GENDERS = ["female", "male", "nonbinary"]

# ~40 occupations spanning ages 16-75 and many life situations.
OCCUPATIONS = [
    "student", "high-school student", "PhD student", "apprentice", "teacher",
    "nurse", "doctor", "paramedic", "designer", "engineer", "software developer",
    "data analyst", "retail worker", "researcher", "freelancer", "office worker",
    "unemployed", "barista", "chef", "social worker", "writer", "journalist",
    "lab technician", "accountant", "mechanic", "plumber", "electrician",
    "translator", "musician", "physiotherapist", "pharmacist", "sales representative",
    "HR manager", "warehouse worker", "delivery driver", "receptionist",
    "consultant", "small-business owner", "farmer", "gardener", "cleaner",
    "IT administrator", "veterinarian", "marketing manager", "architect",
    "real-estate agent", "care worker", "retired", "stay-at-home parent",
]

# Per-participant writing style. Keys must match STYLE_WORDCOUNTS in llm.py.
WRITING_STYLES = ["terse_fragmented", "matter_of_fact", "reflective",
                  "emotional", "rambling"]

ANTICIPATION_EVENTS = [
    "event_exam",
    "event_assignment_due",
    "event_social_meeting",
    "stressor_deadline",
    "stressor_major_decision",
    "stressor_work_pressure",
    "stressor_academic_pressure",
]
ANTICIPATION_EMOTIONS = ["emotion_anxiety", "emotion_stress",
                         "emotion_overwhelm", "emotion_fear"]
ANTICIPATION_THOUGHTS = ["thought_fear_of_failure", "thought_catastrophizing",
                         "thought_uncertain_future", "thought_need_to_be_perfect",
                         "thought_loss_of_control"]

PAST_NEGATIVE_EVENTS = [
    "event_argument",
    "event_rejection",
    "event_criticism",
    "event_missed_deadline",
    "event_failure",
    "event_illness",
    "event_breakup_or_distance",
    "event_poor_sleep_night",
]

CALMING_SYMPTOMS = ["symptom_sleep_loss", "symptom_racing_thoughts", "symptom_tension",
                    "symptom_restlessness", "symptom_stomach_tension", "symptom_chest_tightness"]
ANTICIPATION_SYMPTOMS = ["symptom_racing_thoughts", "symptom_tension",
                         "symptom_restlessness", "symptom_chest_tightness",
                         "symptom_stomach_tension", "symptom_concentration_problem"]
NEUTRAL_ACTIVITIES = ["activity_studying", "activity_working"]
WITHDRAWAL_ACTIVITIES = ["activity_avoiding_people", "activity_procrastination",
                         "activity_scrolling_phone"]
POSITIVE_CALM_EMOTIONS = ["emotion_relief", "emotion_calm", "emotion_happiness",
                          "emotion_contentment", "emotion_hope"]
POSITIVE_THOUGHTS_REFRAME = ["thought_positive_reframe", "thought_perspective_taking",
                             "thought_self_compassion", "thought_acceptance",
                             "thought_gratitude", "thought_self_encouragement"]
SOCIAL_ACTIVITIES = ["activity_talking_friend", "activity_family_call"]
LEISURE_ACTIVITIES = ["activity_creative_hobby", "activity_resting", "activity_cooking",
                      "activity_reading", "activity_walk_outside", "activity_cleaning"]
MOVEMENT_ACTIVITIES = ["activity_exercise", "activity_walk_outside"]

# --- Structural-noise probabilities. Edge-adding is an *independent per-pair*
# --- decision: every currently-unconnected compatible node pair is joined with
# --- probability NOISE_ADD_EDGE_PROB. Together with the one-edge drop and the
# --- extra unconnected node this keeps P(connected | type pair) strictly inside
# --- (0.15, 0.85) for every common pair (verify with graph_stats.py).
NOISE_DROP_EDGE_PROB = 0.30
NOISE_ADD_EDGE_PROB = 0.26
NOISE_EXTRA_NODE_PROB = 0.18

# Ordered (source_type, target_type) -> default edge type for *added* edges.
# Deliberately broad: every reasonable type pair is eligible at least sometimes,
# so no pair stays pinned at 0.0. Polarity refines increases/decreases at add time.
COMPAT = {
    ("event", "emotion"): "causes",
    ("event", "thought"): "causes",
    ("event", "symptom"): "causes",
    ("event", "activity"): "linked_to",
    ("event", "event"): "follows",
    ("event", "stressor"): "linked_to",
    ("event", "coping_action"): "linked_to",
    ("stressor", "emotion"): "causes",
    ("stressor", "thought"): "causes",
    ("stressor", "symptom"): "causes",
    ("stressor", "activity"): "linked_to",
    ("stressor", "stressor"): "linked_to",
    ("stressor", "coping_action"): "linked_to",
    ("thought", "emotion"): "increases",
    ("thought", "symptom"): "increases",
    ("thought", "thought"): "increases",
    ("thought", "activity"): "linked_to",
    ("emotion", "symptom"): "increases",
    ("emotion", "activity"): "linked_to",
    ("emotion", "thought"): "increases",
    ("emotion", "emotion"): "increases",
    ("coping_action", "emotion"): "decreases",
    ("coping_action", "symptom"): "decreases",
    ("coping_action", "thought"): "decreases",
    ("coping_action", "activity"): "linked_to",
    ("coping_action", "coping_action"): "linked_to",
    ("activity", "emotion"): "linked_to",
    ("activity", "symptom"): "linked_to",
    ("activity", "thought"): "linked_to",
    ("activity", "activity"): "linked_to",
    ("activity", "coping_action"): "linked_to",
    ("symptom", "symptom"): "follows",
    ("symptom", "activity"): "linked_to",
    ("symptom", "emotion"): "increases",
}

TEMPLATES = [
    {
        "name": "stressor_reaction",
        "valence": "neg",
        "slots": [
            {"role": "stressor", "types": ["stressor"], "polarities": ["negative"], "anchor": "today"},
            {"role": "emotion_neg", "types": ["emotion"], "polarities": ["negative"], "anchor": "today"},
            {"role": "thought_neg", "types": ["thought"], "polarities": ["negative"], "anchor": "today"},
            {"role": "symptom_neg", "types": ["symptom"], "polarities": ["negative"],
             "exclude": ["symptom_sleep_loss"], "anchor": "this_morning"},
        ],
        "edges": [
            ("stressor", "emotion_neg", "causes"),
            ("stressor", "thought_neg", "causes"),
            ("thought_neg", "emotion_neg", "increases"),
            ("emotion_neg", "symptom_neg", "increases"),
        ],
    },
    {
        "name": "anticipation_anxiety",
        "valence": "neg",
        "slots": [
            {"role": "future_event", "types": ["event", "stressor"], "polarities": ["neutral", "negative"],
             "concept_ids": ANTICIPATION_EVENTS, "anchor": "tomorrow"},
            {"role": "anxiety", "types": ["emotion"], "polarities": ["negative"],
             "concept_ids": ANTICIPATION_EMOTIONS, "anchor": "today"},
            {"role": "thought_neg", "types": ["thought"], "polarities": ["negative"],
             "concept_ids": ANTICIPATION_THOUGHTS, "anchor": "today"},
            {"role": "prep_activity", "types": ["activity"], "polarities": None,
             "concept_ids": NEUTRAL_ACTIVITIES, "anchor": "this_morning"},
            {"role": "symptom_neg", "types": ["symptom"], "polarities": ["negative"],
             "concept_ids": ANTICIPATION_SYMPTOMS, "anchor": "now"},
        ],
        "edges": [
            ("future_event", "anxiety", "causes"),
            ("future_event", "thought_neg", "causes"),
            ("anxiety", "symptom_neg", "increases"),
            ("anxiety", "prep_activity", "linked_to"),
        ],
    },
    {
        "name": "followup_past_event",
        "valence": "neg",
        "needs_prev_day": True,
        "slots": [
            {"role": "past_event", "types": ["event"], "polarities": ["negative"],
             "concept_ids": PAST_NEGATIVE_EVENTS, "anchor": "yesterday"},
            {"role": "symptom_night", "types": ["symptom"], "polarities": ["negative"],
             "concept_ids": ["symptom_sleep_loss"], "anchor": "last_night"},
            {"role": "symptom_day", "types": ["symptom"], "polarities": ["negative"],
             "concept_ids": ["symptom_fatigue", "symptom_low_mood", "symptom_concentration_problem"],
             "anchor": "this_morning"},
            {"role": "emotion_neg", "types": ["emotion"], "polarities": ["negative"], "anchor": "now"},
            {"role": "thought_neg", "types": ["thought"], "polarities": ["negative"], "anchor": "now"},
        ],
        "edges": [
            ("past_event", "emotion_neg", "causes"),
            ("past_event", "thought_neg", "causes"),
            ("thought_neg", "emotion_neg", "increases"),
            ("symptom_night", "symptom_day", "causes"),
        ],
    },
    {
        "name": "coping_relief",
        "valence": "mixed",
        "slots": [
            {"role": "stressor", "types": ["stressor"], "polarities": ["negative"], "anchor": "this_morning"},
            {"role": "emotion_neg", "types": ["emotion"], "polarities": ["negative"], "anchor": "this_morning"},
            {"role": "coping", "types": ["coping_action"], "polarities": ["positive"], "anchor": "this_afternoon"},
            {"role": "emotion_pos", "types": ["emotion"], "polarities": ["positive"], "anchor": "now"},
            {"role": "thought_pos", "types": ["thought"], "polarities": ["positive"], "anchor": "now"},
        ],
        "edges": [
            ("stressor", "emotion_neg", "causes"),
            ("coping", "emotion_neg", "decreases"),
            ("coping", "emotion_pos", "causes"),
            ("thought_pos", "emotion_pos", "increases"),
        ],
    },
    {
        "name": "rumination_loop",
        "valence": "neg",
        "slots": [
            {"role": "past_event", "types": ["event"], "polarities": ["negative"],
             "concept_ids": PAST_NEGATIVE_EVENTS, "anchor": "yesterday"},
            {"role": "thought_neg", "types": ["thought"], "polarities": ["negative"],
             "concept_ids": ["thought_rumination", "thought_self_blame", "thought_catastrophizing",
                             "thought_not_good_enough"], "anchor": "today"},
            {"role": "emotion_neg", "types": ["emotion"], "polarities": ["negative"], "anchor": "now"},
            {"role": "symptom_neg", "types": ["symptom"], "polarities": ["negative"],
             "concept_ids": ["symptom_concentration_problem", "symptom_racing_thoughts",
                             "symptom_restlessness", "symptom_low_mood"], "anchor": "today"},
            {"role": "activity_neg", "types": ["activity"], "polarities": ["negative"],
             "concept_ids": WITHDRAWAL_ACTIVITIES, "anchor": "today"},
        ],
        "edges": [
            ("past_event", "thought_neg", "causes"),
            ("thought_neg", "emotion_neg", "increases"),
            ("thought_neg", "symptom_neg", "increases"),
            ("emotion_neg", "activity_neg", "linked_to"),
        ],
    },
    {
        "name": "stress_then_partial_relief",
        "valence": "mixed",
        "slots": [
            {"role": "stressor", "types": ["stressor"], "polarities": ["negative"], "anchor": "this_morning"},
            {"role": "emotion_neg", "types": ["emotion"], "polarities": ["negative"], "anchor": "this_morning"},
            {"role": "symptom_neg", "types": ["symptom"], "polarities": ["negative"],
             "exclude": ["symptom_sleep_loss"], "anchor": "this_morning"},
            {"role": "coping", "types": ["coping_action"], "polarities": ["positive"], "anchor": "this_afternoon"},
            {"role": "emotion_pos", "types": ["emotion"], "polarities": ["positive"],
             "concept_ids": ["emotion_calm", "emotion_relief", "emotion_hope"], "anchor": "now"},
        ],
        "edges": [
            ("stressor", "emotion_neg", "causes"),
            ("emotion_neg", "symptom_neg", "increases"),
            ("coping", "symptom_neg", "decreases"),
            ("coping", "emotion_pos", "causes"),
        ],
    },
    {
        "name": "good_day",
        "valence": "pos",
        "slots": [
            {"role": "good_event", "types": ["event"], "polarities": ["positive"], "anchor": "today"},
            {"role": "emotion_pos", "types": ["emotion"], "polarities": ["positive"], "anchor": "today"},
            {"role": "thought_pos", "types": ["thought"], "polarities": ["positive"], "anchor": "now"},
            {"role": "activity_pos", "types": ["activity"], "polarities": ["positive"], "anchor": "this_afternoon"},
        ],
        "edges": [
            ("good_event", "emotion_pos", "causes"),
            ("thought_pos", "emotion_pos", "increases"),
            ("emotion_pos", "activity_pos", "linked_to"),
        ],
    },
    {
        "name": "mixed_day",
        "valence": "mixed",
        "slots": [
            {"role": "stressor", "types": ["stressor"], "polarities": ["negative"], "anchor": "this_morning"},
            {"role": "emotion_neg", "types": ["emotion"], "polarities": ["negative"], "anchor": "this_morning"},
            {"role": "thought_neg", "types": ["thought"], "polarities": ["negative"], "anchor": "this_morning"},
            {"role": "symptom_neg", "types": ["symptom"], "polarities": ["negative"],
             "exclude": ["symptom_sleep_loss"], "anchor": "this_morning"},
            {"role": "good_event", "types": ["event"], "polarities": ["positive"], "anchor": "this_afternoon"},
            {"role": "coping", "types": ["coping_action"], "polarities": ["positive"], "anchor": "this_afternoon"},
            {"role": "emotion_pos", "types": ["emotion"], "polarities": ["positive"], "anchor": "now"},
        ],
        "edges": [
            ("stressor", "emotion_neg", "causes"),
            ("stressor", "thought_neg", "causes"),
            ("emotion_neg", "symptom_neg", "increases"),
            ("good_event", "emotion_pos", "causes"),
            ("coping", "symptom_neg", "decreases"),
        ],
    },
    {
        "name": "social_support_day",
        "valence": "pos",
        "slots": [
            {"role": "social", "types": ["activity"], "polarities": ["positive"],
             "concept_ids": SOCIAL_ACTIVITIES, "anchor": "this_afternoon"},
            {"role": "lonely", "types": ["emotion"], "polarities": ["negative"],
             "concept_ids": ["emotion_loneliness", "emotion_sadness"], "anchor": "this_morning"},
            {"role": "emotion_pos", "types": ["emotion"], "polarities": ["positive"], "anchor": "now"},
            {"role": "thought_pos", "types": ["thought"], "polarities": ["positive"], "anchor": "now"},
        ],
        "edges": [
            ("social", "emotion_pos", "causes"),
            ("social", "lonely", "decreases"),
            ("thought_pos", "emotion_pos", "increases"),
        ],
    },
    {
        "name": "physical_health_day",
        "valence": "pos",
        "slots": [
            {"role": "activity_move", "types": ["activity"], "polarities": ["positive"],
             "concept_ids": MOVEMENT_ACTIVITIES, "anchor": "this_afternoon"},
            {"role": "symptom_neg", "types": ["symptom"], "polarities": ["negative"],
             "concept_ids": ["symptom_tension", "symptom_fatigue", "symptom_restlessness", "symptom_low_mood"],
             "anchor": "this_morning"},
            {"role": "emotion_pos", "types": ["emotion"], "polarities": ["positive"], "anchor": "now"},
            {"role": "thought_pos", "types": ["thought"], "polarities": ["positive"], "anchor": "now"},
        ],
        "edges": [
            ("activity_move", "symptom_neg", "decreases"),
            ("activity_move", "emotion_pos", "causes"),
            ("thought_pos", "emotion_pos", "increases"),
        ],
    },
    {
        "name": "work_routine_mild_stress",
        "valence": "neutral",
        "slots": [
            {"role": "routine_event", "types": ["event"], "polarities": ["neutral"], "anchor": "today"},
            {"role": "stressor", "types": ["stressor"], "polarities": ["negative"], "anchor": "today"},
            {"role": "emotion_neg", "types": ["emotion"], "polarities": ["negative"],
             "concept_ids": ["emotion_stress", "emotion_irritability", "emotion_overwhelm"], "anchor": "today"},
            {"role": "activity_work", "types": ["activity"], "polarities": ["neutral"],
             "concept_ids": NEUTRAL_ACTIVITIES, "anchor": "this_morning"},
        ],
        "edges": [
            ("stressor", "emotion_neg", "causes"),
            ("routine_event", "emotion_neg", "causes"),
            ("emotion_neg", "activity_work", "linked_to"),
        ],
    },
    {
        "name": "recovery_after_bad_day",
        "valence": "mixed",
        "needs_prev_day": True,
        "slots": [
            {"role": "past_event", "types": ["event"], "polarities": ["negative"],
             "concept_ids": PAST_NEGATIVE_EVENTS, "anchor": "yesterday"},
            {"role": "symptom_day", "types": ["symptom"], "polarities": ["negative"],
             "concept_ids": ["symptom_fatigue", "symptom_low_mood", "symptom_concentration_problem"],
             "anchor": "this_morning"},
            {"role": "coping", "types": ["coping_action"], "polarities": ["positive"], "anchor": "this_afternoon"},
            {"role": "emotion_pos", "types": ["emotion"], "polarities": ["positive"], "anchor": "now"},
            {"role": "thought_pos", "types": ["thought"], "polarities": ["positive"], "anchor": "now"},
        ],
        "edges": [
            ("past_event", "symptom_day", "causes"),
            ("coping", "symptom_day", "decreases"),
            ("coping", "emotion_pos", "causes"),
            ("thought_pos", "emotion_pos", "increases"),
        ],
    },
    {
        "name": "anticipation_turns_out_fine",
        "valence": "pos",
        "slots": [
            {"role": "anxiety", "types": ["emotion"], "polarities": ["negative"],
             "concept_ids": ["emotion_anxiety", "emotion_stress", "emotion_fear"], "anchor": "this_morning"},
            {"role": "symptom_neg", "types": ["symptom"], "polarities": ["negative"],
             "concept_ids": ANTICIPATION_SYMPTOMS, "anchor": "this_morning"},
            {"role": "the_event", "types": ["event"], "polarities": ["neutral", "positive"],
             "concept_ids": ["event_exam", "event_assignment_due", "event_social_meeting",
                             "event_travel", "event_promotion_or_win"], "anchor": "this_afternoon"},
            {"role": "relief", "types": ["emotion"], "polarities": ["positive"],
             "concept_ids": ["emotion_relief", "emotion_calm", "emotion_happiness"], "anchor": "now"},
            {"role": "thought_pos", "types": ["thought"], "polarities": ["positive"],
             "concept_ids": POSITIVE_THOUGHTS_REFRAME, "anchor": "now"},
        ],
        "edges": [
            ("anxiety", "symptom_neg", "increases"),
            ("the_event", "relief", "causes"),
            ("the_event", "anxiety", "decreases"),
            ("thought_pos", "relief", "increases"),
        ],
    },
    {
        "name": "conflict_then_repair",
        "valence": "mixed",
        "slots": [
            {"role": "conflict", "types": ["event"], "polarities": ["negative"],
             "concept_ids": ["event_argument", "event_criticism", "event_rejection"], "anchor": "this_morning"},
            {"role": "emotion_neg", "types": ["emotion"], "polarities": ["negative"],
             "concept_ids": ["emotion_anger", "emotion_sadness", "emotion_guilt", "emotion_irritability"],
             "anchor": "this_morning"},
            {"role": "repair", "types": ["event"], "polarities": ["positive"],
             "concept_ids": ["event_reconciliation"], "anchor": "this_afternoon"},
            {"role": "emotion_pos", "types": ["emotion"], "polarities": ["positive"],
             "concept_ids": ["emotion_relief", "emotion_calm", "emotion_happiness"], "anchor": "now"},
            {"role": "thought_pos", "types": ["thought"], "polarities": ["positive"], "anchor": "now"},
        ],
        "edges": [
            ("conflict", "emotion_neg", "causes"),
            ("repair", "emotion_pos", "causes"),
            ("repair", "emotion_neg", "decreases"),
            ("thought_pos", "emotion_pos", "increases"),
        ],
    },
    {
        "name": "low_energy_numb_day",
        "valence": "neg",
        "slots": [
            {"role": "numb", "types": ["emotion"], "polarities": ["negative"],
             "concept_ids": ["emotion_numbness", "emotion_sadness"], "anchor": "today"},
            {"role": "fatigue", "types": ["symptom"], "polarities": ["negative"],
             "concept_ids": ["symptom_fatigue", "symptom_low_mood", "symptom_loss_of_interest"],
             "anchor": "today"},
            {"role": "withdrawal", "types": ["activity"], "polarities": ["negative"],
             "concept_ids": WITHDRAWAL_ACTIVITIES, "anchor": "today"},
            {"role": "thought_neg", "types": ["thought"], "polarities": ["negative"],
             "concept_ids": ["thought_rumination", "thought_not_good_enough", "thought_self_blame"],
             "anchor": "today"},
        ],
        "edges": [
            ("thought_neg", "numb", "increases"),
            ("numb", "withdrawal", "linked_to"),
            ("fatigue", "withdrawal", "linked_to"),
        ],
    },
    {
        "name": "weekend_leisure_day",
        "valence": "pos",
        "slots": [
            {"role": "leisure", "types": ["activity"], "polarities": ["positive"],
             "concept_ids": LEISURE_ACTIVITIES, "anchor": "today"},
            {"role": "calm", "types": ["emotion"], "polarities": ["positive"],
             "concept_ids": ["emotion_calm", "emotion_contentment", "emotion_happiness", "emotion_relief"],
             "anchor": "now"},
            {"role": "thought_pos", "types": ["thought"], "polarities": ["positive"], "anchor": "now"},
        ],
        "edges": [
            ("leisure", "calm", "causes"),
            ("thought_pos", "calm", "increases"),
        ],
    },
]

TEMPLATE_BY_NAME = {t["name"]: t for t in TEMPLATES}


def load_pool_index(pool_path):
    with open(pool_path, "r", encoding="utf-8") as f:
        pool = json.load(f)
    index = {}
    by_id = {}
    for concept in pool["node_concepts"]:
        index.setdefault(concept["type"], []).append(concept)
        by_id[concept["concept_id"]] = concept
    return index, by_id


def pick_concept(rng, index, by_id, slot, used):
    exclude = set(slot.get("exclude", []))
    allowed_ids = slot.get("concept_ids")
    if allowed_ids:
        ids = [c for c in allowed_ids if c not in exclude]
        candidates = [by_id[c] for c in ids if c in by_id and c not in used]
        if not candidates:
            candidates = [by_id[c] for c in ids if c in by_id]
        return rng.choice(candidates)

    polarities = slot["polarities"]
    candidates = []
    for t in slot["types"]:
        for c in index.get(t, []):
            if polarities is not None and c["polarity"] not in polarities:
                continue
            if c["concept_id"] in exclude or c["concept_id"] in used:
                continue
            candidates.append(c)
    if not candidates:
        for t in slot["types"]:
            for c in index.get(t, []):
                if polarities is not None and c["polarity"] not in polarities:
                    continue
                if c["concept_id"] in exclude:
                    continue
                candidates.append(c)
    return rng.choice(candidates)


def _next_id(items, key, prefix):
    mx = 0
    for it in items:
        v = it.get(key, "")
        if isinstance(v, str) and v.startswith(prefix):
            try:
                mx = max(mx, int(v[len(prefix):]))
            except ValueError:
                pass
    return f"{prefix}{mx + 1}"


def make_participant_profile(rng, i, by_id, language):
    stressor_ids = [c for c in by_id if by_id[c]["type"] == "stressor"]
    coping_ids = [c for c in by_id if by_id[c]["type"] == "coping_action"
                  and by_id[c]["polarity"] == "positive"]
    recurring = rng.sample(stressor_ids, k=rng.choice([1, 2, 3]))
    coping = rng.sample(coping_ids, k=rng.choice([1, 2, 3]))
    return {
        "participant_id": f"p{i:05d}",
        "language": language,
        "age": rng.randint(16, 75),
        "gender": rng.choice(GENDERS),
        "occupation": rng.choice(OCCUPATIONS),
        "writing_style": rng.choice(WRITING_STYLES),
        "recurring_stressors": [by_id[c]["label"] for c in recurring],
        "coping_style": [by_id[c]["label"] for c in coping],
        "_recurring_ids": recurring,
        "_coping_ids": coping,
    }


def choose_template(rng, day, mood, prev):
    names = [t["name"] for t in TEMPLATES
             if not (day == 1 and t.get("needs_prev_day"))]
    weights = {n: 1.0 for n in names}
    for n in names:
        val = TEMPLATE_BY_NAME[n]["valence"]
        if mood < 4.5 and val == "neg":
            weights[n] += 2.5
        elif 4.5 <= mood <= 6.5 and val in ("mixed", "neutral"):
            weights[n] += 1.5
        elif mood > 6.5 and val == "pos":
            weights[n] += 2.5
        # keep some baseline mixing so no valence class ever drops out
        if val in ("mixed", "neutral"):
            weights[n] += 0.3

    if prev.get("had_future_event") and "followup_past_event" in weights:
        weights["followup_past_event"] += 2.5
    if prev.get("had_coping"):
        for n in ("stress_then_partial_relief", "coping_relief", "recovery_after_bad_day"):
            if n in weights:
                weights[n] += 1.2
    if prev.get("bad_prev_day"):
        for n in ("recovery_after_bad_day", "followup_past_event"):
            if n in weights:
                weights[n] += 1.8

    ns = list(weights)
    return rng.choices(ns, weights=[weights[n] for n in ns], k=1)[0]


def build_day_graph(rng, index, by_id, template, entry_id, profile):
    nodes = []
    role_to_node = {}
    used = set()
    for slot_i, slot in enumerate(template["slots"], start=1):
        concept = pick_concept(rng, index, by_id, slot, used)
        if slot.get("role") == "stressor" and profile["_recurring_ids"] and rng.random() < 0.5:
            concept = by_id[rng.choice(profile["_recurring_ids"])]
        if concept["type"] == "coping_action" and profile["_coping_ids"] and rng.random() < 0.6:
            concept = by_id[rng.choice(profile["_coping_ids"])]
        used.add(concept["concept_id"])
        anchor_key = slot["anchor"]
        if concept["concept_id"] == "symptom_sleep_loss":
            anchor_key = "last_night"
        anchor = ANCHORS[anchor_key]
        node_id = f"n{slot_i}"
        nodes.append({
            "node_id": node_id,
            "concept_id": concept["concept_id"],
            "type": concept["type"],
            "label": concept["label"],
            "time_anchor": {"text": anchor["text"], "day_offset": anchor["day_offset"]},
            "temporal_status": anchor["temporal_status"],
            "polarity": concept["polarity"],
            "confidence": round(rng.uniform(0.75, 1.0), 2),
            "source_entry_id": entry_id,
        })
        role_to_node[slot["role"]] = node_id

    edges = []
    for edge_i, (src_role, tgt_role, etype) in enumerate(template["edges"], start=1):
        edges.append({
            "edge_id": f"e{edge_i}",
            "source_node_id": role_to_node[src_role],
            "target_node_id": role_to_node[tgt_role],
            "type": etype,
            "weight": round(rng.uniform(0.6, 1.0), 2),
            "confidence": round(rng.uniform(0.7, 1.0), 2),
            "source": entry_id,
        })
    return nodes, edges, used


def _degree(edges):
    deg = {}
    for e in edges:
        deg[e["source_node_id"]] = deg.get(e["source_node_id"], 0) + 1
        deg[e["target_node_id"]] = deg.get(e["target_node_id"], 0) + 1
    return deg


def _added_edge_spec(na, nb):
    """Return (src_node, tgt_node, etype) for a plausible edge between two nodes,
    or None if the type pair is not in the compatibility table."""
    for src, tgt in ((na, nb), (nb, na)):
        etype = COMPAT.get((src["type"], tgt["type"]))
        if etype is None:
            continue
        sp, tp = src["polarity"], tgt["polarity"]
        if src["type"] == "coping_action":
            etype = "decreases" if tp == "negative" else "causes"
        elif etype in ("increases", "decreases"):
            if sp == "positive" and tp == "negative":
                etype = "decreases"
            elif sp == "negative" and tp == "positive":
                etype = "decreases"
            elif sp in ("positive", "negative") and tp == sp:
                etype = "increases"
        return src, tgt, etype
    return None


def apply_structure_noise(rng, index, by_id, nodes, edges, entry_id):
    """Randomly perturb a (template + continuity) graph so edge connectivity is not
    deterministic given node types: drop a non-essential edge, optionally add an
    extra node, then independently join unconnected compatible pairs. Runs after
    continuity so carried nodes also participate in the shortcut-breaking pass."""
    used = {n["concept_id"] for n in nodes}  # includes any continuity carry node
    # (a) drop one non-essential edge (never orphan an endpoint)
    if edges and rng.random() < NOISE_DROP_EDGE_PROB:
        deg = _degree(edges)
        droppable = [i for i, e in enumerate(edges)
                     if deg[e["source_node_id"]] > 1 and deg[e["target_node_id"]] > 1]
        if droppable:
            edges.pop(rng.choice(droppable))

    # (b) add one extra node ("mentioned", often not linked). Added before the
    #     linking pass so it *can* be joined -- but usually is not -- which also
    #     lets same-type pairs (thought/thought, activity/activity, ...) occur.
    if rng.random() < NOISE_EXTRA_NODE_PROB:
        day_neg = _day_polarity(nodes) < 0
        pols = ["negative"] if day_neg else ["positive", "neutral"]
        types = ["emotion", "symptom", "thought", "activity", "event", "stressor", "coping_action"]
        rng.shuffle(types)
        concept = None
        for t in types:
            cands = [c for c in index.get(t, [])
                     if c["concept_id"] not in used and c["polarity"] in pols]
            if cands:
                concept = rng.choice(cands)
                break
        if concept is not None:
            used.add(concept["concept_id"])
            anchor = ANCHORS["today"]
            nodes.append({
                "node_id": _next_id(nodes, "node_id", "n"),
                "concept_id": concept["concept_id"],
                "type": concept["type"],
                "label": concept["label"],
                "time_anchor": {"text": anchor["text"], "day_offset": anchor["day_offset"]},
                "temporal_status": anchor["temporal_status"],
                "polarity": concept["polarity"],
                "confidence": round(rng.uniform(0.75, 1.0), 2),
                "source_entry_id": entry_id,
            })

    # (c) independently connect each currently-unconnected compatible node pair
    #     with probability NOISE_ADD_EDGE_PROB (breaks the type->edge shortcut for
    #     rare cross-branch and same-type pairs without touching pairs templates
    #     already join)
    if len(nodes) >= 2:
        connected = {(e["source_node_id"], e["target_node_id"]) for e in edges}
        connected |= {(b, a) for a, b in connected}
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                if (a["node_id"], b["node_id"]) in connected:
                    continue
                if rng.random() >= NOISE_ADD_EDGE_PROB:
                    continue
                spec = _added_edge_spec(a, b)
                if spec is None:
                    continue
                src, tgt, etype = spec
                edges.append({
                    "edge_id": _next_id(edges, "edge_id", "e"),
                    "source_node_id": src["node_id"],
                    "target_node_id": tgt["node_id"],
                    "type": etype,
                    "weight": round(rng.uniform(0.6, 1.0), 2),
                    "confidence": round(rng.uniform(0.7, 1.0), 2),
                    "source": entry_id,
                })
                connected.add((a["node_id"], b["node_id"]))
                connected.add((b["node_id"], a["node_id"]))


def inject_continuity(rng, nodes, edges, prev, entry_id):
    carry = prev.get("carry")
    if not carry:
        return None
    targets = [n for n in nodes if n["type"] in ("symptom", "emotion", "thought")
               and n["time_anchor"]["day_offset"] >= 0]
    if not targets:
        return None
    target = rng.choice(targets)
    y_id = _next_id(nodes, "node_id", "n")
    nodes.append({
        "node_id": y_id,
        "concept_id": carry["concept_id"],
        "type": carry["type"],
        "label": carry["label"],
        "time_anchor": {"text": "yesterday", "day_offset": -1},
        "temporal_status": "past",
        "polarity": carry["polarity"],
        "confidence": round(rng.uniform(0.75, 1.0), 2),
        "source_entry_id": entry_id,
    })
    etype = "causes" if carry["type"] in ("stressor", "event", "emotion") else "follows"
    edges.append({
        "edge_id": _next_id(edges, "edge_id", "e"),
        "source_node_id": y_id,
        "target_node_id": target["node_id"],
        "type": etype,
        "weight": round(rng.uniform(0.6, 1.0), 2),
        "confidence": round(rng.uniform(0.7, 1.0), 2),
        "source": entry_id,
    })
    return carry["label"]


def _day_polarity(nodes):
    pos = sum(1 for n in nodes if n["type"] in ("emotion", "symptom") and n["polarity"] == "positive")
    neg = sum(1 for n in nodes if n["type"] in ("emotion", "symptom") and n["polarity"] == "negative")
    return pos - neg


def generate_dataset(pool_path, n_participants, n_days, seed, language_counts=None):
    """language_counts: optional {"en": E, "de": D} with E + D == n_participants.
    Languages alternate en,de,en,de,... until one quota is exhausted, then the
    remainder gets the other language. With equal counts this reproduces the
    default parity assignment exactly. A participant's graphs depend only on its
    index p (language never feeds the rng), so graphs are identical either way."""
    index, by_id = load_pool_index(pool_path)
    samples = []
    remaining = dict(language_counts) if language_counts else None
    for p in range(1, n_participants + 1):
        rng = random.Random(seed * 100000 + p)
        if remaining is None:
            language = "de" if p % 2 == 0 else "en"
        else:
            pref = "de" if p % 2 == 0 else "en"
            other = "en" if pref == "de" else "de"
            language = pref if remaining.get(pref, 0) > 0 else other
            remaining[language] -= 1
        profile = make_participant_profile(rng, p, by_id, language)
        public = {k: v for k, v in profile.items() if not k.startswith("_")}
        baseline = round(rng.uniform(4.5, 6.5), 1)
        mood = baseline
        prev = {}
        for day in range(1, n_days + 1):
            sample_id = f"{profile['participant_id']}_d{day:02d}"
            entry_id = f"entry_{sample_id}"
            tname = choose_template(rng, day, mood, prev)
            template = TEMPLATE_BY_NAME[tname]
            nodes, edges, _used = build_day_graph(rng, index, by_id, template, entry_id, profile)
            carried = inject_continuity(rng, nodes, edges, prev, entry_id) if day > 1 else None
            apply_structure_noise(rng, index, by_id, nodes, edges, entry_id)

            delta = (0.15 * _day_polarity(nodes)
                     + 0.25 * (baseline - mood)
                     + rng.gauss(0, 0.25)
                     + (0.4 if tname in ("coping_relief", "stress_then_partial_relief",
                                         "good_day", "recovery_after_bad_day",
                                         "conflict_then_repair", "anticipation_turns_out_fine",
                                         "social_support_day", "physical_health_day",
                                         "weekend_leisure_day") else 0.0))
            mood = round(max(1.0, min(9.0, mood + delta)), 1)

            had_future = any(n["temporal_status"] == "future" for n in nodes)
            had_coping = any(n["type"] == "coping_action" for n in nodes)
            unresolved = [n for n in nodes if n["polarity"] == "negative"
                          and n["type"] in ("stressor", "symptom", "emotion", "event")
                          and n["time_anchor"]["day_offset"] >= 0]
            carry_node = rng.choice(unresolved) if unresolved else None
            prev = {
                "had_future_event": had_future,
                "had_coping": had_coping,
                "bad_prev_day": mood < 4.5,
                "carry": ({"concept_id": carry_node["concept_id"], "type": carry_node["type"],
                           "label": carry_node["label"], "polarity": carry_node["polarity"]}
                          if carry_node and rng.random() < 0.7 else None),
            }

            samples.append({
                "schema_version": "0.1",
                "sample_id": sample_id,
                "source": "synthetic",
                "scenario": tname,
                "participant": dict(public),
                "entry": {"entry_id": entry_id, "day_index": day, "text": ""},
                "day_index": day,
                "mood_score": mood,
                "previous_day_context": (f"Yesterday involved {carried}." if carried else ""),
                "graph": {"graph_id": f"graph_{sample_id}", "nodes": nodes, "edges": edges},
            })
    return samples
