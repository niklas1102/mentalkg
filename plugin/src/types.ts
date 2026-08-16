export type ServerNode = {
  node_id: string;
  concept_id: string;
  type: string;
  label: string;
  polarity?: string;
  time_anchor?: { text?: string; day_offset?: number };
  temporal_status?: string;
  confidence?: number;
  source_entry_id?: string;
};

export type ServerEdge = {
  edge_id?: string;
  source_node_id: string;
  target_node_id: string;
  type: string;
  weight?: number;
  confidence?: number;
  source?: string;
};

export type PredictedConcept = { concept_id: string; score: number };

export type ExtractResponse = {
  nodes: ServerNode[];
  edges: ServerEdge[];
  predicted_concepts: PredictedConcept[];
  entry_text?: string;
  node_threshold?: number;
  edge_threshold?: number;
  note?: string;
};

/** Everything is optional — notes need no frontmatter at all. */
export type Frontmatter = {
  date?: string;
  mood_score?: number;
};

export type JournalDay = {
  file: import("obsidian").TFile;
  path: string;
  date: string;
  dateSource: import("./date-parse").DateSource;
  mood_score: number | null;
  body: string;
};

export const VIEW_TYPE = "journal-graph-view";
export const summaryPath = (date: string) => `therapist-summary-${date}.md`;

export const TYPE_COLORS: Record<string, string> = {
  emotion: "#D4737E",
  symptom: "#9B7EC8",
  event: "#5B8DC9",
  stressor: "#DA9046",
  activity: "#6BA582",
  thought: "#55A3A3",
  coping_action: "#C9A63C",
};
export const FALLBACK_COLOR = "#9CA3AF";

/** Text glyphs for polarity, matched by the badge drawn on graph nodes. */
export const POLARITY_GLYPH: Record<string, string> = {
  positive: "+",
  negative: "×",
};

export const HONESTY =
  "Node/edge connectivity are model-predicted; edge relation TYPE is a heuristic, not a trained model.";
