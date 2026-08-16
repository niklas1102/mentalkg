import { Network } from "vis-network/standalone/esm/vis-network";
import { DataSet } from "vis-data/standalone";
import {
  ExtractResponse,
  FALLBACK_COLOR,
  POLARITY_GLYPH,
  ServerNode,
  TYPE_COLORS,
} from "./types";
import { ThemeColors, readThemeColors, withAlpha } from "./theme";

export type ClickHandler = (node: ServerNode | null) => void;

export type TimelineStats = {
  added: number;
  recurring: number;
  removed: number;
  firstDay: boolean;
};

type NodeMeta = { node: ServerNode; r: number; streak: number; isNew: boolean };

type VisItem = { id: string } & Record<string, unknown>;

const R_MIN = 11;
const R_MAX = 16;
const BADGE_R = 5.5;
const STREAK_R = 6.5;

const nodeRadius = (confidence?: number) => {
  const c = typeof confidence === "number" ? Math.max(0, Math.min(1, confidence)) : 0.5;
  return R_MIN + (R_MAX - R_MIN) * c;
};

type CtxArgs = {
  ctx: CanvasRenderingContext2D;
  x: number;
  y: number;
  state: { selected: boolean; hover: boolean };
};

/** Normalized per-day graph keyed by concept, for the timeline diff. */
type DayGraph = {
  nodes: Map<string, ServerNode>;
  edges: Map<string, { from: string; to: string; type: string; weight: number }>;
};

function normalizeDay(payload: ExtractResponse): DayGraph {
  const nodes = new Map<string, ServerNode>();
  for (const n of payload.nodes) {
    const existing = nodes.get(n.concept_id);
    if (!existing || (n.confidence ?? 0) > (existing.confidence ?? 0)) {
      nodes.set(n.concept_id, n);
    }
  }
  const idToConcept = new Map<string, string>();
  for (const n of payload.nodes) idToConcept.set(n.node_id, n.concept_id);
  const edges = new Map<string, { from: string; to: string; type: string; weight: number }>();
  for (const e of payload.edges) {
    const from = idToConcept.get(e.source_node_id);
    const to = idToConcept.get(e.target_node_id);
    if (!from || !to || from === to) continue;
    const key = `${from}→${to}:${e.type}`;
    const w = e.weight ?? 0.5;
    const existing = edges.get(key);
    if (!existing || w > existing.weight) edges.set(key, { from, to, type: e.type, weight: w });
  }
  return { nodes, edges };
}

export class GraphRenderer {
  private network?: Network;
  private nodeIndex = new Map<string, ServerNode>();
  private theme: ThemeColors = readThemeColors();
  private useCustomShapes: boolean;
  private lastPayload?: ExtractResponse;
  private onClick?: ClickHandler;

  // Timeline continuity state. Metas are read by ctxRenderers at draw time —
  // never baked into closures, because vis ignores ctxRenderer swaps on update().
  private tlActive = false;
  private tlNodes?: DataSet<VisItem>;
  private tlEdges?: DataSet<VisItem>;
  private tlMeta = new Map<string, NodeMeta>();
  private selectedId: string | null = null;
  private settleGen = 0;

  constructor(private host: HTMLElement) {
    this.useCustomShapes = this.selfTestRenderer();
  }

  /** Exercise the badge/ring/streak renderer against a scratch canvas before trusting it. */
  private selfTestRenderer(): boolean {
    try {
      const ctx = document.createElement("canvas").getContext("2d");
      if (!ctx) return false;
      const probe: ServerNode = {
        node_id: "probe",
        concept_id: "probe",
        type: "emotion",
        label: "probe",
        polarity: "positive",
        confidence: 0.5,
      };
      const meta: NodeMeta = { node: probe, r: nodeRadius(0.5), streak: 3, isNew: true };
      const out = this.makeCtxRenderer(() => meta)({
        ctx,
        x: 0,
        y: 0,
        state: { selected: false, hover: false },
      });
      out.drawNode();
      out.drawExternalLabel();
      return out.nodeDimensions.width > 0;
    } catch {
      return false;
    }
  }

  setTheme(theme: ThemeColors) {
    this.theme = theme;
    if (!this.network) return;
    if (this.useCustomShapes) {
      // Node colors live in ctxRenderer closures that read this.theme — a redraw suffices.
      this.network.setOptions({ edges: this.edgeOptions(), nodes: this.nodeOptions() });
      if (this.tlActive && this.tlEdges) {
        this.tlEdges.update(this.tlEdges.get().map((e) => this.recolorEdge(e)));
      }
      this.network.redraw();
    } else if (this.lastPayload && this.onClick) {
      // Fallback nodes carry baked-in colors; rebuild data in place.
      this.render(this.lastPayload, this.onClick);
    }
  }

  private recolorEdge(e: VisItem): VisItem {
    const isNew = e.jgNew === true;
    return { ...e, color: this.edgeColor(isNew) };
  }

  private edgeColor(isNew: boolean) {
    const t = this.theme;
    return isNew
      ? { color: t.accent, highlight: t.accent, hover: t.accent }
      : { color: withAlpha(t.textMuted, 0.45), highlight: t.accent, hover: t.accent };
  }

  // ─── Current-note mode: full render per refresh (network reused) ─────────
  render(payload: ExtractResponse, onClick: ClickHandler) {
    this.resetTimelineState();
    this.lastPayload = payload;
    this.onClick = onClick;
    this.nodeIndex.clear();
    for (const n of payload.nodes) this.nodeIndex.set(n.node_id, n);
    try {
      this.applyData(payload);
    } catch (err) {
      if (!this.useCustomShapes) throw err;
      this.useCustomShapes = false;
      this.network?.destroy();
      this.network = undefined;
      this.host.empty();
      this.applyData(payload);
    }
  }

  private applyData(payload: ExtractResponse) {
    const nodes = new DataSet(this.toVisNodes(payload));
    const edges = new DataSet(this.toVisEdges(payload));
    const data = { nodes: nodes as unknown as never, edges: edges as unknown as never };

    if (!this.network) {
      this.createNetwork(data);
    } else {
      // Reuse the network so the previous graph stays on the canvas until the
      // new data draws — no blank flash between refreshes.
      this.network.setOptions({ physics: { enabled: true }, edges: this.edgeOptions() });
      this.network.setData(data);
    }

    this.network!.once("stabilizationIterationsDone", () => {
      this.network?.setOptions({ physics: { enabled: false } });
      this.network?.fit({ animation: false });
    });
    // Surface a broken custom renderer synchronously so render() can fall back.
    this.network!.redraw();
  }

  private createNetwork(data: { nodes: never; edges: never }) {
    this.network = new Network(this.host, data, {
      physics: {
        // fit:false — all fits are manual; stabilize() must never move the camera.
        stabilization: { iterations: 250, fit: false },
        barnesHut: {
          gravitationalConstant: -6000,
          springLength: 160,
          springConstant: 0.04,
          damping: 0.35,
          avoidOverlap: 0.6,
        },
      },
      nodes: this.nodeOptions(),
      edges: this.edgeOptions(),
      interaction: { hover: true, tooltipDelay: 200 },
    });
    this.network.on("click", (params: { nodes: string[] }) => {
      const id = params.nodes?.[0];
      this.selectedId = id ?? null;
      this.onClick?.(id ? this.nodeIndex.get(id) ?? null : null);
    });
    this.network.once("afterDrawing", () => {
      this.network?.fit({ animation: false });
    });
  }

  // ─── Timeline mode: incremental diff, stable positions ───────────────────
  renderTimeline(
    current: ExtractResponse,
    prev: ExtractResponse | null,
    streaks: Map<string, number>,
    onClick: ClickHandler,
  ): TimelineStats {
    this.onClick = onClick;
    this.lastPayload = current;

    const cur = normalizeDay(current);
    const prevDay = prev ? normalizeDay(prev) : null;
    const firstDay = prevDay === null;

    let added = 0;
    let recurring = 0;
    let removed = 0;
    if (prevDay) {
      for (const cid of cur.nodes.keys()) {
        if (prevDay.nodes.has(cid)) recurring += 1;
        else added += 1;
      }
      for (const cid of prevDay.nodes.keys()) {
        if (!cur.nodes.has(cid)) removed += 1;
      }
    }
    const stats: TimelineStats = { added, recurring, removed, firstDay };

    // Refresh draw-time metas for the current day.
    for (const [cid, node] of cur.nodes) {
      this.tlMeta.set(cid, {
        node,
        r: nodeRadius(node.confidence),
        streak: streaks.get(cid) ?? 1,
        isNew: !firstDay && !prevDay!.nodes.has(cid),
      });
    }
    for (const cid of [...this.tlMeta.keys()]) {
      if (!cur.nodes.has(cid)) this.tlMeta.delete(cid);
    }
    this.nodeIndex.clear();
    for (const [cid, node] of cur.nodes) this.nodeIndex.set(cid, node);

    if (!this.useCustomShapes) {
      // Emergency fallback: no continuity, plain per-day render keyed by concept.
      this.fullTimelineBuild(cur, prevDay);
      return stats;
    }

    try {
      if (!this.network || !this.tlActive || !this.tlNodes || !this.tlEdges) {
        this.fullTimelineBuild(cur, prevDay);
      } else {
        this.incrementalTimelineStep(cur, prevDay);
      }
    } catch (err) {
      if (!this.useCustomShapes) throw err;
      // Any incremental hiccup: rebuild from scratch rather than show a broken canvas.
      this.tlActive = false;
      this.fullTimelineBuild(cur, prevDay);
    }
    return stats;
  }

  private timelineNodeObject(cid: string, seed?: { x: number; y: number }): VisItem {
    const meta = this.tlMeta.get(cid)!;
    if (!this.useCustomShapes) {
      const n = meta.node;
      const glyph = POLARITY_GLYPH[n.polarity ?? ""];
      const t = this.theme;
      const background = TYPE_COLORS[n.type] ?? FALLBACK_COLOR;
      return {
        id: cid,
        label: (n.label || n.concept_id) + (glyph ? ` ${glyph}` : ""),
        shape: "dot",
        size: meta.r,
        borderWidth: glyph ? 3 : 1,
        title: this.nodeTitle(n, meta),
        color: {
          background,
          border:
            n.polarity === "positive"
              ? t.positiveBorder
              : n.polarity === "negative"
                ? t.negativeBorder
                : t.border,
          highlight: { background, border: t.accent },
        },
      };
    }
    return {
      id: cid,
      shape: "custom",
      size: meta.r,
      title: this.nodeTitle(meta.node, meta),
      ctxRenderer: this.makeCtxRenderer(() => this.tlMeta.get(cid)),
      ...(seed ? { x: seed.x, y: seed.y } : {}),
    };
  }

  private timelineEdgeObject(
    key: string,
    e: { from: string; to: string; type: string; weight: number },
    isNew: boolean,
  ): VisItem {
    return {
      id: key,
      from: e.from,
      to: e.to,
      title: e.type,
      arrows: "to",
      width: 1 + 3 * e.weight,
      color: this.edgeColor(isNew),
      jgNew: isNew,
    };
  }

  private fullTimelineBuild(cur: DayGraph, prevDay: DayGraph | null) {
    this.tlNodes = new DataSet<VisItem>([...cur.nodes.keys()].map((cid) => this.timelineNodeObject(cid)));
    this.tlEdges = new DataSet<VisItem>(
      [...cur.edges.entries()].map(([key, e]) =>
        this.timelineEdgeObject(key, e, prevDay ? !prevDay.edges.has(key) : false),
      ),
    );
    const data = {
      nodes: this.tlNodes as unknown as never,
      edges: this.tlEdges as unknown as never,
    };
    if (!this.network) {
      this.createNetwork(data);
    } else {
      this.network.setOptions({ physics: { enabled: true }, edges: this.edgeOptions() });
      this.network.setData(data);
    }
    this.network!.once("stabilizationIterationsDone", () => {
      this.network?.setOptions({ physics: { enabled: false } });
      this.network?.fit({ animation: false });
    });
    this.network!.redraw();
    this.tlActive = true;
  }

  private incrementalTimelineStep(cur: DayGraph, prevDay: DayGraph | null) {
    const nodesDs = this.tlNodes!;
    const edgesDs = this.tlEdges!;
    const network = this.network!;

    const onScreen = new Set(nodesDs.getIds() as string[]);
    const kept: string[] = [];
    const addedIds: string[] = [];
    for (const cid of cur.nodes.keys()) {
      if (onScreen.has(cid)) kept.push(cid);
      else addedIds.push(cid);
    }
    const gone = [...onScreen].filter((cid) => !cur.nodes.has(cid));

    if (this.selectedId && gone.includes(this.selectedId)) {
      this.selectedId = null;
      this.onClick?.(null);
    }

    // Seed new nodes near their on-screen neighbors so they enter in context.
    const positions = network.getPositions(kept) as Record<string, { x: number; y: number }>;
    const keptPos = kept.map((id) => positions[id]).filter(Boolean);
    const centroid = keptPos.length
      ? {
          x: keptPos.reduce((s, p) => s + p.x, 0) / keptPos.length,
          y: keptPos.reduce((s, p) => s + p.y, 0) / keptPos.length,
        }
      : { x: 0, y: 0 };
    const jitter = (range: number) => (Math.random() - 0.5) * 2 * range;
    const seedFor = (cid: string) => {
      const neighborPos: { x: number; y: number }[] = [];
      for (const e of cur.edges.values()) {
        if (e.from === cid && positions[e.to]) neighborPos.push(positions[e.to]);
        if (e.to === cid && positions[e.from]) neighborPos.push(positions[e.from]);
      }
      if (neighborPos.length) {
        return {
          x: neighborPos.reduce((s, p) => s + p.x, 0) / neighborPos.length + jitter(40),
          y: neighborPos.reduce((s, p) => s + p.y, 0) / neighborPos.length + jitter(40),
        };
      }
      return { x: centroid.x + jitter(90), y: centroid.y + jitter(90) };
    };

    // Edges: remove gone, then update/add all current with explicit colors.
    const curEdgeKeys = new Set(cur.edges.keys());
    edgesDs.remove((edgesDs.getIds() as string[]).filter((k) => !curEdgeKeys.has(k)));
    edgesDs.update(
      [...cur.edges.entries()].map(([key, e]) =>
        this.timelineEdgeObject(key, e, prevDay ? !prevDay.edges.has(key) : false),
      ),
    );

    // Nodes: pin survivors (patches must NEVER contain x/y — that would snap
    // the node to a stale stored position), remove gone, add new with seeds.
    nodesDs.update(
      kept.map((cid) => {
        const meta = this.tlMeta.get(cid)!;
        return {
          id: cid,
          size: meta.r,
          title: this.nodeTitle(meta.node, meta),
          fixed: true,
        };
      }),
    );
    nodesDs.remove(gone);
    nodesDs.add(addedIds.map((cid) => this.timelineNodeObject(cid, seedFor(cid))));

    // Gentle settle: stabilize() drives ticks even with physics disabled and,
    // with stabilization.fit=false, never snaps the camera. After the settle,
    // glide-fit so the graph is always centered — the animation keeps the
    // continuity readable (surviving nodes visibly stay in place).
    const gen = ++this.settleGen;
    network.once("stabilizationIterationsDone", () => {
      if (gen !== this.settleGen || !this.network || !this.tlNodes) return;
      this.tlNodes.update(
        kept.filter((cid) => this.tlMeta.has(cid)).map((cid) => ({ id: cid, fixed: false })),
      );
      this.network.fit({ animation: { duration: 400, easingFunction: "easeInOutQuad" } });
    });
    network.stabilize(100);
    network.redraw();
  }

  private resetTimelineState() {
    this.tlActive = false;
    this.tlNodes = undefined;
    this.tlEdges = undefined;
    this.tlMeta.clear();
    this.selectedId = null;
  }

  // ─── Shared bits ──────────────────────────────────────────────────────────
  private nodeOptions() {
    return {
      font: { size: 12, color: this.theme.textNormal },
      borderWidth: 1.5,
    };
  }

  private edgeOptions() {
    const t = this.theme;
    return {
      font: {
        size: 10,
        align: "middle",
        color: t.textMuted,
        strokeWidth: 0,
        background: withAlpha(t.bgPrimary, 0.85),
      },
      smooth: { enabled: true, type: "dynamic", roundness: 0.5 },
      color: { color: withAlpha(t.textMuted, 0.45), highlight: t.accent, hover: t.accent },
    };
  }

  private nodeTitle(n: ServerNode, meta?: NodeMeta) {
    const base = `${n.type} | ${n.concept_id}\npolarity: ${n.polarity ?? "—"} | confidence: ${
      typeof n.confidence === "number" ? n.confidence.toFixed(2) : "—"
    }`;
    if (!meta) return base;
    const extras: string[] = [];
    if (meta.streak >= 2) extras.push(`${meta.streak} days in a row`);
    if (meta.isNew) extras.push("new today");
    return extras.length ? `${base}\n${extras.join(" · ")}` : base;
  }

  private toVisNodes(payload: ExtractResponse) {
    return payload.nodes.map((n) => {
      const r = nodeRadius(n.confidence);
      if (this.useCustomShapes) {
        const meta: NodeMeta = { node: n, r, streak: 1, isNew: false };
        return {
          id: n.node_id,
          shape: "custom",
          size: r,
          title: this.nodeTitle(n),
          ctxRenderer: this.makeCtxRenderer(() => meta),
        };
      }
      const glyph = POLARITY_GLYPH[n.polarity ?? ""];
      const t = this.theme;
      const border =
        n.polarity === "positive"
          ? t.positiveBorder
          : n.polarity === "negative"
            ? t.negativeBorder
            : t.border;
      const background = TYPE_COLORS[n.type] ?? FALLBACK_COLOR;
      return {
        id: n.node_id,
        label: (n.label || n.concept_id) + (glyph ? ` ${glyph}` : ""),
        shape: "dot",
        size: r,
        borderWidth: glyph ? 3 : 1,
        title: this.nodeTitle(n),
        color: {
          background,
          border,
          highlight: { background, border: t.accent },
        },
      };
    });
  }

  private toVisEdges(payload: ExtractResponse) {
    const showLabels = payload.edges.length <= 12;
    return payload.edges.map((e, i) => ({
      id: e.edge_id ?? `e${i}`,
      from: e.source_node_id,
      to: e.target_node_id,
      ...(showLabels ? { label: e.type } : {}),
      title: e.type,
      arrows: "to",
      width: 1 + 3 * (e.weight ?? 0.5),
    }));
  }

  private makeCtxRenderer(getMeta: () => NodeMeta | undefined) {
    return ({ ctx, x, y, state: { selected, hover } }: CtxArgs) => {
      const meta = getMeta();
      const r = meta?.r ?? R_MIN;
      return {
        nodeDimensions: { width: 2 * r, height: 2 * r },
        drawNode: () => {
          if (!meta) return;
          const { node: n, streak, isNew } = meta;
          const t = this.theme;

          if (isNew) {
            ctx.beginPath();
            ctx.arc(x, y, r + 4, 0, 2 * Math.PI);
            ctx.lineWidth = 2.5;
            ctx.strokeStyle = t.accent;
            ctx.stroke();
          }

          ctx.beginPath();
          ctx.arc(x, y, r, 0, 2 * Math.PI);
          ctx.fillStyle = TYPE_COLORS[n.type] ?? FALLBACK_COLOR;
          ctx.fill();
          ctx.lineWidth = selected ? 3 : hover ? 2.5 : 1.5;
          ctx.strokeStyle =
            selected || hover
              ? t.accent
              : n.polarity === "positive"
                ? t.positiveBorder
                : n.polarity === "negative"
                  ? t.negativeBorder
                  : t.border;
          ctx.stroke();

          if (n.polarity === "positive" || n.polarity === "negative") {
            const bx = x + r * 0.72;
            const by = y - r * 0.72;
            ctx.beginPath();
            ctx.arc(bx, by, BADGE_R, 0, 2 * Math.PI);
            ctx.fillStyle = t.badgeFill;
            ctx.fill();
            ctx.lineWidth = 1;
            ctx.strokeStyle = t.border;
            ctx.stroke();
            ctx.lineWidth = 1.7;
            ctx.lineCap = "round";
            ctx.strokeStyle = n.polarity === "positive" ? t.positiveGlyph : t.negativeGlyph;
            ctx.beginPath();
            if (n.polarity === "positive") {
              ctx.moveTo(bx - 2.7, by);
              ctx.lineTo(bx + 2.7, by);
              ctx.moveTo(bx, by - 2.7);
              ctx.lineTo(bx, by + 2.7);
            } else {
              ctx.moveTo(bx - 2.1, by - 2.1);
              ctx.lineTo(bx + 2.1, by + 2.1);
              ctx.moveTo(bx + 2.1, by - 2.1);
              ctx.lineTo(bx - 2.1, by + 2.1);
            }
            ctx.stroke();
          }

          if (streak >= 2) {
            const sx = x + r * 0.72;
            const sy = y + r * 0.72;
            ctx.beginPath();
            ctx.arc(sx, sy, STREAK_R, 0, 2 * Math.PI);
            ctx.fillStyle = t.badgeFill;
            ctx.fill();
            ctx.lineWidth = 1;
            ctx.strokeStyle = t.border;
            ctx.stroke();
            ctx.font = `700 8px ${t.fontFamily}`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillStyle = "#374151";
            ctx.fillText(String(streak), sx, sy + 0.5);
          }
        },
        drawExternalLabel: () => {
          if (!meta) return;
          const t = this.theme;
          const text = meta.node.label || meta.node.concept_id;
          ctx.font = `${selected ? "600 " : ""}12px ${t.fontFamily}`;
          ctx.textAlign = "center";
          ctx.textBaseline = "top";
          ctx.lineWidth = 3;
          ctx.strokeStyle = t.bgPrimary;
          ctx.strokeText(text, x, y + r + 4);
          ctx.fillStyle = selected ? t.textNormal : t.textMuted;
          ctx.fillText(text, x, y + r + 4);
        },
      };
    };
  }

  fit() {
    this.network?.fit({ animation: false });
  }

  redraw() {
    this.network?.redraw();
  }

  destroy() {
    this.network?.destroy();
    this.network = undefined;
    this.nodeIndex.clear();
    this.lastPayload = undefined;
    this.resetTimelineState();
    this.host.empty();
  }

  isEmpty() {
    return !this.network;
  }
}
