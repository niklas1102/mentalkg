import { ItemView, Notice, TFile, WorkspaceLeaf, normalizePath } from "obsidian";
import { aggregate } from "./aggregate";
import { localToday } from "./date-parse";
import { ExtractCache, extract } from "./extract";
import { ExportProgressModal } from "./export-modal";
import { GraphRenderer, TimelineStats } from "./graph-render";
import { loadJournalDay, loadJournalDays } from "./journal";
import { MoodChart } from "./mood-chart";
import { JournalGraphSettings } from "./settings";
import { readThemeColors } from "./theme";
import { stripMarkdownNoise } from "./text-clean";
import {
  ExtractResponse,
  FALLBACK_COLOR,
  HONESTY,
  JournalDay,
  POLARITY_GLYPH,
  ServerNode,
  summaryPath,
  TYPE_COLORS,
  VIEW_TYPE,
} from "./types";

type Mode = "current" | "timeline";

const MIN_TEXT_LENGTH = 15;

const DATE_SOURCE_LABEL: Record<JournalDay["dateSource"], string> = {
  frontmatter: "from frontmatter",
  filename: "from filename",
  ctime: "from file creation time",
};

const WEEKDAYS_SHORT = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const weekdayShort = (date: string) => {
  const [y, m, d] = date.split("-").map(Number);
  return WEEKDAYS_SHORT[new Date(Date.UTC(y, m - 1, d)).getUTCDay()] ?? "";
};

const PLAY_INTERVAL_MS = 1800;
const EMPTY_SET: ReadonlySet<string> = new Set();

export class GraphView extends ItemView {
  private mode: Mode = "current";
  private days: JournalDay[] = [];
  private dayIdx = 0;

  private root!: HTMLElement;
  private modeBar!: HTMLElement;
  private msgEl!: HTMLElement;
  private dateSrcEl!: HTMLElement;
  private retryEl!: HTMLElement;
  private liveEl!: HTMLElement;

  // Timeline-only controls
  private dateLabel!: HTMLElement;
  private slider!: HTMLInputElement;
  private prevBtn!: HTMLButtonElement;
  private nextBtn!: HTMLButtonElement;
  private playBtn!: HTMLButtonElement;
  private exportBtn!: HTMLButtonElement;
  private prefetchStatusEl!: HTMLElement;
  private moodHost!: HTMLElement;
  private mood?: MoodChart;
  private playTimer: number | null = null;

  // Memoized per-day concept sets for streak computation (payloads are immutable).
  private conceptSetMemo = new Map<string, Set<string>>();

  // Shared graph + cards
  private graphWrap!: HTMLElement;
  private graphHost!: HTMLElement;
  private legendEl!: HTMLElement;
  private detailEl!: HTMLElement;
  private predictedEl!: HTMLElement;

  private renderer?: GraphRenderer;
  private cache = new ExtractCache();
  private resizeObs?: ResizeObserver;
  private lastAction?: () => Promise<void>;

  // Current-note refresh guard: at most one extraction in flight, one queued.
  private busy = false;
  private queuedJob: (() => Promise<void>) | null = null;

  // Timeline prefetch: generation token cancels the loop, dedupe map shares
  // in-flight requests between prefetch, slider, and export.
  private prefetchGen = 0;
  private inFlightByPath = new Map<string, Promise<ExtractResponse>>();
  private failedPaths = new Set<string>();

  constructor(
    leaf: WorkspaceLeaf,
    private getSettings: () => JournalGraphSettings,
    private onNewEntry: () => Promise<void>,
  ) {
    super(leaf);
  }

  getViewType() {
    return VIEW_TYPE;
  }
  getDisplayText() {
    return "Journal Graph";
  }
  getIcon() {
    return "network";
  }
  getMode(): Mode {
    return this.mode;
  }

  // ─── Layout ─────────────────────────────────────────────────────────────
  async onOpen() {
    this.root = this.contentEl;
    this.root.empty();
    this.root.addClass("jg-root");

    const header = this.root.createDiv({ cls: "jg-header" });
    header.createSpan({ cls: "jg-title", text: "Journal Graph" });
    this.liveEl = header.createSpan({ cls: "jg-live is-hidden" });
    const newBtn = header.createEl("button", { cls: "jg-btn jg-new-entry", text: "+ New entry" });
    newBtn.onclick = () => void this.onNewEntry();

    this.modeBar = this.root.createDiv({ cls: "jg-modebar" });
    this.buildModeBar();

    this.msgEl = this.root.createDiv({ cls: "jg-msg" });

    this.retryEl = this.root.createDiv({ cls: "jg-retry is-hidden" });
    this.dateSrcEl = this.root.createDiv({ cls: "jg-datesrc" });

    // Timeline controls (built but hidden until timeline mode)
    const tl = this.root.createDiv({ cls: "jg-timeline-controls" });
    const row1 = tl.createDiv({ cls: "jg-row" });
    this.prevBtn = row1.createEl("button", { text: "◀", cls: "jg-btn" });
    this.playBtn = row1.createEl("button", { text: "▶ Play", cls: "jg-btn jg-play" });
    this.nextBtn = row1.createEl("button", { text: "▶", cls: "jg-btn" });
    this.dateLabel = row1.createDiv({ cls: "jg-date-label" });

    this.slider = tl.createEl("input", { type: "range", cls: "jg-slider" });
    this.slider.min = "0";
    this.slider.max = "0";
    this.slider.value = "0";

    this.prefetchStatusEl = tl.createDiv({ cls: "jg-prefetch" });
    this.moodHost = tl.createDiv({ cls: "jg-mood" });
    this.exportBtn = tl.createEl("button", {
      text: "Export therapist summary",
      cls: "jg-btn jg-btn-primary jg-export-btn",
    });

    this.prevBtn.onclick = () => {
      this.stopPlayback();
      void this.gotoDay(this.dayIdx - 1);
    };
    this.nextBtn.onclick = () => {
      this.stopPlayback();
      void this.gotoDay(this.dayIdx + 1);
    };
    this.playBtn.onclick = () => this.togglePlayback();
    this.slider.oninput = () => {
      this.stopPlayback();
      void this.gotoDay(Number(this.slider.value));
    };
    this.exportBtn.onclick = () => this.runExport();

    // Shared graph area — legend is rendered as its own block BELOW the canvas
    // (no overlay) so it can never obscure nodes.
    this.graphWrap = this.root.createDiv({ cls: "jg-graph-wrap" });
    this.graphHost = this.graphWrap.createDiv({ cls: "jg-graph" });
    this.legendEl = this.root.createDiv({ cls: "jg-legend" });
    this.buildLegend();

    this.root.createDiv({ cls: "jg-section-title", text: "Selected node" });
    this.detailEl = this.root.createDiv({ cls: "jg-detail" });
    this.showEmptyDetail();

    this.root.createDiv({ cls: "jg-section-title", text: "Predicted concepts" });
    this.predictedEl = this.root.createDiv({ cls: "jg-predicted" });
    this.predictedEl.setText("—");

    const note = this.root.createDiv({ cls: "jg-note" });
    note.setText(HONESTY);

    this.renderer = new GraphRenderer(this.graphHost);
    this.resizeObs = new ResizeObserver(() => this.renderer?.redraw());
    this.resizeObs.observe(this.graphHost);

    this.registerEvent(this.app.workspace.on("css-change", () => this.onThemeChange()));

    this.applyMode();
  }

  async onClose() {
    this.stopPlayback();
    this.cancelPrefetch();
    this.renderer?.destroy();
    this.renderer = undefined;
    this.resizeObs?.disconnect();
    this.resizeObs = undefined;
    this.mood?.destroy();
    this.mood = undefined;
    this.cache.clear();
    this.conceptSetMemo.clear();
  }

  private onThemeChange() {
    const theme = readThemeColors();
    this.renderer?.setTheme(theme);
    if (this.mode === "timeline" && this.days.length) {
      this.mood?.render(this.days, this.dayIdx, theme);
    }
    this.buildLegend();
  }

  // ─── Mode toggle ────────────────────────────────────────────────────────
  private buildModeBar() {
    this.modeBar.empty();
    const mkBtn = (label: string, value: Mode) => {
      const b = this.modeBar.createEl("button", { text: label, cls: "jg-mode-btn" });
      if (this.mode === value) b.addClass("is-active");
      b.onclick = () => {
        if (this.mode !== value) {
          this.mode = value;
          this.applyMode();
        }
      };
      return b;
    };
    mkBtn("Current note", "current");
    mkBtn("Timeline", "timeline");
  }

  private applyMode() {
    this.buildModeBar();
    this.buildLegend();
    const isTimeline = this.mode === "timeline";
    this.root
      .querySelector(".jg-timeline-controls")
      ?.toggleClass("is-hidden", !isTimeline);
    if (isTimeline) {
      void this.activateTimeline();
    } else {
      this.stopPlayback();
      this.cancelPrefetch();
      this.showMessage(
        this.getSettings().autoRefresh
          ? "Open a note and start writing — the graph updates as you type."
          : 'Click "+ New entry" above, or use the network ribbon icon to extract the active note.',
        false,
      );
      this.clearOutputs();
    }
    this.updateHeaderStatus();
  }

  // ─── Playback (auto-advance through the timeline) ───────────────────────
  private togglePlayback() {
    if (this.playTimer !== null) {
      this.stopPlayback();
      return;
    }
    if (!this.days.length) return;
    if (this.dayIdx >= this.days.length - 1) void this.gotoDay(0);
    this.playBtn.setText("■ Stop");
    this.playBtn.addClass("is-playing");
    this.playTimer = window.setInterval(() => {
      const next = this.dayIdx + 1;
      if (next >= this.days.length) {
        this.stopPlayback();
        return;
      }
      // Only advance onto cached (or empty) days; skip the tick while prefetch catches up.
      const d = this.days[next];
      if (this.cache.has(d.path) || stripMarkdownNoise(d.body).length < MIN_TEXT_LENGTH) {
        void this.gotoDay(next);
      }
    }, PLAY_INTERVAL_MS);
    this.registerInterval(this.playTimer);
  }

  private stopPlayback() {
    if (this.playTimer !== null) {
      window.clearInterval(this.playTimer);
      this.playTimer = null;
    }
    if (this.playBtn) {
      this.playBtn.setText("▶ Play");
      this.playBtn.removeClass("is-playing");
    }
  }

  setMode(mode: Mode) {
    if (this.mode === mode) {
      if (mode === "timeline") void this.activateTimeline();
      return;
    }
    this.mode = mode;
    this.applyMode();
  }

  // ─── Live indicator ─────────────────────────────────────────────────────
  private updateHeaderStatus() {
    if (!this.liveEl) return;
    this.liveEl.empty();
    if (this.busy) {
      this.liveEl.removeClass("is-hidden");
      this.liveEl.createSpan({ cls: "jg-spinner" });
      this.liveEl.createSpan({ text: "extracting…" });
    } else if (this.getSettings().autoRefresh && this.mode === "current") {
      this.liveEl.removeClass("is-hidden");
      this.liveEl.createSpan({ cls: "jg-live-dot" });
      this.liveEl.createSpan({ text: "live" });
    } else {
      this.liveEl.addClass("is-hidden");
    }
  }

  private async runGuarded(job: () => Promise<void>) {
    if (this.busy) {
      this.queuedJob = job; // latest wins; anything older is dropped
      return;
    }
    this.busy = true;
    this.updateHeaderStatus();
    try {
      await job();
    } finally {
      this.busy = false;
      this.updateHeaderStatus();
      const next = this.queuedJob;
      this.queuedJob = null;
      if (next) void this.runGuarded(next);
    }
  }

  // ─── Public entrypoints used by JournalGraphPlugin ──────────────────────
  async runCurrentNote(file: TFile | null) {
    if (this.mode !== "current") {
      this.mode = "current";
      this.applyMode();
    }
    if (!file) {
      new Notice("No active note. Open a markdown note first.");
      return;
    }
    this.lastAction = () => this.runCurrentNote(file);
    await this.runGuarded(() => this.extractCurrentNote(file));
  }

  private async extractCurrentNote(file: TFile) {
    this.hideRetry();
    const day = await loadJournalDay(this.app, file);
    this.dateSrcEl.setText(`date ${day.date} · ${DATE_SOURCE_LABEL[day.dateSource]}`);

    const text = stripMarkdownNoise(day.body);
    if (text.length < MIN_TEXT_LENGTH) {
      this.showMessage("Write a few sentences — the graph appears once there is enough text.", false);
      return;
    }
    this.showMessage("Extracting graph…", false);
    try {
      const payload = await extract(this.getSettings().serverUrl, text);
      this.cache.set(file.path, payload);
      this.renderPayload(payload);
      if (payload.nodes.length === 0) {
        this.showMessage("No concepts found in this text yet — keep writing.", false);
      } else {
        this.showMessage(
          `Extracted ${payload.nodes.length} nodes, ${payload.edges.length} edges.`,
          false,
        );
      }
    } catch (err) {
      this.handleFetchError(err);
    }
  }

  // ─── Timeline mode ──────────────────────────────────────────────────────
  private async activateTimeline() {
    const settings = this.getSettings();
    this.dateSrcEl.setText("");
    this.showMessage("Loading journal notes…", false);
    try {
      this.days = await loadJournalDays(this.app, settings);
      this.conceptSetMemo.clear();
    } catch (err) {
      this.showMessage(`Could not read journal notes: ${(err as Error).message}`, true);
      return;
    }
    if (!this.days.length) {
      this.showMessage(
        settings.treatAllAsJournal
          ? "No markdown notes found in the vault."
          : `No notes found under "${settings.journalFolder}/". Change the folder in the plugin settings, or enable "treat all notes as journal entries".`,
        true,
      );
      this.slider.max = "0";
      this.slider.value = "0";
      this.dateLabel.setText("");
      this.moodHost.empty();
      this.clearOutputs();
      return;
    }

    this.dayIdx = Math.min(this.dayIdx, this.days.length - 1);
    this.slider.max = String(this.days.length - 1);
    this.slider.value = String(this.dayIdx);

    this.mood ??= new MoodChart(this.moodHost);
    this.mood.render(this.days, this.dayIdx, readThemeColors());

    await this.gotoDay(this.dayIdx);
    this.startPrefetch();
  }

  /** Cache-respecting fetch shared by the slider, the prefetch loop, and the export. */
  private fetchDay(day: JournalDay): Promise<ExtractResponse> {
    const cached = this.cache.get(day.path);
    if (cached) return Promise.resolve(cached);
    const existing = this.inFlightByPath.get(day.path);
    if (existing) return existing;
    const p = extract(this.getSettings().serverUrl, stripMarkdownNoise(day.body))
      .then((payload) => {
        this.cache.set(day.path, payload);
        return payload;
      })
      .finally(() => this.inFlightByPath.delete(day.path));
    this.inFlightByPath.set(day.path, p);
    return p;
  }

  private cancelPrefetch() {
    this.prefetchGen += 1;
    this.prefetchStatusEl?.setText("");
  }

  private startPrefetch() {
    this.prefetchGen += 1;
    this.failedPaths.clear();
    void this.prefetchLoop(this.prefetchGen);
  }

  private async prefetchLoop(gen: number) {
    const eligible = this.days.filter(
      (d) => stripMarkdownNoise(d.body).length >= MIN_TEXT_LENGTH,
    );
    const total = eligible.length;
    while (this.prefetchGen === gen) {
      const pending = eligible.filter(
        (d) => !this.cache.has(d.path) && !this.failedPaths.has(d.path),
      );
      if (!pending.length) break;
      // The day under the slider jumps the queue.
      const current = this.days[this.dayIdx];
      const target = pending.find((d) => d.path === current?.path) ?? pending[0];
      this.prefetchStatusEl.setText(`caching ${total - pending.length + 1}/${total}…`);
      try {
        await this.fetchDay(target);
      } catch {
        this.failedPaths.add(target.path);
      }
    }
    if (this.prefetchGen === gen) this.prefetchStatusEl.setText("");
  }

  private async gotoDay(rawIdx: number) {
    if (!this.days.length) return;
    const idx = Math.max(0, Math.min(this.days.length - 1, rawIdx));
    this.dayIdx = idx;
    this.slider.value = String(idx);
    const day = this.days[idx];
    this.dateLabel.setText(
      `${weekdayShort(day.date)} · ${day.date}${
        typeof day.mood_score === "number" ? " · mood " + day.mood_score.toFixed(1) : ""
      } · ${idx + 1}/${this.days.length}`,
    );
    this.dateSrcEl.setText(`date ${day.date} · ${DATE_SOURCE_LABEL[day.dateSource]}`);
    this.mood?.render(this.days, idx, readThemeColors());
    this.lastAction = () => this.gotoDay(idx);
    this.hideRetry();

    const text = stripMarkdownNoise(day.body);
    if (text.length < MIN_TEXT_LENGTH) {
      this.showMessage(`${day.date} has (almost) no text — skipping extraction.`, false);
      this.clearOutputs(false);
      return;
    }

    const cached = this.cache.get(day.path);
    if (cached) {
      this.renderTimelineDay(cached, idx, day);
      return;
    }
    this.showMessage(`Extracting ${day.date}…`, false);
    try {
      const payload = await this.fetchDay(day);
      if (this.dayIdx !== idx) return; // user scrubbed away — drop the stale result
      this.renderTimelineDay(payload, idx, day);
    } catch (err) {
      if (this.dayIdx === idx) this.handleFetchError(err);
    }
  }

  /** Render one timeline day as a diff against the previous day. */
  private renderTimelineDay(payload: ExtractResponse, idx: number, day: JournalDay) {
    if (!this.renderer) return;
    const prev = idx > 0 ? this.cache.get(this.days[idx - 1].path) ?? null : null;
    const streaks = this.computeStreaks(idx, payload);
    const stats: TimelineStats = this.renderer.renderTimeline(payload, prev, streaks, (node) => {
      if (node) this.showDetail(node);
      else this.showEmptyDetail();
    });
    this.renderPredicted(payload);
    if (stats.firstDay) {
      this.showMessage(
        idx === 0
          ? `${day.date}: ${payload.nodes.length} concepts (first entry).`
          : `${day.date}: ${payload.nodes.length} concepts (previous day not cached yet).`,
        false,
      );
    } else {
      this.showMessage(
        `${day.date} · ${stats.added} new · ${stats.recurring} recurring · ${stats.removed} gone`,
        false,
      );
    }
  }

  /** Consecutive-day streak per concept in the current payload, walking back through the cache. */
  private computeStreaks(idx: number, payload: ExtractResponse): Map<string, number> {
    const streaks = new Map<string, number>();
    for (const n of payload.nodes) {
      if (streaks.has(n.concept_id)) continue;
      let s = 1;
      for (let j = idx - 1; j >= 0 && idx - j <= 60; j--) {
        const set = this.conceptSet(j);
        if (!set || !set.has(n.concept_id)) break;
        s += 1;
      }
      streaks.set(n.concept_id, s);
    }
    return streaks;
  }

  /** Concept ids of day i, or undefined while that day is not cached yet. */
  private conceptSet(i: number): ReadonlySet<string> | undefined {
    const day = this.days[i];
    if (!day) return EMPTY_SET;
    if (stripMarkdownNoise(day.body).length < MIN_TEXT_LENGTH) return EMPTY_SET;
    const memo = this.conceptSetMemo.get(day.path);
    if (memo) return memo;
    const payload = this.cache.get(day.path);
    if (!payload) return undefined;
    const set = new Set(payload.nodes.map((n) => n.concept_id));
    this.conceptSetMemo.set(day.path, set);
    return set;
  }

  // ─── Therapist export ───────────────────────────────────────────────────
  private async runExport() {
    if (!this.days.length) {
      new Notice("No journal notes loaded yet. Open Timeline first.");
      return;
    }
    this.stopPlayback();
    this.cancelPrefetch();
    const modal = new ExportProgressModal(this.app);
    modal.open();
    this.exportBtn.disabled = true;

    try {
      const ok: { day: JournalDay; payload: ExtractResponse }[] = [];
      let skipped = 0;
      const total = this.days.length;
      for (let i = 0; i < total; i++) {
        if (modal.isCancelled()) return;
        modal.setProgress(i + 1, total);
        const day = this.days[i];
        const text = stripMarkdownNoise(day.body);
        if (text.length < MIN_TEXT_LENGTH) {
          skipped += 1;
          continue;
        }
        try {
          ok.push({ day, payload: await this.fetchDay(day) });
        } catch {
          skipped += 1;
        }
      }
      if (modal.isCancelled()) return;

      if (ok.length === 0) {
        modal.markFinished();
        this.showMessage(
          `Could not extract any entry (server unreachable at ${this.getSettings().serverUrl}, or all notes empty). No summary written.`,
          true,
        );
        new Notice("Therapist export failed — no successful entries.");
        return;
      }

      const { markdown, succeeded } = aggregate(ok, skipped);
      const path = normalizePath(summaryPath(localToday()));
      const existing = this.app.vault.getAbstractFileByPath(path);
      let target: TFile;
      if (existing instanceof TFile) {
        await this.app.vault.modify(existing, markdown);
        target = existing;
      } else {
        target = await this.app.vault.create(path, markdown);
      }
      modal.markFinished();
      await this.app.workspace.getLeaf(true).openFile(target, { state: { mode: "preview" } });
      new Notice(`Therapist summary written (${succeeded} ok, ${skipped} skipped).`);
    } finally {
      this.exportBtn.disabled = false;
      modal.markFinished();
      if (this.mode === "timeline") this.startPrefetch();
    }
  }

  // ─── Shared rendering helpers ───────────────────────────────────────────
  private renderPayload(payload: ExtractResponse) {
    if (!this.renderer) return;
    this.renderer.render(payload, (node) => {
      if (node) this.showDetail(node);
      else this.showEmptyDetail();
    });
    this.renderPredicted(payload);
  }

  private renderPredicted(payload: ExtractResponse) {
    const items = payload.predicted_concepts;
    this.predictedEl.empty();
    if (!items?.length) {
      this.predictedEl.setText("—");
      return;
    }
    const meta = new Map<string, { type: string; label: string }>();
    for (const nd of payload.nodes) {
      if (!meta.has(nd.concept_id)) {
        meta.set(nd.concept_id, { type: nd.type, label: nd.label || nd.concept_id });
      }
    }
    const wrap = this.predictedEl.createDiv({ cls: "jg-chips" });
    for (const p of items.slice(0, 10)) {
      const m = meta.get(p.concept_id);
      const chip = wrap.createDiv({ cls: "jg-chip" });
      const top = chip.createDiv({ cls: "jg-chip-top" });
      const dot = top.createSpan({ cls: "jg-chip-dot" });
      dot.style.background = TYPE_COLORS[m?.type ?? ""] ?? FALLBACK_COLOR;
      top.createSpan({ cls: "jg-chip-label", text: m?.label ?? p.concept_id.replace(/_/g, " ") });
      const bar = chip.createDiv({ cls: "jg-chip-bar" });
      const barFill = bar.createDiv({ cls: "jg-chip-bar-fill" });
      barFill.style.width = `${Math.round(100 * Math.max(0, Math.min(1, p.score)))}%`;
    }
    if (items.length > 10) {
      this.predictedEl.createDiv({
        cls: "jg-predicted-more",
        text: `+ ${items.length - 10} more`,
      });
    }
  }

  private clearOutputs(clearDateSrc = true) {
    this.renderer?.destroy();
    this.predictedEl.setText("—");
    this.showEmptyDetail();
    if (clearDateSrc) this.dateSrcEl.setText("");
  }

  private showEmptyDetail() {
    this.detailEl.empty();
    this.detailEl.addClass("jg-detail-empty");
    this.detailEl.setText("Click a node to inspect its fields.");
  }

  private showDetail(n: ServerNode) {
    this.detailEl.empty();
    this.detailEl.removeClass("jg-detail-empty");
    const kv = this.detailEl.createDiv({ cls: "jg-kv" });
    const row = (k: string, v: string) => {
      kv.createDiv({ cls: "jg-kv-key", text: k });
      kv.createDiv({ text: v });
    };
    const glyph = POLARITY_GLYPH[n.polarity ?? ""];
    row("concept_id", n.concept_id);
    row("type", n.type);
    row("label", n.label);
    row("polarity", n.polarity ? `${n.polarity}${glyph ? ` (${glyph})` : ""}` : "—");
    row("temporal_status", n.temporal_status ?? "—");
    row("confidence", typeof n.confidence === "number" ? n.confidence.toFixed(3) : "—");
    if (n.time_anchor) {
      row(
        "time_anchor",
        `${n.time_anchor.text ?? "—"} (day_offset=${n.time_anchor.day_offset ?? "—"})`,
      );
    }
  }

  private buildLegend() {
    this.legendEl.empty();
    const title = this.legendEl.createDiv({ cls: "jg-legend-title" });
    title.setText("color = type · badge = polarity");
    const typeRow = this.legendEl.createDiv({ cls: "jg-legend-types" });
    for (const [type, color] of Object.entries(TYPE_COLORS)) {
      const row = typeRow.createSpan({ cls: "jg-legend-row" });
      const sw = row.createSpan({ cls: "jg-swatch" });
      sw.style.background = color;
      row.createSpan({ text: type });
    }
    const badgeRow = this.legendEl.createDiv({ cls: "jg-legend-badges" });
    const pos = badgeRow.createSpan({ cls: "jg-legend-row" });
    pos.createSpan({ cls: "jg-badge jg-badge-pos", text: "+" });
    pos.createSpan({ text: "positive" });
    const neg = badgeRow.createSpan({ cls: "jg-legend-row" });
    neg.createSpan({ cls: "jg-badge jg-badge-neg", text: "×" });
    neg.createSpan({ text: "negative" });
    badgeRow.createSpan({
      cls: "jg-legend-note",
      text: "plain = neutral · size = confidence",
    });
    if (this.mode === "timeline") {
      this.legendEl.createDiv({
        cls: "jg-legend-note",
        text: "ring = new today · number = days in a row",
      });
    }
  }

  private showMessage(text: string, isError: boolean) {
    this.msgEl.setText(text);
    this.msgEl.toggleClass("jg-msg-error", isError);
    if (!isError) this.hideRetry();
  }

  private hideRetry() {
    this.retryEl.empty();
    this.retryEl.addClass("is-hidden");
  }

  private handleFetchError(err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    const url = this.getSettings().serverUrl;
    this.showMessage(
      `Could not reach the extraction server at ${url} — is it running? (${msg})`,
      true,
    );
    this.retryEl.empty();
    this.retryEl.removeClass("is-hidden");
    const btn = this.retryEl.createEl("button", { text: "Retry", cls: "jg-btn" });
    btn.onclick = () => {
      const action = this.lastAction;
      if (action) void action();
    };
  }
}
