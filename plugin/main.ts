import { Debouncer, Editor, MarkdownView, Notice, Plugin, TFile, debounce, normalizePath } from "obsidian";
import { localToday } from "./src/date-parse";
import { DEFAULT_SETTINGS, JournalGraphSettingTab, JournalGraphSettings } from "./src/settings";
import { GraphView } from "./src/view";
import { VIEW_TYPE } from "./src/types";

export default class JournalGraphPlugin extends Plugin {
  settings: JournalGraphSettings = { ...DEFAULT_SETTINGS };
  private autoRefresh: Debouncer<[], void> = debounce(
    () => this.refreshOpenView(),
    DEFAULT_SETTINGS.autoRefreshDelayMs,
    true,
  );

  async onload() {
    await this.loadSettings();
    this.addSettingTab(new JournalGraphSettingTab(this.app, this));

    this.registerView(
      VIEW_TYPE,
      (leaf) => new GraphView(leaf, () => this.settings, () => this.newEntryToday()),
    );

    this.addRibbonIcon("network", "Extract graph for current note", () =>
      this.runCurrentNote(),
    );

    this.addCommand({
      id: "build-graph-from-note",
      name: "Extract graph for current note",
      callback: () => this.runCurrentNote(),
    });

    this.addCommand({
      id: "open-timeline",
      name: "Open journal timeline",
      callback: () => this.openTimeline(),
    });

    this.addCommand({
      id: "new-journal-entry",
      name: "New journal entry for today",
      callback: () => this.newEntryToday(),
    });

    this.addCommand({
      id: "paste-as-live-typing",
      name: "Paste clipboard as live typing (demo)",
      // No V and no Option: V-combos collide with clipboard managers,
      // Option-combos break on non-US macOS layouts.
      hotkeys: [{ modifiers: ["Mod", "Ctrl"], key: "j" }],
      callback: () => void this.typeClipboard(),
    });

    this.registerEvent(
      this.app.workspace.on("editor-change", () => {
        if (this.settings.autoRefresh) this.autoRefresh();
      }),
    );
  }

  async loadSettings() {
    this.settings = { ...DEFAULT_SETTINGS, ...((await this.loadData()) ?? {}) };
    this.rebuildDebounce();
  }

  async saveSettings() {
    await this.saveData(this.settings);
    this.rebuildDebounce();
  }

  private rebuildDebounce() {
    this.autoRefresh.cancel();
    this.autoRefresh = debounce(
      () => this.refreshOpenView(),
      this.settings.autoRefreshDelayMs,
      true,
    );
  }

  /**
   * Auto-refresh: re-extract the active note, but only if the panel is already
   * open AND showing Current-note mode — typing must never yank Timeline away.
   */
  private refreshOpenView() {
    const leaf = this.app.workspace.getLeavesOfType(VIEW_TYPE)[0];
    if (!leaf || !(leaf.view instanceof GraphView)) return;
    if (leaf.view.getMode() !== "current") return;
    const file = this.app.workspace.getActiveViewOfType(MarkdownView)?.file ?? null;
    if (!file || file.extension !== "md") return;
    void leaf.view.runCurrentNote(file);
  }

  private async runCurrentNote() {
    const view = await this.activateView();
    const file = this.app.workspace.getActiveFile();
    await view.runCurrentNote(file);
  }

  private async openTimeline() {
    const view = await this.activateView();
    view.setMode("timeline");
  }

  private typingTimer: number | null = null;

  /**
   * Electron clipboard first (synchronous, reliable on desktop); the web
   * clipboard API only as a time-boxed fallback — it can hang without ever
   * resolving in Electron, which must not freeze the command silently.
   */
  private async readClipboardText(): Promise<string> {
    try {
      const req = (window as unknown as { require?: (m: string) => { clipboard?: { readText?: () => string } } }).require;
      const t = req?.("electron")?.clipboard?.readText?.();
      if (typeof t === "string" && t) return t;
    } catch {
      // no Electron require — fall through
    }
    try {
      const t = await Promise.race([
        navigator.clipboard.readText(),
        new Promise<string>((resolve) => window.setTimeout(() => resolve(""), 1500)),
      ]);
      return t ?? "";
    } catch {
      return "";
    }
  }

  /**
   * Types the clipboard content into the editor over ~5 seconds, like sped-up
   * typing. Runs through the normal editor API, so auto-refresh fires
   * afterwards exactly as with real typing. Works even when focus sits in the
   * note title or the file explorer — it grabs the active note's editor.
   */
  private async typeClipboard() {
    if (this.typingTimer !== null) return;
    const view = this.app.workspace.getActiveViewOfType(MarkdownView);
    if (!view) {
      new Notice("Open a note first, then run the live-typing command.");
      return;
    }
    const editor: Editor = view.editor;
    const text = await this.readClipboardText();
    if (!text.trim()) {
      new Notice("Clipboard is empty — copy the entry text first.");
      return;
    }
    editor.focus();
    const totalMs = 5000;
    const tickMs = 45;
    const chunkSize = Math.max(2, Math.ceil(text.length / (totalMs / tickMs)));
    let pos = 0;
    // Track our own insertion offset instead of always appending at the end of
    // the document — a concurrent edit (e.g. a stray paste from a clipboard
    // manager) can then no longer interleave into the middle of our text.
    let insertOffset = editor.posToOffset({
      line: editor.lastLine(),
      ch: editor.getLine(editor.lastLine()).length,
    });
    this.typingTimer = window.setInterval(() => {
      if (pos >= text.length || !this.app.workspace.getActiveViewOfType(MarkdownView)) {
        if (this.typingTimer !== null) window.clearInterval(this.typingTimer);
        this.typingTimer = null;
        return;
      }
      const chunk = text.slice(pos, pos + chunkSize);
      pos += chunkSize;
      editor.replaceRange(chunk, editor.offsetToPos(insertOffset));
      insertOffset += chunk.length;
      editor.setCursor(editor.offsetToPos(insertOffset));
    }, tickMs);
    this.registerInterval(this.typingTimer);
  }

  async newEntryToday() {
    const folder = this.settings.journalFolder || "journal";
    const path = normalizePath(`${folder}/${localToday()}.md`);
    let file = this.app.vault.getAbstractFileByPath(path);
    if (!(file instanceof TFile)) {
      try {
        await this.app.vault.createFolder(normalizePath(folder));
      } catch {
        // folder already exists
      }
      file = await this.app.vault.create(path, "");
    }
    await this.app.workspace.getLeaf(false).openFile(file as TFile);
    const view = await this.activateView();
    await view.runCurrentNote(file as TFile);
  }

  private async activateView(): Promise<GraphView> {
    const existing = this.app.workspace.getLeavesOfType(VIEW_TYPE)[0];
    const leaf = existing ?? this.app.workspace.getRightLeaf(false);
    if (!leaf) throw new Error("Could not open right-side leaf.");
    if (!existing) await leaf.setViewState({ type: VIEW_TYPE, active: true });
    this.app.workspace.revealLeaf(leaf);
    return leaf.view as GraphView;
  }
}
