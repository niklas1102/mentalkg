import { App, TFile, TFolder } from "obsidian";
import { resolveNoteDate } from "./date-parse";
import { JournalGraphSettings } from "./settings";
import { Frontmatter, JournalDay } from "./types";

const FM_RE = /^---\n([\s\S]*?)\n---\s*\n?([\s\S]*)$/;

/** Frontmatter keys accepted as a mood value, in priority order. */
const MOOD_KEYS = ["mood_score", "mood", "rating"] as const;

function toMood(value: unknown): number | null {
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : null;
}

/**
 * Fallback regex parser for when the metadata cache hasn't populated yet.
 * Only splits off the frontmatter block and picks up date/mood aliases.
 */
export function parseFrontmatter(raw: string): { fm: Frontmatter; body: string } {
  const m = FM_RE.exec(raw);
  if (!m) return { fm: {}, body: raw };
  const fields: Record<string, string> = {};
  for (const line of m[1].split("\n")) {
    const colon = line.indexOf(":");
    if (colon < 0) continue;
    const key = line.slice(0, colon).trim();
    const value = line.slice(colon + 1).trim();
    if (key) fields[key] = value;
  }
  const fm: Frontmatter = {};
  if (fields.date) fm.date = fields.date;
  for (const k of MOOD_KEYS) {
    const mood = k in fields ? toMood(fields[k]) : null;
    if (mood !== null) {
      fm.mood_score = mood;
      break;
    }
  }
  return { fm, body: m[2].trim() };
}

/** Read a note's frontmatter (metadata cache preferred) and body. */
export async function readNote(
  app: App,
  file: TFile,
): Promise<{ fm: Frontmatter; body: string }> {
  const raw = await app.vault.cachedRead(file);
  const parsed = parseFrontmatter(raw);
  const cached = app.metadataCache.getFileCache(file)?.frontmatter;
  if (!cached) return parsed;
  const fm: Frontmatter = {};
  if (cached.date != null) fm.date = String(cached.date);
  for (const k of MOOD_KEYS) {
    const mood = k in cached ? toMood(cached[k]) : null;
    if (mood !== null) {
      fm.mood_score = mood;
      break;
    }
  }
  return { fm, body: parsed.body };
}

/** Build a JournalDay for a single note. Never fails on missing frontmatter. */
export async function loadJournalDay(app: App, file: TFile): Promise<JournalDay> {
  const { fm, body } = await readNote(app, file);
  const { date, source } = resolveNoteDate(file, fm.date);
  return {
    file,
    path: file.path,
    date,
    dateSource: source,
    mood_score: fm.mood_score ?? null,
    body,
  };
}

function collectFolderFiles(folder: TFolder): TFile[] {
  const out: TFile[] = [];
  for (const child of folder.children) {
    if (child instanceof TFile && child.extension === "md") out.push(child);
    else if (child instanceof TFolder) out.push(...collectFolderFiles(child));
  }
  return out;
}

/**
 * List journal notes sorted by date ascending. Notes are never skipped for
 * missing frontmatter — dates fall back to filename or creation time.
 */
export async function loadJournalDays(
  app: App,
  settings: JournalGraphSettings,
): Promise<JournalDay[]> {
  let files: TFile[];
  if (settings.treatAllAsJournal) {
    files = app.vault.getMarkdownFiles();
  } else {
    const folder = app.vault.getAbstractFileByPath(settings.journalFolder);
    if (!folder || !(folder instanceof TFolder)) return [];
    files = collectFolderFiles(folder);
  }
  const days = await Promise.all(files.map((f) => loadJournalDay(app, f)));
  days.sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : a.path < b.path ? -1 : 1));
  return days;
}
