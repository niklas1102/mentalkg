import { TFile } from "obsidian";

export type DateSource = "frontmatter" | "filename" | "ctime";

export type ResolvedDate = { date: string; source: DateSource };

const pad = (n: number) => String(n).padStart(2, "0");

/** Today's date in the user's local time zone, as YYYY-MM-DD. */
export function localToday(): string {
  const now = new Date();
  return `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
}

function isValidYmd(y: number, m: number, d: number): boolean {
  if (y < 1900 || y > 2100 || m < 1 || m > 12 || d < 1 || d > 31) return false;
  const dt = new Date(Date.UTC(y, m - 1, d));
  return dt.getUTCFullYear() === y && dt.getUTCMonth() === m - 1 && dt.getUTCDate() === d;
}

/**
 * Pull an ISO date out of an arbitrary string (frontmatter value or filename).
 * Supports: 2026-07-14 / 2026_07_14 / 2026.07.14 / 20260714,
 * 14.07.2026 / 14-07-2026 (day-first), and surrounding text
 * ("daily-2026-07-14", "2026-07-14 Monday").
 */
export function parseDateFrom(s: string): string | null {
  const text = s.trim();

  // Year-first: YYYY-MM-DD with -, _, . or / separators, anywhere in the string.
  let m = /(\d{4})[-_./](\d{1,2})[-_./](\d{1,2})/.exec(text);
  if (m) {
    const [y, mo, d] = [Number(m[1]), Number(m[2]), Number(m[3])];
    if (isValidYmd(y, mo, d)) return `${y}-${pad(mo)}-${pad(d)}`;
  }

  // Day-first: DD.MM.YYYY / DD-MM-YYYY / DD/MM/YYYY.
  m = /(\d{1,2})[-./](\d{1,2})[-./](\d{4})/.exec(text);
  if (m) {
    const [d, mo, y] = [Number(m[1]), Number(m[2]), Number(m[3])];
    if (isValidYmd(y, mo, d)) return `${y}-${pad(mo)}-${pad(d)}`;
  }

  // Compact: YYYYMMDD as a standalone token.
  m = /(?:^|\D)(\d{4})(\d{2})(\d{2})(?:\D|$)/.exec(text);
  if (m) {
    const [y, mo, d] = [Number(m[1]), Number(m[2]), Number(m[3])];
    if (isValidYmd(y, mo, d)) return `${y}-${pad(mo)}-${pad(d)}`;
  }

  return null;
}

/**
 * Resolve the date of a note: frontmatter `date` first, then filename,
 * then the file's creation time.
 */
export function resolveNoteDate(file: TFile, fmDate: unknown): ResolvedDate {
  if (fmDate != null) {
    const parsed = parseDateFrom(String(fmDate));
    if (parsed) return { date: parsed, source: "frontmatter" };
  }
  const fromName = parseDateFrom(file.basename);
  if (fromName) return { date: fromName, source: "filename" };
  const c = new Date(file.stat.ctime);
  return {
    date: `${c.getFullYear()}-${pad(c.getMonth() + 1)}-${pad(c.getDate())}`,
    source: "ctime",
  };
}
