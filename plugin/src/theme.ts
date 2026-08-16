export type ThemeColors = {
  bgPrimary: string;
  bgSecondary: string;
  textNormal: string;
  textMuted: string;
  border: string;
  accent: string;
  fontFamily: string;
  isDark: boolean;
  positiveBorder: string;
  negativeBorder: string;
  positiveGlyph: string;
  negativeGlyph: string;
  badgeFill: string;
};

/** Read the active Obsidian theme at call time. Never cache across theme changes. */
export function readThemeColors(): ThemeColors {
  const s = getComputedStyle(document.body);
  const v = (name: string, fallback: string) => s.getPropertyValue(name).trim() || fallback;
  const isDark = document.body.classList.contains("theme-dark");
  return {
    bgPrimary: v("--background-primary", isDark ? "#1e1e1e" : "#ffffff"),
    bgSecondary: v("--background-secondary", isDark ? "#262626" : "#f6f6f6"),
    textNormal: v("--text-normal", isDark ? "#dcddde" : "#222222"),
    textMuted: v("--text-muted", isDark ? "#999999" : "#666666"),
    border: v("--background-modifier-border", isDark ? "#3f3f3f" : "#dddddd"),
    accent: v("--interactive-accent", "#7c5cff"),
    fontFamily: s.fontFamily || "sans-serif",
    isDark,
    positiveBorder: isDark ? "rgba(74,222,128,0.9)" : "rgba(21,128,61,0.85)",
    negativeBorder: isDark ? "rgba(248,113,113,0.9)" : "rgba(185,28,28,0.85)",
    positiveGlyph: "#15803d",
    negativeGlyph: "#b91c1c",
    badgeFill: "#ffffff",
  };
}

let alphaCtx: CanvasRenderingContext2D | null = null;

/**
 * CSS vars can resolve to hsl()/named colors that canvas gradients and vis-network
 * accept but we cannot add alpha to directly — normalize through a canvas context.
 */
export function withAlpha(color: string, alpha: number): string {
  if (!alphaCtx) alphaCtx = document.createElement("canvas").getContext("2d");
  if (!alphaCtx) return color;
  alphaCtx.fillStyle = "#000";
  alphaCtx.fillStyle = color;
  const normalized = alphaCtx.fillStyle;
  if (normalized.startsWith("#")) {
    const r = parseInt(normalized.slice(1, 3), 16);
    const g = parseInt(normalized.slice(3, 5), 16);
    const b = parseInt(normalized.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }
  const m = normalized.match(/rgba?\(([^)]+)\)/);
  if (m) {
    const [r, g, b] = m[1].split(",").map((x) => parseFloat(x));
    return `rgba(${r},${g},${b},${alpha})`;
  }
  return color;
}
