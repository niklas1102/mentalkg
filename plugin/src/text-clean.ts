/**
 * Strip markdown noise before sending note text to the extraction server:
 * code blocks out, headings/checkboxes/links reduced to their plain text,
 * tags and embeds removed.
 */
export function stripMarkdownNoise(body: string): string {
  let t = body;

  // Fenced code blocks (``` … ```) and inline code.
  t = t.replace(/```[\s\S]*?(```|$)/g, " ");
  t = t.replace(/`([^`\n]*)`/g, "$1");

  // Image/file embeds ![[...]] and markdown images ![alt](url).
  t = t.replace(/!\[\[[^\]]*\]\]/g, " ");
  t = t.replace(/!\[[^\]]*\]\([^)]*\)/g, " ");

  // Wikilinks: [[Target|Alias]] -> Alias, [[Target]] -> Target.
  t = t.replace(/\[\[([^\]|]*)\|([^\]]*)\]\]/g, "$2");
  t = t.replace(/\[\[([^\]]*)\]\]/g, "$1");

  // Markdown links: [text](url) -> text.
  t = t.replace(/\[([^\]]*)\]\([^)]*\)/g, "$1");

  // ATX headings: drop the # markers, keep the heading text.
  t = t.replace(/^#{1,6}\s+/gm, "");

  // Checkboxes: "- [ ] task" / "* [x] task" -> "task".
  t = t.replace(/^(\s*)[-*+]\s+\[[ xX]\]\s+/gm, "$1");

  // Remaining list bullets and blockquote markers.
  t = t.replace(/^(\s*)[-*+]\s+/gm, "$1");
  t = t.replace(/^\s*>\s?/gm, "");

  // Inline #tags (but not "# heading" already handled, nor mid-word #).
  t = t.replace(/(^|\s)#[\w/-]+/g, "$1");

  // Bold/italic markers.
  t = t.replace(/(\*\*|__)(.*?)\1/g, "$2");
  t = t.replace(/(\*|_)(.*?)\1/g, "$2");

  // Horizontal rules and setext underlines.
  t = t.replace(/^\s*(---+|===+|\*\*\*+)\s*$/gm, "");

  // Collapse whitespace.
  t = t.replace(/[ \t]+/g, " ");
  t = t.replace(/\n{3,}/g, "\n\n");

  return t.trim();
}
