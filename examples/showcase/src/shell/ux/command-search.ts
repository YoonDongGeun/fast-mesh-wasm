// The command palette's search (#235): a fuzzy subsequence / substring
// score per command, over its title, category and keywords, and the best
// matches for a query.
import type { CommandPaletteItem } from "./command-registry";

/**
 * Scores a fuzzy subsequence / substring query against `text`.
 * Returns `null` when `query` does not match, otherwise higher scores are better.
 */
export function fuzzyMatchCommand(query: string, text: string): number | null {
  const q = query.trim().toLowerCase();
  if (q === "") return 1;
  const hay = text.toLowerCase();
  const idx = hay.indexOf(q);
  if (idx >= 0) {
    const wordStart = idx === 0 || /[\s\-/_:&]/.test(hay[idx - 1] ?? "");
    return 1000 - idx * 2 + (wordStart ? 200 : 0);
  }
  let qi = 0;
  let gap = 0;
  let lastMatch = -1;
  for (let hi = 0; hi < hay.length && qi < q.length; hi++) {
    if (hay[hi] === q[qi]) {
      if (lastMatch >= 0) gap += hi - lastMatch - 1;
      lastMatch = hi;
      qi++;
    }
  }
  if (qi < q.length) return null;
  return Math.max(1, 300 - gap * 3 - (lastMatch - q.length));
}

/** The best 24 commands for `query`, highest score first (ties keep registry order). */
export function filterCommands(commands: readonly CommandPaletteItem[], query: string): CommandPaletteItem[] {
  const scored: { item: CommandPaletteItem; score: number }[] = [];
  for (const item of commands) {
    const s = fuzzyMatchCommand(query, `${item.title} ${item.category} ${item.keywords}`);
    if (s !== null) {
      scored.push({ item, score: s });
    }
  }
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, 24).map((e) => e.item);
}
