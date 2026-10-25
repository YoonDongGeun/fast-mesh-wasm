import { readFileSync, writeFileSync } from "node:fs";

export const NOISE_PCT = 5;

const pct = (base, head) => (base === 0 ? (head === 0 ? 0 : 100) : ((head - base) / base) * 100);

const statusOf = (deltaPct) => {
  if (deltaPct <= -NOISE_PCT) return "🟢";
  if (deltaPct >= NOISE_PCT) return "🔴";
  return "➖";
};

export function compare(baseline, current) {
  const baseByName = new Map((baseline.cases ?? []).map((c) => [c.name, c]));
  const headByName = new Map((current.cases ?? []).map((c) => [c.name, c]));
  const rows = [];
  const added = [];
  const removed = [];

  for (const [name, b] of baseByName) {
    const h = headByName.get(name);
    if (h === undefined || h.unit !== b.unit) {
      removed.push(name);
    }
  }
  for (const [name, h] of headByName) {
    const b = baseByName.get(name);
    if (b === undefined || b.unit !== h.unit) {
      added.push(name);
    }
  }

  for (const [name, b] of baseByName) {
    const h = headByName.get(name);
    if (h === undefined || h.unit !== b.unit) continue;
    const deltaPct = pct(b.median, h.median);
    const p95DeltaPct = pct(b.p95, h.p95);
    const row = {
      name,
      unit: b.unit,
      base: b.median,
      head: h.median,
      deltaPct,
      p95DeltaPct,
      status: statusOf(deltaPct),
    };
    if (b.heapMB !== undefined && h.heapMB !== undefined) {
      row.heapDeltaMB = h.heapMB - b.heapMB;
    }
    rows.push(row);
  }

  return { rows, added, removed };
}

const fmt = (v) => (typeof v === "number" ? v.toFixed(2) : String(v));
const fmtDelta = (v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)}%`;

export function renderMarkdown(result, { baseRef, headRef }) {
  const lines = [`# Bench: ${baseRef} → ${headRef}`, ""];
  lines.push("| case | unit | base | head | Δ median | Δ p95 | heap Δ | |");
  lines.push("| --- | --- | --- | --- | --- | --- | --- | --- |");
  for (const r of result.rows) {
    const heap = r.heapDeltaMB === undefined ? "n/a" : `${r.heapDeltaMB >= 0 ? "+" : ""}${fmt(r.heapDeltaMB)} MB`;
    lines.push(
      `| ${r.name} | ${r.unit} | ${fmt(r.base)} | ${fmt(r.head)} | ${fmtDelta(r.deltaPct)} | ${fmtDelta(r.p95DeltaPct)} | ${heap} | ${r.status} |`,
    );
  }
  if (result.rows.length === 0) {
    lines.push("| (no common cases) | | | | | | | |");
  }
  lines.push("");
  lines.push(result.added.length > 0 ? `Added: ${result.added.join(", ")}` : "Added: none");
  lines.push(result.removed.length > 0 ? `Removed: ${result.removed.join(", ")}` : "Removed: none");
  lines.push("");
  return lines.join("\n");
}

if (import.meta.filename === process.argv[1]) {
  const args = process.argv.slice(2);
  const outIdx = args.indexOf("--out");
  let outPath = null;
  if (outIdx !== -1) {
    outPath = args[outIdx + 1];
    args.splice(outIdx, 2);
  }
  const [basePath, headPath] = args;
  if (!basePath || !headPath) {
    console.error("usage: node scripts/bench-compare.mjs <base.json> <head.json> [--out body.md]");
    process.exit(1);
  }
  const baseDoc = JSON.parse(readFileSync(basePath, "utf8"));
  const headDoc = JSON.parse(readFileSync(headPath, "utf8"));
  const result = compare(baseDoc, headDoc);
  const md = renderMarkdown(result, { baseRef: basePath, headRef: headPath });
  if (outPath) {
    writeFileSync(outPath, md);
  } else {
    process.stdout.write(md);
  }
}
