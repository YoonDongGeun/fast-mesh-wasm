// The quality report (#257): one mesh's object info, its repair analysis and
// the active inspect map's readouts, as an HTML page or a CSV file. Rows are
// built once (`qualityReportRows`) and both formats print them: the HTML
// with the app's readout formatting, the CSV at full precision, every length
// in the display unit (the rows hold mm). Pure: no scene, no `document`.
import type { AnalysisSummary } from "../../mesh-repair/ux/repair-summary";
import { type LengthUnit, toDisplayArea, toDisplayLength, toDisplayVolume, formatArea, formatLength, formatVolume } from "../../../shared/ux/units";
import { formatDistance } from "../../collision/ux/collision-math";
import { formatCoord } from "./inspect-tools";
import { QUALITY_METRICS } from "./inspect-maps";
import { boxSize, type MeshInfo } from "./object-info";
import type { inspectMapProbe } from "./use-inspect-maps";

/** The shown inspect map, as `inspectMapProbe` publishes it (names, not handles). */
export type ShownInspectMap = NonNullable<(typeof inspectMapProbe)["shown"]>;

/** How a row's value prints: a count, a length (one number or a point), an area, a volume, degrees, a plain ratio or text. */
export type ReportValueKind = "count" | "length" | "area" | "volume" | "degrees" | "ratio" | "text";

export interface ReportRow {
  section: "Object" | "Repair analysis" | "Inspect map";
  quantity: string;
  /** In mm, mm², mm³ or degrees; `null` prints as empty. A point is three lengths. */
  value: number | readonly number[] | string | null;
  kind: ReportValueKind;
}

export interface QualityReportInput {
  /** The mesh's name. */
  name: string;
  info: MeshInfo;
  /** The repair analysis of the mesh, `null` without one. */
  repair: { summary: AnalysisSummary; selfIntersects: boolean | null } | null;
  /** The shown inspect map, `null` without one. */
  map: ShownInspectMap | null;
  /** When the report was made. */
  generated: Date;
}

const yes = (b: boolean) => (b ? "yes" : "no");

/** The report's rows in order: object info, then the repair analysis and the map when there are. */
export function qualityReportRows(input: QualityReportInput): ReportRow[] {
  const { info, repair, map } = input;
  const rows: ReportRow[] = [];
  const obj = (quantity: string, value: ReportRow["value"], kind: ReportValueKind) =>
    rows.push({ section: "Object", quantity, value, kind });
  obj("Triangles", info.triangles, "count");
  obj("Vertices", info.vertices, "count");
  obj("Edges", info.edges, "count");
  obj("Components", info.components, "count");
  obj("Surface area", info.area, "area");
  obj(
    "Volume",
    info.volume ?? (info.volumeReason === "non-manifold" ? "non-manifold (no volume)" : info.volumeReason === "open" ? "open mesh (no volume)" : null),
    info.volume === null ? "text" : "volume",
  );
  obj("Average edge length", info.meanEdge, "length");
  for (const [label, box] of [
    ["World box", info.world],
    ["Local box", info.local],
  ] as const) {
    if (box !== null) {
      obj(`${label} size`, boxSize(box), "length");
      obj(`${label} min`, box.min, "length");
      obj(`${label} max`, box.max, "length");
    }
  }

  if (repair !== null) {
    const s = repair.summary;
    const rep = (quantity: string, value: ReportRow["value"], kind: ReportValueKind) =>
      rows.push({ section: "Repair analysis", quantity, value, kind });
    rep("Clean", yes(s.clean), "text");
    rep("Holes", s.holes.length, "count");
    if (s.holes.length > 0) {
      rep("Largest hole perimeter", Math.max(...s.holes.map((h) => h.perimeter)), "length");
    }
    rep("Open boundary chains", s.openChains, "count");
    rep("Components", s.components.length, "count");
    rep("Open components", s.components.filter((c) => !c.closed).length, "count");
    rep("Non-manifold edges", s.nonManifoldEdges, "count");
    rep("Non-manifold vertices", s.nonManifoldVertices, "count");
    rep("Pinch vertices", s.pinchVertices, "count");
    rep("Degenerate triangles", s.degenerate, "count");
    rep("Duplicate triangles", s.duplicate, "count");
    rep("Orientation conflicts", s.orientationConflicts, "count");
    rep("Self-intersections", repair.selfIntersects === null ? "not checked" : yes(repair.selfIntersects), "text");
  }

  if (map !== null) {
    const m = (quantity: string, value: ReportRow["value"], kind: ReportValueKind) =>
      rows.push({ section: "Inspect map", quantity, value, kind });
    m("Map", map.kind, "text");
    m("Mapped mesh", map.mesh ?? "?", "text");
    const p = map.params;
    if (map.kind === "deviation" && "signed" in p) {
      const r = map.readout as { min: number; max: number; mean: number; rms: number; p95: number; samples: number };
      m("Reference mesh", map.reference ?? "?", "text");
      m("Signed", yes(p.signed), "text");
      m("Minimum deviation", r.min, "length");
      m("Maximum deviation", r.max, "length");
      m("Mean deviation", r.mean, "length");
      m("RMS deviation", r.rms, "length");
      m("95th percentile deviation", r.p95, "length");
      m("Vertices measured", r.samples, "count");
    } else if (map.kind === "overhang" && "criticalAngleDeg" in p) {
      const r = map.readout as { area: number; faces: number; maxAngleDeg: number };
      m("Critical angle", p.criticalAngleDeg, "degrees");
      m("Build direction", `(${p.direction.map(formatCoord).join(", ")})`, "text");
      m("Overhang area", r.area, "area");
      m("Overhang faces", r.faces, "count");
      m("Steepest overhang", r.maxAngleDeg, "degrees");
    } else if (map.kind === "quality" && "metric" in p) {
      const r = map.readout as { past: number; total: number; min: number; max: number; mean: number };
      const kind: ReportValueKind = p.metric === "aspectRatio" ? "ratio" : "degrees";
      m("Metric", QUALITY_METRICS.find((q) => q.metric === p.metric)?.label ?? p.metric, "text");
      m("Threshold", p.threshold, kind);
      m(p.metric === "aspectRatio" ? "Triangles above the threshold" : "Triangles below the threshold", r.past, "count");
      m("Triangles measured", r.total, "count");
      m("Minimum", r.min, kind);
      m("Mean", r.mean, kind);
      m("Maximum", r.max, kind);
    }
  }
  return rows;
}

/** The CSV unit column of a kind. */
function csvUnit(kind: ReportValueKind, unit: LengthUnit): string {
  switch (kind) {
    case "length":
      return unit;
    case "area":
      return `${unit}^2`;
    case "volume":
      return `${unit}^3`;
    case "degrees":
      return "deg";
    default:
      return "";
  }
}

/** A number in the display unit at full precision. */
function csvNumber(x: number, kind: ReportValueKind, unit: LengthUnit): string {
  if (!Number.isFinite(x)) {
    return "";
  }
  switch (kind) {
    case "length":
      return String(toDisplayLength(x, unit));
    case "area":
      return String(toDisplayArea(x, unit));
    case "volume":
      return String(toDisplayVolume(x, unit));
    default:
      return String(x);
  }
}

function escapeCsv(cell: string): string {
  return /[",\n]/.test(cell) ? `"${cell.replace(/"/g, '""')}"` : cell;
}

/**
 * The report as CSV: `section,quantity,value,unit`, a point split into its
 * `x`, `y` and `z` rows, lengths, areas and volumes in `unit` at full
 * precision. The mesh and the unit lead as rows of their own.
 */
export function qualityReportCsv(input: QualityReportInput, unit: LengthUnit): string {
  const lines = ["section,quantity,value,unit"];
  const push = (section: string, quantity: string, value: string, u: string) =>
    lines.push([section, quantity, value, u].map(escapeCsv).join(","));
  push("Report", "Mesh", input.name, "");
  push("Report", "Generated", input.generated.toISOString(), "");
  push("Report", "Length unit", unit, "");
  for (const r of qualityReportRows(input)) {
    const u = csvUnit(r.kind, unit);
    const v = r.value;
    if (typeof v === "number") {
      push(r.section, r.quantity, csvNumber(v, r.kind, unit), u);
    } else if (typeof v === "string") {
      push(r.section, r.quantity, v, "");
    } else if (v === null) {
      push(r.section, r.quantity, "", u);
    } else {
      v.forEach((x, i) => push(r.section, `${r.quantity} ${"xyz"[i]}`, csvNumber(x, r.kind, unit), u));
    }
  }
  return lines.join("\n") + "\n";
}

/** A row's value as the panels print it. */
function displayValue(r: ReportRow, unit: LengthUnit): string {
  const v = r.value;
  if (v === null) {
    return "—";
  }
  if (typeof v === "string") {
    return v;
  }
  if (typeof v !== "number") {
    return `(${v.map((x) => formatCoord(toDisplayLength(x, unit))).join(", ")}) ${unit}`;
  }
  const x = v;
  switch (r.kind) {
    case "count":
      return x.toLocaleString("en");
    case "length":
      return formatLength(x, unit);
    case "area":
      return formatArea(x, unit);
    case "volume":
      return formatVolume(x, unit);
    case "degrees":
      return `${formatDistance(x)}°`;
    default:
      return formatDistance(x);
  }
}

function escapeHtml(text: string): string {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c] as string);
}

/** The report as a standalone HTML page: one table per section, values as the panels print them. */
export function qualityReportHtml(input: QualityReportInput, unit: LengthUnit): string {
  const rows = qualityReportRows(input);
  const sections: ReportRow["section"][] = ["Object", "Repair analysis", "Inspect map"];
  const title = `Quality report: ${input.name}`;
  const body = sections
    .map((section) => {
      const mine = rows.filter((r) => r.section === section);
      if (mine.length === 0) {
        return "";
      }
      const trs = mine
        .map((r) => `      <tr><th scope="row">${escapeHtml(r.quantity)}</th><td>${escapeHtml(displayValue(r, unit))}</td></tr>`)
        .join("\n");
      return `  <h2>${escapeHtml(section)}</h2>\n  <table>\n    <tbody>\n${trs}\n    </tbody>\n  </table>`;
    })
    .filter((s) => s !== "")
    .join("\n");
  return [
    "<!doctype html>",
    '<html lang="en">',
    "<head>",
    '  <meta charset="utf-8">',
    `  <title>${escapeHtml(title)}</title>`,
    "  <style>",
    "    body { font: 14px/1.4 system-ui, sans-serif; margin: 24px; color: #1b1d21; }",
    "    h1 { font-size: 20px; margin: 0 0 4px; }",
    "    h2 { font-size: 15px; margin: 20px 0 6px; }",
    "    table { border-collapse: collapse; min-width: 360px; }",
    "    th, td { text-align: left; padding: 3px 12px 3px 0; border-bottom: 1px solid #e3e5e8; }",
    "    th { font-weight: 500; color: #555b66; }",
    "    td { font-variant-numeric: tabular-nums; }",
    "  </style>",
    "</head>",
    "<body>",
    `  <h1>${escapeHtml(title)}</h1>`,
    `  <p>Length unit: ${unit} · generated ${escapeHtml(input.generated.toISOString())}</p>`,
    body,
    "</body>",
    "</html>",
    "",
  ].join("\n");
}
