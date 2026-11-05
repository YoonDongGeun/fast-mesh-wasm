// Showcase export reports (issue #127): section contours to DXF/SVG and the
// collision summary to CSV.
//
// The builders below are pure (no scene, no `document`) and unit-tested:
// they take contours already read from `scene.section(handle, mesh)` and
// readout rows already named through the app's items, and return file text.
// The `download*` helpers wrap the File System Access picker fallback pattern
// from `load/project.ts` (anchor download); they run only from click handlers.
import type { SectionContour } from "fast-mesh-wasm";
import { type LengthUnit, toDisplayLength } from "../panels/units";

/** A contour's shape, without its scene origin: what the builders read. */
export interface ContourLike {
  points: ArrayLike<number>;
  closed: boolean;
}

/** One collision pair as a named CSV row: `gap`/`depth` are `null` before measured. */
export interface CollisionCsvRow {
  a: string;
  b: string;
  gap: number | null;
  intersects: boolean;
  depth: number | null;
}

type Vec3 = readonly [number, number, number];

/** An orthonormal in-plane basis for `normal`: `u` from the smallest axis, `v = normal × u`. */
export function planeBasis(normal: Vec3): { u: Vec3; v: Vec3 } {
  const n = normalize(normal);
  const ax = Math.abs(n[0]) <= Math.abs(n[1]) && Math.abs(n[0]) <= Math.abs(n[2]) ? 0 : Math.abs(n[1]) <= Math.abs(n[2]) ? 1 : 2;
  const hint: Vec3 = ax === 0 ? [1, 0, 0] : ax === 1 ? [0, 1, 0] : [0, 0, 1];
  const u = normalize(sub(hint, scale(n, dot(hint, n))));
  return { u, v: cross(n, u) };
}

/** Projects world contour points onto the plane: 2D coordinates about `origin`. */
export function projectContours(
  contours: readonly ContourLike[],
  origin: Vec3,
  normal: Vec3,
): { closed: boolean; points: [x: number, y: number][] }[] {
  const { u, v } = planeBasis(normal);
  return contours.map((c) => {
    const points: [number, number][] = [];
    for (let i = 0; i + 2 < c.points.length; i += 3) {
      const p: Vec3 = [c.points[i], c.points[i + 1], c.points[i + 2]];
      const d = sub(p, origin);
      points.push([dot(d, u), dot(d, v)]);
    }
    return { closed: c.closed, points };
  });
}

/** Short decimals: full float precision bloats the files without meaning. */
function num(x: number): string {
  if (!Number.isFinite(x)) {
    return "0";
  }
  if (x === 0) {
    return "0";
  }
  return String(Number(x.toPrecision(6)));
}

/**
 * Section contours as SVG: closed contours are polygons, open ones polylines,
 * in plane coordinates about the section origin. Always a valid document,
 * even with no contours.
 */
export function contoursToSvg(contours: readonly ContourLike[], origin: Vec3, normal: Vec3): string {
  const flat = projectContours(contours, origin, normal);
  let minX = 0;
  let minY = 0;
  let maxX = 1;
  let maxY = 1;
  if (flat.some((c) => c.points.length > 0)) {
    minX = minY = Infinity;
    maxX = maxY = -Infinity;
    for (const c of flat) {
      for (const [x, y] of c.points) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
    if (maxX - minX < 1e-9) {
      maxX = minX + 1;
    }
    if (maxY - minY < 1e-9) {
      maxY = minY + 1;
    }
  }
  const pad = 0.05;
  const vx = minX - (maxX - minX) * pad;
  const vw = (maxX - minX) * (1 + 2 * pad);
  const vh = (maxY - minY) * (1 + 2 * pad);
  const lines = flat
    .filter((c) => c.points.length > 0)
    .map((c) => {
      const pts = c.points.map(([x, y]) => `${num(x)},${num(-y)}`).join(" ");
      const tag = c.closed ? "polygon" : "polyline";
      return `  <${tag} points="${pts}" fill="none" stroke="black" stroke-width="0.01"/>`;
    });
  // SVG y grows downwards: negate y so counter-clockwise (about the normal) stays counter-clockwise.
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${num(vx)} ${num(-maxY - (maxY - minY) * pad)} ${num(vw)} ${num(vh)}">`,
    ...lines,
    `</svg>`,
  ].join("\n");
}

/**
 * Section contours as minimal DXF (R12-compatible ENTITIES of LWPOLYLINEs):
 * closed contours carry the closed flag. Always a valid file, even empty.
 */
export function contoursToDxf(contours: readonly ContourLike[], origin: Vec3, normal: Vec3): string {
  const flat = projectContours(contours, origin, normal);
  const out = ["0", "SECTION", "2", "ENTITIES"];
  for (const c of flat) {
    if (c.points.length === 0) {
      continue;
    }
    out.push("0", "LWPOLYLINE", "8", "0", "90", String(c.points.length), "70", c.closed ? "1" : "0");
    for (const [x, y] of c.points) {
      out.push("10", num(x), "20", num(y));
    }
  }
  out.push("0", "ENDSEC", "0", "EOF");
  return out.join("\n") + "\n";
}

/** Reads one mesh's section result into plain contours (copies the points). */
export function toContourLikes(result: { contours: readonly SectionContour[] }): ContourLike[] {
  return result.contours.map((c) => ({ points: Float32Array.from(c.points), closed: c.closed }));
}

function escapeCsv(cell: string): string {
  return /[",\n]/.test(cell) ? `"${cell.replace(/"/g, '""')}"` : cell;
}

/**
 * The collision summary as CSV: one row per pair (`mesh_a`, `mesh_b`,
 * `min_gap_<unit>`, `intersects`, `max_depth_<unit>`), unmeasured gaps/depths
 * empty, lengths in `unit` at full precision (the rows hold mm). Header only
 * when there are no rows.
 */
export function collisionRowsToCsv(rows: readonly CollisionCsvRow[], unit: LengthUnit = "mm"): string {
  const lines = [`mesh_a,mesh_b,min_gap_${unit},intersects,max_depth_${unit}`];
  for (const r of rows) {
    lines.push(
      [
        escapeCsv(r.a),
        escapeCsv(r.b),
        r.gap === null ? "" : String(toDisplayLength(r.gap, unit)),
        r.intersects ? "yes" : "no",
        r.depth === null ? "" : String(toDisplayLength(r.depth, unit)),
      ].join(","),
    );
  }
  return lines.join("\n") + "\n";
}

/** A download file name: the stem sanitized, with the extension. */
export function reportFileName(stem: string, extension: "dxf" | "svg" | "csv"): string {
  const clean = stem.replace(/[\\/:*?"<>|]/g, "_").trim() || "export";
  return `${clean}.${extension}`;
}

/** Triggers a download of raw bytes (the fallback when no save picker exists). */
export function downloadBytes(bytes: Uint8Array, fileName: string, mime: string): void {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  const url = URL.createObjectURL(new Blob([copy.buffer as ArrayBuffer], { type: mime }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/** Triggers a download of text (DXF, SVG, CSV). */
export function downloadReportText(text: string, fileName: string, mime: string): void {
  downloadBytes(new TextEncoder().encode(text), fileName, mime);
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function scale(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function normalize(a: Vec3): Vec3 {
  const n = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
}
