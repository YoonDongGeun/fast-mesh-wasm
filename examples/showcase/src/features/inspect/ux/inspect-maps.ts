// Inspection colour maps on one mesh (#251): deviation from a reference
// mesh, overhanging faces against a build direction, and triangle quality.
// Each runs one SDK inspect query (`meshDeviation`, `overhangs`,
// `triangleQuality`), on a local `Scene` or a `WorkerScene` alike, and keeps
// the query's own numbers as the readouts. Its per-vertex values go to the
// surface field overlay with a colormap range and an optional band of values
// drawn in one colour: the tolerance band, faces that need no support, or
// triangles past the quality threshold. No React and no three, so vitest
// covers it; `use-inspect-maps.ts` wires it to the panel.
import type { MeshChange, MeshHandle, SceneLike } from "fast-mesh-wasm";
import type { OverlayBand, SurfaceColormap, SurfaceFieldOverlayData } from "../../../app/state/app-state";
import type { Vec3 } from "../../../io/mesh-data";
import { formatDistance } from "../../collision/ux/collision-math";
import { formatArea, formatLength, type LengthUnit } from "../../../shared/ux/units";
import { formatCoord } from "./inspect-tools";

export type InspectMapKind = "deviation" | "overhang" | "quality";

/** The quality metrics the panel offers (the SDK's radius ratio is left out). */
export type QualityMapMetric = "aspectRatio" | "minAngleDeg";

/** Where the overhang build direction comes from: world +Z, or the camera's up when the map is made. */
export type BuildDirectionSource = "worldZ" | "viewUp";

export const INSPECT_MAP_KINDS: readonly { kind: InspectMapKind; label: string; title: string }[] = [
  { kind: "deviation", label: "Deviation", title: "Distance from each vertex to a reference mesh" },
  { kind: "overhang", label: "Overhang", title: "Faces that tilt down past the critical angle from the build direction" },
  { kind: "quality", label: "Quality", title: "Triangle shape: aspect ratio or minimum angle" },
];

export const QUALITY_METRICS: readonly {
  metric: QualityMapMetric;
  label: string;
  /** The threshold a new pick of this metric starts at (the SDK's default). */
  threshold: number;
  title: string;
}[] = [
  {
    metric: "aspectRatio",
    label: "Aspect ratio",
    threshold: 10,
    title: "Longest edge against the inradius, 1 for an equilateral triangle; above the threshold counts",
  },
  {
    metric: "minAngleDeg",
    label: "Minimum angle",
    threshold: 10,
    title: "Smallest interior angle in degrees, 60 for an equilateral triangle; below the threshold counts",
  },
];

/** The map controls' values. Lengths are in mm, angles in degrees. */
export interface InspectMapSettings {
  kind: InspectMapKind;
  /** The deviation's reference mesh (`null`, or one no longer loaded: the first other mesh). */
  reference: MeshHandle | null;
  /** Signed deviation (negative inside the reference) or unsigned distance. */
  signed: boolean;
  /** Whether deviations within `tolerance` show in one colour. */
  toleranceOn: boolean;
  /** The tolerance band's half-width, mm. */
  tolerance: number;
  buildFrom: BuildDirectionSource;
  /** Faces tilted down past this angle from horizontal need support. */
  criticalAngleDeg: number;
  metric: QualityMapMetric;
  /** A triangle counts past it: above for aspect ratio, below for the minimum angle. */
  threshold: number;
}

export const DEFAULT_INSPECT_MAP_SETTINGS: InspectMapSettings = {
  kind: "deviation",
  reference: null,
  signed: false,
  toleranceOn: false,
  tolerance: 0.1,
  buildFrom: "worldZ",
  criticalAngleDeg: 45,
  metric: "aspectRatio",
  threshold: 10,
};

/** One query, resolved: every input it ran with, so a refresh runs it again unchanged. */
export type InspectMapRequest =
  | { kind: "deviation"; mesh: MeshHandle; reference: MeshHandle; signed: boolean }
  | { kind: "overhang"; mesh: MeshHandle; direction: Vec3; criticalAngleDeg: number }
  | { kind: "quality"; mesh: MeshHandle; metric: QualityMapMetric; threshold: number };

/** The deviation statistics, in mm (signed ones keep their sign). */
export interface DeviationReadout {
  min: number;
  max: number;
  mean: number;
  rms: number;
  p95: number;
  /** Vertices measured. */
  samples: number;
}

/** The faces past the critical angle: their area (mm²) and count, and the steepest overhang. */
export interface OverhangReadout {
  area: number;
  faces: number;
  maxAngleDeg: number;
}

/** Triangles past the threshold out of all live ones, and the metric's spread. */
export interface QualityReadout {
  past: number;
  total: number;
  min: number;
  max: number;
  mean: number;
}

export type InspectMapReadout = DeviationReadout | OverhangReadout | QualityReadout;

/** A query's result: its request, its readouts, and one value per vertex slot. */
export type InspectMapReport =
  | { kind: "deviation"; request: Extract<InspectMapRequest, { kind: "deviation" }>; readout: DeviationReadout; values: Float32Array }
  | { kind: "overhang"; request: Extract<InspectMapRequest, { kind: "overhang" }>; readout: OverhangReadout; values: Float32Array }
  | { kind: "quality"; request: Extract<InspectMapRequest, { kind: "quality" }>; readout: QualityReadout; values: Float32Array };

/** How a report's values are coloured. */
export interface InspectMapStyle {
  map: SurfaceColormap;
  min: number;
  max: number;
  band: OverlayBand | null;
}

/** Deviations within the tolerance. */
export const TOLERANCE_COLOR: readonly [number, number, number] = [0.22, 0.72, 0.32];
/** Faces at or under the critical angle: no support needed. */
export const NO_SUPPORT_COLOR: readonly [number, number, number] = [0.62, 0.62, 0.62];
/** Triangles past the quality threshold. */
export const PAST_THRESHOLD_COLOR: readonly [number, number, number] = [0.9, 0.12, 0.1];

/** World +Z, the default build direction. */
export const WORLD_UP: Vec3 = [0, 0, 1];

/**
 * The build direction: world +Z, or the camera's up in world space (the
 * second column of its world matrix, column-major), normalised. Without a
 * camera, or with a degenerate one, it is world +Z.
 */
export function resolveBuildDirection(source: BuildDirectionSource, cameraWorld: ArrayLike<number> | null): Vec3 {
  if (source === "worldZ" || cameraWorld === null) {
    return [...WORLD_UP];
  }
  const x = cameraWorld[4];
  const y = cameraWorld[5];
  const z = cameraWorld[6];
  const len = Math.hypot(x, y, z);
  if (!(len > 0) || !Number.isFinite(len)) {
    return [...WORLD_UP];
  }
  return [x / len, y / len, z / len];
}

/**
 * The deviation reference: the picked mesh while it is loaded and is not the
 * mapped mesh, else the first other mesh, else `null`.
 */
export function resolveReference(
  picked: MeshHandle | null,
  mesh: MeshHandle | null,
  meshes: readonly MeshHandle[],
): MeshHandle | null {
  if (picked !== null && picked !== mesh && meshes.includes(picked)) {
    return picked;
  }
  return meshes.find((m) => m !== mesh) ?? null;
}

/**
 * The query for `settings` on `mesh`, or `null` when a deviation has no
 * reference. `direction` is the resolved build direction (overhangs only).
 */
export function buildInspectMapRequest(
  settings: InspectMapSettings,
  mesh: MeshHandle,
  reference: MeshHandle | null,
  direction: Vec3,
): InspectMapRequest | null {
  switch (settings.kind) {
    case "deviation":
      return reference === null || reference === mesh
        ? null
        : { kind: "deviation", mesh, reference, signed: settings.signed };
    case "overhang":
      return {
        kind: "overhang",
        mesh,
        direction: [...direction],
        criticalAngleDeg: Math.min(90, Math.max(0, settings.criticalAngleDeg)),
      };
    case "quality":
      return { kind: "quality", mesh, metric: settings.metric, threshold: Math.max(0, settings.threshold) };
  }
}

/** Runs the request's SDK query. A local scene answers at once and a Worker one with a promise: this takes either. */
export async function runInspectMap(scene: SceneLike, request: InspectMapRequest): Promise<InspectMapReport> {
  const inspect = scene.inspect;
  if (inspect === undefined) {
    throw new Error("This scene has no inspect queries");
  }
  switch (request.kind) {
    case "deviation": {
      const s = await inspect.meshDeviation(request.mesh, request.reference, { signed: request.signed });
      return {
        kind: "deviation",
        request,
        readout: { min: s.min, max: s.max, mean: s.mean, rms: s.rms, p95: s.p95, samples: s.sampleCount },
        values: s.vertexDeviations,
      };
    }
    case "overhang": {
      const r = await inspect.overhangs(request.mesh, {
        buildDirection: request.direction,
        criticalAngleDeg: request.criticalAngleDeg,
      });
      return {
        kind: "overhang",
        request,
        readout: { area: r.totalSupportArea, faces: r.overhangFaceCount, maxAngleDeg: r.maxOverhangAngleDeg },
        values: r.vertexAnglesDeg,
      };
    }
    case "quality": {
      const q = await inspect.triangleQuality(request.mesh, {
        metric: request.metric,
        sliverThreshold: request.threshold,
      });
      return {
        kind: "quality",
        request,
        readout: { past: q.sliverTriangleCount, total: q.liveTriangleCount, min: q.min, max: q.max, mean: q.mean },
        values: q.vertexQuality,
      };
    }
  }
}

/** `[lo, hi]` with `hi > lo`: a flat range gets a unit span, as the surface field's does. */
function span(lo: number, hi: number): [number, number] {
  return hi > lo ? [lo, hi] : [lo, lo + 1];
}

/**
 * The report's colouring. Deviation: coolwarm over `[0, max]` (unsigned) or
 * symmetric about zero (signed), the tolerance band in green. Overhang: turbo
 * from the critical angle to 90°, the faces at or under it in grey. Quality:
 * viridis from the best value to the threshold, triangles past it in red.
 * Only the tolerance comes from `settings`; the rest is the report's request.
 */
export function styleInspectMap(report: InspectMapReport, settings: InspectMapSettings): InspectMapStyle {
  switch (report.kind) {
    case "deviation": {
      const { min, max } = report.readout;
      const reach = Math.max(Math.abs(min), Math.abs(max));
      const [lo, hi] = report.request.signed ? (reach > 0 ? [-reach, reach] : [-1, 1]) : span(0, max);
      const tol = settings.toleranceOn && settings.tolerance > 0 ? settings.tolerance : null;
      return {
        map: "coolwarm",
        min: lo,
        max: hi,
        band: tol === null ? null : { lo: -tol, hi: tol, color: TOLERANCE_COLOR },
      };
    }
    case "overhang": {
      const crit = report.request.criticalAngleDeg;
      const [lo, hi] = span(crit, 90);
      return { map: "turbo", min: lo, max: hi, band: { lo: -Infinity, hi: crit, color: NO_SUPPORT_COLOR } };
    }
    case "quality": {
      const t = report.request.threshold;
      if (report.request.metric === "aspectRatio") {
        const [lo, hi] = span(1, t);
        return { map: "viridis", min: lo, max: hi, band: { lo: t, hi: Infinity, color: PAST_THRESHOLD_COLOR } };
      }
      const [lo, hi] = span(t, 60);
      return { map: "viridis", min: lo, max: hi, band: { lo: -Infinity, hi: t, color: PAST_THRESHOLD_COLOR } };
    }
  }
}

/** The overlay data that draws the report on its mesh. */
export function inspectMapOverlay(report: InspectMapReport, style: InspectMapStyle): SurfaceFieldOverlayData {
  return {
    mesh: report.request.mesh,
    values: report.values,
    map: style.map,
    min: style.min,
    max: style.max,
    band: style.band,
  };
}

/** A number without a unit, in `formatDistance`'s digits. */
function plain(x: number): string {
  return formatDistance(x);
}

/** An angle in degrees. */
function degrees(x: number): string {
  return `${formatDistance(x)}°`;
}

/** A band's swatch colour for CSS. */
export function cssColor(c: readonly [number, number, number]): string {
  return `rgb(${c.map((v) => Math.round(Math.min(1, Math.max(0, v)) * 255)).join(", ")})`;
}

/** The legend: the colormap's two ends, and the band's swatch and meaning. */
export interface InspectMapLegend {
  map: SurfaceColormap;
  lo: string;
  hi: string;
  band: { color: string; label: string } | null;
}

/** The legend of a styled report, lengths in `unit`. */
export function inspectMapLegend(report: InspectMapReport, style: InspectMapStyle, unit: LengthUnit): InspectMapLegend {
  const band = (label: string) => (style.band === null ? null : { color: cssColor(style.band.color), label });
  switch (report.kind) {
    case "deviation": {
      const tol = style.band === null ? 0 : style.band.hi;
      return {
        map: style.map,
        lo: formatLength(style.min, unit),
        hi: formatLength(style.max, unit),
        band: band(report.request.signed ? `within ±${formatLength(tol, unit)}` : `within ${formatLength(tol, unit)}`),
      };
    }
    case "overhang": {
      const crit = report.request.criticalAngleDeg;
      return { map: style.map, lo: degrees(style.min), hi: degrees(style.max), band: band(`up to ${degrees(crit)}: no support`) };
    }
    case "quality": {
      const t = report.request.threshold;
      if (report.request.metric === "aspectRatio") {
        return { map: style.map, lo: plain(style.min), hi: plain(style.max), band: band(`above ${plain(t)}`) };
      }
      return { map: style.map, lo: degrees(style.min), hi: degrees(style.max), band: band(`below ${degrees(t)}`) };
    }
  }
}

/** A direction's components, for the readout. */
function formatDirection(d: readonly number[]): string {
  return `(${d.map((x) => formatCoord(x)).join(", ")})`;
}

/** The readout lines of a report, lengths in `unit`, the reference by `referenceName`. */
export function inspectMapReadoutLines(
  report: InspectMapReport,
  unit: LengthUnit,
  referenceName: string | null = null,
): string[] {
  switch (report.kind) {
    case "deviation": {
      const r = report.readout;
      const L = (x: number) => formatLength(x, unit);
      return [
        `max ${L(r.max)} · mean ${L(r.mean)} · RMS ${L(r.rms)}`,
        `min ${L(r.min)} · p95 ${L(r.p95)} · ${r.samples} vertices`,
        `${report.request.signed ? "signed" : "unsigned"}, to ${referenceName ?? "the reference"}`,
      ];
    }
    case "overhang": {
      const r = report.readout;
      return [
        `overhang area ${formatArea(r.area, unit)} · ${r.faces} ${r.faces === 1 ? "face" : "faces"}`,
        `steepest ${degrees(r.maxAngleDeg)} · critical ${degrees(report.request.criticalAngleDeg)} · build ${formatDirection(report.request.direction)}`,
      ];
    }
    case "quality": {
      const r = report.readout;
      const fmt = report.request.metric === "aspectRatio" ? plain : degrees;
      const side = report.request.metric === "aspectRatio" ? "above" : "below";
      return [
        `${r.past} of ${r.total} triangles ${side} ${fmt(report.request.threshold)}`,
        `min ${fmt(r.min)} · mean ${fmt(r.mean)} · max ${fmt(r.max)}`,
      ];
    }
  }
}

/** Whether a change record's buffer range list says something changed. */
function touched(r: MeshChange["positions"]): boolean {
  return r === "all" || r.length > 0;
}

/**
 * What a batch of scene changes means for a shown map: `"clear"` when its
 * mesh or its reference was removed, `"refresh"` when either one's geometry
 * or placement changed (an edit, an undo or redo, a move), else `null`.
 * Colour-only changes leave it alone.
 */
export function inspectMapChangeAction(
  changes: readonly MeshChange[],
  request: InspectMapRequest,
): "clear" | "refresh" | null {
  const watched = request.kind === "deviation" ? [request.mesh, request.reference] : [request.mesh];
  let refresh = false;
  for (const c of changes) {
    if (!watched.includes(c.mesh)) {
      continue;
    }
    if (c.removed) {
      return "clear";
    }
    if (c.placementChanged || c.capacityChanged || touched(c.positions) || touched(c.indices)) {
      refresh = true;
    }
  }
  return refresh ? "refresh" : null;
}
