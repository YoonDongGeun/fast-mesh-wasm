// The geodesic distance map (#252): the surface distance from the picked
// source vertices to every vertex (`inspect.geodesicDistances`, on a local
// `Scene` or a `WorkerScene` alike), drawn through the surface field overlay
// from zero at the sources to the farthest vertex reached, with optional
// iso-lines at a fixed spacing. Vertices no source reaches (another part of
// the mesh) have no finite distance and show grey. No React and no three, so
// vitest covers it; `use-geodesic-map.ts` wires it to the panel.
import type { GeodesicMethod, MeshHandle, SceneLike } from "fast-mesh-wasm";
import type { SurfaceColormap, SurfaceFieldOverlayData } from "../../../app/state/app-state";
import { formatLength, type LengthUnit } from "../../../shared/ux/units";
import { cssColor } from "./inspect-maps";
import { methodLabel } from "./surface-paths";

/** One map query: the mesh, its source vertex slots and the method. */
export interface GeodesicMapRequest {
  mesh: MeshHandle;
  sources: readonly number[];
  method: GeodesicMethod;
}

/** A map query's result: its request, one distance per vertex slot, and the readouts. */
export interface GeodesicMapReport {
  request: GeodesicMapRequest;
  /** World distance per vertex slot; `Infinity` where no source reaches (and on unused slots). */
  values: Float32Array;
  /** The farthest distance reached. */
  max: number;
  /** The mean distance over the vertices reached. */
  mean: number;
  /** Vertices of live triangles with a distance. */
  reached: number;
  /** Vertices of live triangles no source reaches. */
  unreached: number;
}

/** The map's colormap: turbo, from blue at the sources to red at the farthest vertex. */
export const GEODESIC_COLORMAP: SurfaceColormap = "turbo";

/** The overlay's colour for a vertex without a finite distance (see `evaluateSurfaceColormap`). */
export const UNREACHED_COLOR: readonly [number, number, number] = [0.5, 0.5, 0.5];

/** At most this many iso-lines: a smaller spacing is widened to fit. */
export const MAX_ISO_LEVELS = 200;

/** Whether two requests ask for the same map. */
export function sameGeodesicRequest(a: GeodesicMapRequest | null, b: GeodesicMapRequest | null): boolean {
  if (a === null || b === null) {
    return a === b;
  }
  return (
    a.mesh === b.mesh &&
    a.method === b.method &&
    a.sources.length === b.sources.length &&
    a.sources.every((s, i) => s === b.sources[i])
  );
}

/** Runs the request's SDK query (a local scene answers at once, a Worker one with a promise) and reads it out. */
export async function runGeodesicMap(
  scene: Pick<SceneLike, "inspect" | "views">,
  request: GeodesicMapRequest,
): Promise<GeodesicMapReport> {
  const inspect = scene.inspect;
  if (inspect === undefined) {
    throw new Error("This scene has no geodesic distances");
  }
  if (request.sources.length === 0) {
    throw new Error("Pick at least one source vertex.");
  }
  const values = await inspect.geodesicDistances(request.mesh, [...request.sources], { method: request.method });
  return { request, values, ...summarizeDistances(values, scene.views(request.mesh).indices()) };
}

/**
 * The distances' readouts over the vertices of live triangles: the
 * farthest and the mean finite distance, and how many are reached or not.
 */
export function summarizeDistances(
  values: Float32Array,
  indices: ArrayLike<number>,
): { max: number; mean: number; reached: number; unreached: number } {
  const used = new Uint8Array(values.length);
  for (let k = 0; k + 2 < indices.length; k += 3) {
    const a = indices[k];
    const b = indices[k + 1];
    const c = indices[k + 2];
    if (a === 0 && b === 0 && c === 0) {
      continue; // a dead slot
    }
    used[a] = 1;
    used[b] = 1;
    used[c] = 1;
  }
  let max = 0;
  let sum = 0;
  let reached = 0;
  let unreached = 0;
  for (let i = 0; i < values.length; i++) {
    if (used[i] === 0) {
      continue;
    }
    const v = values[i];
    if (Number.isFinite(v)) {
      reached++;
      sum += v;
      if (v > max) {
        max = v;
      }
    } else {
      unreached++;
    }
  }
  return { max, mean: reached > 0 ? sum / reached : 0, reached, unreached };
}

/** The overlay data that draws the report on its mesh: zero to the farthest distance (a unit span when flat). */
export function geodesicOverlay(report: GeodesicMapReport): SurfaceFieldOverlayData {
  return {
    mesh: report.request.mesh,
    values: report.values,
    map: GEODESIC_COLORMAP,
    min: 0,
    max: report.max > 0 ? report.max : 1,
    band: null,
  };
}

/** The legend: zero and the farthest distance in `unit`, and grey for vertices not reached when there are any. */
export function geodesicLegend(
  report: GeodesicMapReport,
  unit: LengthUnit,
): { map: SurfaceColormap; lo: string; hi: string; band: { color: string; label: string } | null } {
  return {
    map: GEODESIC_COLORMAP,
    lo: formatLength(0, unit),
    hi: formatLength(report.max, unit),
    band: report.unreached > 0 ? { color: cssColor(UNREACHED_COLOR), label: "not reached" } : null,
  };
}

/** The readout lines of a report, lengths in `unit`. */
export function geodesicReadoutLines(report: GeodesicMapReport, unit: LengthUnit): string[] {
  const n = report.request.sources.length;
  const lines = [
    `${n} ${n === 1 ? "source" : "sources"} · farthest ${formatLength(report.max, unit)} · mean ${formatLength(report.mean, unit)}`,
    `${methodLabel(report.request.method)} · ${report.reached} ${report.reached === 1 ? "vertex" : "vertices"} reached`,
  ];
  if (report.unreached > 0) {
    lines.push(`${report.unreached} on parts no source reaches (grey)`);
  }
  return lines;
}

/** A round iso-line spacing for distances up to `max`: about a tenth of it, 1, 2 or 5 times a power of ten. */
export function autoIsoSpacing(max: number): number {
  if (!(max > 0) || !Number.isFinite(max)) {
    return 1;
  }
  const raw = max / 10;
  const p = 10 ** Math.floor(Math.log10(raw));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

/** The spacing iso-lines are drawn at: `spacing`, widened so at most {@link MAX_ISO_LEVELS} lines fit under `max`. */
export function isoLevelSpacing(spacing: number, max: number): number {
  const floor = max > 0 && Number.isFinite(max) ? max / MAX_ISO_LEVELS : 0;
  return spacing > floor ? spacing : floor > 0 ? floor : 1;
}

/**
 * The iso-lines of per-vertex `values` at every multiple of `spacing`
 * (zero excluded), as world line segments: two endpoints per segment, each
 * `x, y, z`. Each triangle with three finite values that a level passes
 * strictly between its lowest and highest value adds one segment per level,
 * from the crossings on its two edges that straddle it. `positions` are the
 * mesh's local render positions and `matrix` its column-major world matrix
 * (`scene.matrix4`). With `offset > 0` every segment is drawn twice, moved
 * `offset` along the face's world normal and against it, so it shows over
 * the surface from either side.
 */
export function isoLineSegments(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  matrix: ArrayLike<number>,
  values: ArrayLike<number>,
  spacing: number,
  offset = 0,
): Float32Array {
  if (!(spacing > 0) || !Number.isFinite(spacing)) {
    return new Float32Array(0);
  }
  const m = matrix;
  const nv = Math.min(Math.floor(positions.length / 3), values.length);
  const world = new Float64Array(3 * nv);
  for (let i = 0; i < nv; i++) {
    const x = positions[3 * i];
    const y = positions[3 * i + 1];
    const z = positions[3 * i + 2];
    world[3 * i] = m[0] * x + m[4] * y + m[8] * z + m[12];
    world[3 * i + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
    world[3 * i + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
  }
  const copies = offset > 0 ? [offset, -offset] : [0];
  let out = new Float32Array(1024);
  let at = 0;
  const push = (x: number, y: number, z: number) => {
    if (at + 3 > out.length) {
      const grown = new Float32Array(out.length * 2);
      grown.set(out);
      out = grown;
    }
    out[at++] = x;
    out[at++] = y;
    out[at++] = z;
  };
  const tri = [0, 0, 0];
  const val = [0, 0, 0];
  const cross = [0, 0, 0, 0, 0, 0];
  for (let k = 0; k + 2 < indices.length; k += 3) {
    const a = indices[k];
    const b = indices[k + 1];
    const c = indices[k + 2];
    if ((a === 0 && b === 0 && c === 0) || a >= nv || b >= nv || c >= nv) {
      continue;
    }
    const va = values[a];
    const vb = values[b];
    const vc = values[c];
    if (!Number.isFinite(va) || !Number.isFinite(vb) || !Number.isFinite(vc)) {
      continue;
    }
    const lo = Math.min(va, vb, vc);
    const hi = Math.max(va, vb, vc);
    const first = Math.max(1, Math.floor(lo / spacing) + 1);
    const last = Math.ceil(hi / spacing) - 1;
    if (last < first) {
      continue;
    }
    tri[0] = a;
    tri[1] = b;
    tri[2] = c;
    val[0] = va;
    val[1] = vb;
    val[2] = vc;
    // The face's world normal, for the offset.
    let nx = 0;
    let ny = 0;
    let nz = 0;
    if (offset > 0) {
      const ux = world[3 * b] - world[3 * a];
      const uy = world[3 * b + 1] - world[3 * a + 1];
      const uz = world[3 * b + 2] - world[3 * a + 2];
      const wx = world[3 * c] - world[3 * a];
      const wy = world[3 * c + 1] - world[3 * a + 1];
      const wz = world[3 * c + 2] - world[3 * a + 2];
      nx = uy * wz - uz * wy;
      ny = uz * wx - ux * wz;
      nz = ux * wy - uy * wx;
      const len = Math.hypot(nx, ny, nz);
      if (len > 0) {
        nx /= len;
        ny /= len;
        nz /= len;
      }
    }
    for (let level = first; level <= last; level++) {
      const L = level * spacing;
      if (!(L > lo && L < hi)) {
        continue;
      }
      let found = 0;
      for (let e = 0; e < 3 && found < 2; e++) {
        const i0 = tri[e];
        const i1 = tri[(e + 1) % 3];
        const v0 = val[e];
        const v1 = val[(e + 1) % 3];
        if (v0 >= L === v1 >= L) {
          continue;
        }
        const t = (L - v0) / (v1 - v0);
        cross[3 * found] = world[3 * i0] + t * (world[3 * i1] - world[3 * i0]);
        cross[3 * found + 1] = world[3 * i0 + 1] + t * (world[3 * i1 + 1] - world[3 * i0 + 1]);
        cross[3 * found + 2] = world[3 * i0 + 2] + t * (world[3 * i1 + 2] - world[3 * i0 + 2]);
        found++;
      }
      if (found < 2) {
        continue;
      }
      for (const d of copies) {
        push(cross[0] + d * nx, cross[1] + d * ny, cross[2] + d * nz);
        push(cross[3] + d * nx, cross[4] + d * ny, cross[5] + d * nz);
      }
    }
  }
  return out.slice(0, at);
}
