// Geodesic picks and surface paths (#252): the app state of the inspect
// panel's geodesic map and path tool, and the math behind them. Clicks on
// the selected mesh snap to its nearest vertex (the SDK's geodesic queries
// start and end on vertices) and land as geodesic sources or path points.
// Each path point after the first adds a leg, the shortest path on the
// surface from the previous point (`inspect.shortestSurfacePath`); closing
// the loop adds the leg back to the first point, and a closed loop is the
// curve that trim and split cut along. Picks are world points: the camera
// can move freely, and any change of the mesh's shape or placement clears
// them. No React and no three, so vitest covers it.
import type { GeodesicMethod, MeshChange, MeshHandle, RayHit, SceneLike, SurfacePathPoint } from "fast-mesh-wasm";
import type { Vec3 } from "../../../io/mesh-data";
import { formatLength, type LengthUnit } from "../../../shared/ux/units";
import { localVertex, snapSlot, worldFromMatrix } from "./inspect-tools";

/** What a click on the mesh adds: a geodesic map source, or a path point. */
export type SurfacePickMode = "sources" | "path";

/** A clicked surface point, snapped to the mesh's nearest vertex. */
export interface SurfacePick {
  /** The vertex slot. */
  slot: number;
  /** The vertex in world coordinates. */
  point: Vec3;
}

/** The surface path between two consecutive path points. */
export interface PathLeg {
  /** World points from the leg's start to its end, `x, y, z` each. */
  positions: Float64Array;
  /** The outward world normal of the face under each point (unit, `x, y, z` each), to draw just off the surface. */
  normals: Float32Array;
  /** The leg's world length. */
  length: number;
}

/** A path on one mesh: its clicked points and the legs between them. */
export interface SurfacePath {
  points: readonly SurfacePick[];
  /** `legs[i]` joins `points[i]` to `points[i + 1]`; a closed path's last leg joins the last point to the first. */
  legs: readonly PathLeg[];
  closed: boolean;
}

/** The geodesic map and path tool's app state (`surfacePath`). */
export interface SurfacePathToolState {
  /** The mesh the picks are on: the selected mesh when the first pick was armed. */
  mesh: MeshHandle | null;
  /** The armed pick, if any: taps on `mesh` add sources or path points. */
  armed: SurfacePickMode | null;
  /** The distance method of the map and the path legs. */
  method: GeodesicMethod;
  /** The geodesic map's sources, in pick order. */
  sources: readonly SurfacePick[];
  /** The path, once its first point landed. */
  path: SurfacePath | null;
  /** Legs being traced (taps queue while one is). */
  tracing: number;
  /** Why the latest pick or cut failed, if it did. */
  error: string | null;
  /** The shown geodesic map's iso-lines: world segment endpoints, two points per segment. */
  isoLines: Float32Array | null;
}

export const DEFAULT_SURFACE_PATH_TOOL: SurfacePathToolState = {
  mesh: null,
  armed: null,
  method: "dijkstra",
  sources: [],
  path: null,
  tracing: 0,
  error: null,
  isoLines: null,
};

/** How far the path and the iso-lines are drawn off the surface, as a fraction of the mesh's radius. */
export const SURFACE_LINE_OFFSET = 0.002;

/** The state with every pick gone (the method stays): the mesh changed, or the tool moved to another mesh. */
export function clearedPicks(state: SurfacePathToolState, mesh: MeshHandle | null = null): SurfacePathToolState {
  return { ...state, mesh, armed: null, sources: [], path: null, error: null, isoLines: null };
}

/** The distance methods the panel offers: the SDK's graph distance and its heat method. */
export const GEODESIC_METHODS: readonly { method: GeodesicMethod; label: string; title: string }[] = [
  {
    method: "dijkstra",
    label: "Fast",
    title: "Shortest distances along the mesh edges: quick on large meshes, a little longer than the true distance",
  },
  {
    method: "heatMethod",
    label: "Accurate",
    title: "The heat method: smooth distances across the faces; slower on large meshes",
  },
];

/**
 * From this many triangles on, the accurate method takes seconds a map or
 * leg (about one per million triangles, and half of that again for the first
 * on a mesh), and the panel says so.
 */
export const SLOW_METHOD_TRIANGLES = 1_000_000;

/** The note the panel shows for a method that is slow on a mesh of `triangles`, or `null`. */
export function slowMethodNote(method: GeodesicMethod, triangles: number): string | null {
  return method === "heatMethod" && triangles >= SLOW_METHOD_TRIANGLES
    ? `Accurate takes a few seconds on ${triangles} triangles for each map or path leg. Cancel stops it.`
    : null;
}

/** A method's label in the readouts. */
export function methodLabel(method: GeodesicMethod): string {
  return method === "heatMethod" ? "heat method" : "edge graph";
}

/** The scene calls a pick and a leg read the mesh through. */
export type PathScene = Pick<SceneLike, "views" | "matrix4" | "inspect">;

/** A ray hit snapped to the hit triangle's nearest corner, in world coordinates. */
export function pickFromHit(
  scene: Pick<SceneLike, "views" | "matrix4">,
  hit: Pick<RayHit, "mesh" | "triangle" | "barycentric">,
): SurfacePick {
  const views = scene.views(hit.mesh);
  const slot = views.indices()[3 * hit.triangle + snapSlot(hit.barycentric)];
  const point = worldFromMatrix(scene.matrix4(hit.mesh), localVertex(views.positions(), slot));
  return { slot, point };
}

/** The sources with `pick` added; a vertex already a source is not added twice. */
export function addSource(sources: readonly SurfacePick[], pick: SurfacePick): readonly SurfacePick[] {
  return sources.some((s) => s.slot === pick.slot) ? sources : [...sources, pick];
}

/** What a path tap does: start the path, add a leg to a new point, close the loop, or nothing. */
export type PathTapAction =
  | { kind: "start" }
  | { kind: "extend"; from: SurfacePick }
  | { kind: "close"; from: SurfacePick; to: SurfacePick }
  | { kind: "ignore"; reason: string | null };

/**
 * What tapping `pick` does to `path`: the first point starts it; a tap on
 * the first point of a path with at least three points closes it; a tap on
 * the latest point, on another point of the path, or any tap on a closed
 * path does nothing.
 */
export function pathTapAction(path: SurfacePath | null, pick: SurfacePick): PathTapAction {
  if (path === null || path.points.length === 0) {
    return { kind: "start" };
  }
  if (path.closed) {
    return { kind: "ignore", reason: "The loop is closed: Undo point reopens it, Clear starts over." };
  }
  const first = path.points[0];
  const last = path.points[path.points.length - 1];
  if (pick.slot === last.slot) {
    return { kind: "ignore", reason: null };
  }
  if (pick.slot === first.slot) {
    return path.points.length >= 3
      ? { kind: "close", from: last, to: first }
      : { kind: "ignore", reason: "A loop needs at least three points." };
  }
  if (path.points.some((p) => p.slot === pick.slot)) {
    return { kind: "ignore", reason: "That point is already on the path." };
  }
  return { kind: "extend", from: last };
}

/** The path with `pick` as its first point. */
export function startPath(pick: SurfacePick): SurfacePath {
  return { points: [pick], legs: [], closed: false };
}

/** The path extended to `pick` along `leg`. */
export function extendPath(path: SurfacePath, pick: SurfacePick, leg: PathLeg): SurfacePath {
  return { points: [...path.points, pick], legs: [...path.legs, leg], closed: false };
}

/** The path closed by `leg`, back to its first point. */
export function closePath(path: SurfacePath, leg: PathLeg): SurfacePath {
  return { points: path.points, legs: [...path.legs, leg], closed: true };
}

/** Whether the path can close: at least three points and still open. */
export function canClose(path: SurfacePath | null): boolean {
  return path !== null && !path.closed && path.points.length >= 3;
}

/**
 * The path without its latest step: a closed loop reopens (its closing leg
 * goes), else the last point and its leg go. `null` once no point is left.
 */
export function undoPathPoint(path: SurfacePath): SurfacePath | null {
  if (path.closed) {
    return { points: path.points, legs: path.legs.slice(0, -1), closed: false };
  }
  if (path.points.length <= 1) {
    return null;
  }
  return { points: path.points.slice(0, -1), legs: path.legs.slice(0, -1), closed: false };
}

/** The path's world length: its legs' lengths added up. */
export function pathLength(path: SurfacePath | null): number {
  return path === null ? 0 : path.legs.reduce((sum, leg) => sum + leg.length, 0);
}

/** The length of a polyline, `x, y, z` each. */
export function polylineLength(positions: ArrayLike<number>): number {
  let sum = 0;
  for (let i = 3; i + 2 < positions.length; i += 3) {
    sum += Math.hypot(positions[i] - positions[i - 3], positions[i + 1] - positions[i - 2], positions[i + 2] - positions[i - 1]);
  }
  return sum;
}

/**
 * The SDK's surface path as a leg: its world points, the outward world
 * normal of the face under each one (from the mesh's current render
 * buffers and matrix), and its length.
 */
export function legFromSurfacePath(
  scene: Pick<SceneLike, "views" | "matrix4">,
  mesh: MeshHandle,
  points: readonly SurfacePathPoint[],
): PathLeg {
  const views = scene.views(mesh);
  const pos = views.positions();
  const idx = views.indices();
  const m = scene.matrix4(mesh);
  const positions = new Float64Array(3 * points.length);
  const normals = new Float32Array(3 * points.length);
  const corner = (t: number, k: number): Vec3 => worldFromMatrix(m, localVertex(pos, idx[3 * t + k]));
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    positions[3 * i] = p.position[0];
    positions[3 * i + 1] = p.position[1];
    positions[3 * i + 2] = p.position[2];
    const t = p.triangle;
    if (t >= 0 && 3 * t + 2 < idx.length) {
      const a = corner(t, 0);
      const b = corner(t, 1);
      const c = corner(t, 2);
      const n = cross(sub(b, a), sub(c, a));
      const len = Math.hypot(n[0], n[1], n[2]);
      if (len > 0) {
        normals[3 * i] = n[0] / len;
        normals[3 * i + 1] = n[1] / len;
        normals[3 * i + 2] = n[2] / len;
      }
    }
  }
  return { positions, normals, length: polylineLength(positions) };
}

/**
 * Traces the shortest surface path from `from` to `to` on `mesh` with the
 * SDK (on the Worker for a `WorkerScene`), as a leg. The SDK's path starts
 * on the source vertex and ends on the target. Aborting `signal` cancels it
 * there and rejects with the signal's reason.
 */
export async function traceLeg(
  scene: PathScene,
  mesh: MeshHandle,
  from: SurfacePick,
  to: SurfacePick,
  method: GeodesicMethod,
  signal?: AbortSignal,
): Promise<PathLeg> {
  const inspect = scene.inspect;
  if (inspect === undefined) {
    throw new Error("This scene has no surface paths");
  }
  const points = await inspect.shortestSurfacePath(mesh, from.slot, to.slot, {
    method,
    ...(signal === undefined ? {} : { signal }),
  });
  if (points.length < 2) {
    throw new Error("No surface path joins these points.");
  }
  return legFromSurfacePath(scene, mesh, points);
}

/** An SDK error for a leg, in panel words. */
export function legErrorMessage(e: unknown): string {
  const text = e instanceof Error ? e.message : String(e);
  return /reachable|component|non-finite|finite and non-negative/i.test(text)
    ? "These points are on separate parts of the mesh: no surface path joins them."
    : text;
}

/**
 * The closed path as the trim curve: every leg's points in order, the
 * point shared by two legs once, and without the repeated first point.
 */
export function pathCurve(path: SurfacePath): Vec3[] {
  const out: Vec3[] = [];
  const same = (a: Vec3, b: Vec3) => a[0] === b[0] && a[1] === b[1] && a[2] === b[2];
  for (const leg of path.legs) {
    const p = leg.positions;
    for (let i = 0; i + 2 < p.length; i += 3) {
      const v: Vec3 = [p[i], p[i + 1], p[i + 2]];
      if (out.length === 0 || !same(out[out.length - 1], v)) {
        out.push(v);
      }
    }
  }
  if (path.closed && out.length > 1 && same(out[0], out[out.length - 1])) {
    out.pop();
  }
  return out;
}

/**
 * The path's legs as line segments for drawing (two endpoints per segment,
 * each `x, y, z`), every point moved `offset` along its face's normal. A
 * negative offset goes under the surface: drawn at both, the line shows
 * over the surface from either side of an open mesh.
 */
export function pathSegments(path: SurfacePath | null, offset: number): Float32Array {
  if (path === null) {
    return new Float32Array(0);
  }
  const total = path.legs.reduce((n, leg) => n + Math.max(0, leg.positions.length / 3 - 1), 0);
  const out = new Float32Array(6 * total);
  let at = 0;
  for (const leg of path.legs) {
    const p = leg.positions;
    const n = leg.normals;
    for (let i = 3; i + 2 < p.length; i += 3) {
      for (const j of [i - 3, i]) {
        out[at++] = p[j] + offset * n[j];
        out[at++] = p[j + 1] + offset * n[j + 1];
        out[at++] = p[j + 2] + offset * n[j + 2];
      }
    }
  }
  return out;
}

/** Whether a change record's buffer range list says something changed. */
function touched(r: MeshChange["positions"]): boolean {
  return r === "all" || r.length > 0;
}

/**
 * Whether a batch of scene changes clears the picks on `mesh`: its removal,
 * or a change of its shape (an edit, a trim, an undo or redo) or placement.
 * Colour-only changes keep them.
 */
export function surfacePicksChanged(changes: readonly MeshChange[], mesh: MeshHandle): boolean {
  return changes.some(
    (c) =>
      c.mesh === mesh &&
      (c.removed || c.placementChanged || c.capacityChanged || touched(c.positions) || touched(c.indices)),
  );
}

/** The path readout: its points, its length in `unit`, and whether it is a closed loop. */
export function pathReadout(path: SurfacePath | null, unit: LengthUnit): string | null {
  if (path === null) {
    return null;
  }
  const n = path.points.length;
  const points = `${n} ${n === 1 ? "point" : "points"}`;
  if (path.legs.length === 0) {
    return `${points} · click the next point`;
  }
  return `${points} · ${formatLength(pathLength(path), unit)}${path.closed ? " · closed loop" : ""}`;
}

function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
