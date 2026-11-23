// The geodesic and path picks' engine calls (#252): landing a tap on the
// mesh (a source, or a path point with the leg traced to it), closing the
// loop, and cutting the mesh along a closed loop.
//
// Taps and Close loop run one at a time, in order (a leg on a large mesh
// takes a while, on the Worker in the default engine mode), so quick clicks
// queue instead of racing; a leg that lands after its path was undone or
// cleared is dropped.
//
// Trim deletes one side of the loop (`boolean.trimByCurve`) as one undo
// step. Split keeps both sides: inside one history step it fills a new mesh
// (added empty first: adding a mesh is not an undo step) with a copy of the
// mesh, then trims the mesh to the outside and the copy to the inside. Undo
// puts the mesh back whole and empties the copy.
//
// A loop need not cut anything away (once around a ring, say). The engine
// then still imprints it, replacing the triangles it crosses, so triangle
// counts do not tell a cut from an imprint; the surface does, which an
// imprint keeps and a cut does not. Such a loop is undone and reported, and
// the picks stay.
import type { MeshHandle, RayHit, SceneLike, TrimReport, Vec3 } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { AppState, Item } from "../../../app/state/app-state";
import { extractLiveMeshData } from "../../../engine/workbench-ops";
import { localToWorld, remapCorners } from "../../mesh-edit/ux/batch-boolean";
import {
  addSource,
  closePath,
  extendPath,
  legErrorMessage,
  type PathScene,
  pathCurve,
  pathTapAction,
  pickFromHit,
  startPath,
  type SurfacePath,
  type SurfacePick,
  traceLeg,
} from "./surface-paths";

/** The per-app queue of taps and Close loop presses. */
const queues = new WeakMap<AppState, Promise<void>>();

/** Runs `task` after every earlier one on `app`'s queue; its failure does not stop the queue. */
function enqueue(app: AppState, task: () => Promise<void>): Promise<void> {
  const run = (queues.get(app) ?? Promise.resolve()).then(task);
  queues.set(
    app,
    run.catch(() => undefined),
  );
  return run;
}

/** Waits for every queued tap and Close loop press on `app`. */
export function surfacePathIdle(app: AppState): Promise<void> {
  return queues.get(app) ?? Promise.resolve();
}

/** Traces the leg `from` → `to` and lands `next(leg)` unless the path or its mesh changed meanwhile. */
async function traceAndLand(
  app: AppState,
  scene: PathScene,
  mesh: MeshHandle,
  from: SurfacePick,
  to: SurfacePick,
  next: (path: SurfacePath, leg: Awaited<ReturnType<typeof traceLeg>>) => SurfacePath,
): Promise<void> {
  const start = app.getSnapshot().surfacePath;
  const before = start.path;
  if (before === null) {
    return;
  }
  app.setSurfacePath({ tracing: start.tracing + 1 });
  try {
    const leg = await traceLeg(scene, mesh, from, to, start.method);
    const now = app.getSnapshot().surfacePath;
    if (now.mesh === mesh && now.path === before) {
      app.setSurfacePath({ path: next(before, leg), error: null });
    }
  } catch (e) {
    const now = app.getSnapshot().surfacePath;
    if (now.mesh === mesh && now.path === before) {
      app.setSurfacePath({ error: legErrorMessage(e) });
    }
  } finally {
    const now = app.getSnapshot().surfacePath;
    app.setSurfacePath({ tracing: Math.max(0, now.tracing - 1) });
  }
}

/**
 * Lands a hit on the armed pick: a source is added at once; a path point
 * starts the path, or adds the leg traced to it, or closes the loop (a tap
 * on the first point). A hit on another mesh, or with nothing armed, does
 * nothing.
 */
export async function landSurfacePick(
  app: AppState,
  scene: PathScene,
  hit: Pick<RayHit, "mesh" | "triangle" | "barycentric">,
): Promise<void> {
  const s = app.getSnapshot().surfacePath;
  if (s.armed === null || s.mesh === null || hit.mesh !== s.mesh) {
    return;
  }
  const mesh = s.mesh;
  let pick: SurfacePick;
  try {
    pick = pickFromHit(scene, hit);
  } catch {
    return; // removed meanwhile
  }
  if (s.armed === "sources") {
    const sources = addSource(s.sources, pick);
    if (sources !== s.sources) {
      app.setSurfacePath({ sources, error: null });
    }
    return;
  }
  const action = pathTapAction(s.path, pick);
  switch (action.kind) {
    case "start":
      app.setSurfacePath({ path: startPath(pick), error: null });
      return;
    case "ignore":
      if (action.reason !== null) {
        app.setSurfacePath({ error: action.reason });
      }
      return;
    case "extend":
      await traceAndLand(app, scene, mesh, action.from, pick, (path, leg) => extendPath(path, pick, leg));
      return;
    case "close":
      await traceAndLand(app, scene, mesh, action.from, action.to, closePath);
      return;
  }
}

/** Queues a tap whose raycast is `hit` (started at once, so it sees the camera of the tap): it lands in tap order. */
export function queueSurfaceTap(
  app: AppState,
  scene: PathScene,
  hit: Promise<Pick<RayHit, "mesh" | "triangle" | "barycentric"> | null>,
): Promise<void> {
  // Settled here so a failed raycast (a mesh removed meanwhile) drops only this tap.
  const settled = hit.catch(() => null);
  return enqueue(app, async () => {
    const h = await settled;
    if (h !== null) {
      await landSurfacePick(app, scene, h);
    }
  });
}

/** Queues Close loop: the leg from the last point back to the first, once at least three points are down. */
export function closeSurfaceLoop(app: AppState, scene: PathScene): Promise<void> {
  return enqueue(app, async () => {
    const s = app.getSnapshot().surfacePath;
    const path = s.path;
    if (s.mesh === null || path === null || path.closed || path.points.length < 3) {
      return;
    }
    await traceAndLand(app, scene, s.mesh, path.points[path.points.length - 1], path.points[0], closePath);
  });
}

/** Which side of the loop Trim keeps. */
export type TrimKeep = "inside" | "outside";

/** Shown when the loop leaves the mesh in one piece. */
export const NO_CUT_MESSAGE = "The loop does not cut the surface into two sides: nothing to trim.";

type TrimCall = (
  mesh: MeshHandle,
  curve: readonly Vec3[],
  options: { closed: boolean; deleteInside: boolean },
) => TrimReport | Promise<TrimReport>;

/** The scene's `boolean.trimByCurve`, on a local `Scene` or a `WorkerScene`. */
function trimCall(scene: SceneLike): TrimCall {
  const domain = scene.boolean as { trimByCurve?: TrimCall } | undefined;
  const trim = domain?.trimByCurve;
  if (trim === undefined) {
    throw new Error("This scene cannot trim along a curve.");
  }
  return (mesh, curve, options) => trim(mesh, curve, options);
}

/** The loop as the trim curve, checked: a closed path with at least three distinct points. */
function loopCurve(path: SurfacePath | null): Vec3[] {
  if (path === null || !path.closed) {
    throw new Error("Close the loop first.");
  }
  const curve = pathCurve(path);
  if (curve.length < 3) {
    throw new Error("The loop is too short to cut along.");
  }
  return curve;
}

/** The triangles a mesh's index buffer holds, dead slots (`0, 0, 0`) left out. */
export function liveTriangleCount(indices: ArrayLike<number>): number {
  let n = 0;
  for (let k = 0; k + 2 < indices.length; k += 3) {
    if (indices[k] !== 0 || indices[k + 1] !== 0 || indices[k + 2] !== 0) {
      n++;
    }
  }
  return n;
}

/** What {@link surfaceCut} compares of a mesh before and after a trim. */
export interface SurfaceStats {
  /** Live triangles. */
  triangles: number;
  /** Their area, in local units. */
  area: number;
  /** The largest local coordinate over the extent: how coarsely `f32` places a vertex for the mesh's size. */
  reach: number;
}

/** A mesh's {@link SurfaceStats}, in one pass over its render buffers. */
export function surfaceStats(positions: ArrayLike<number>, indices: ArrayLike<number>): SurfaceStats {
  let triangles = 0;
  let area = 0;
  let far = 0;
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let k = 0; k + 2 < indices.length; k += 3) {
    const i0 = indices[k];
    const i1 = indices[k + 1];
    const i2 = indices[k + 2];
    if (i0 === 0 && i1 === 0 && i2 === 0) {
      continue;
    }
    triangles++;
    for (let corner = 0; corner < 3; corner++) {
      const i = corner === 0 ? i0 : corner === 1 ? i1 : i2;
      for (let c = 0; c < 3; c++) {
        const v = positions[3 * i + c];
        if (v < lo[c]) lo[c] = v;
        if (v > hi[c]) hi[c] = v;
        const a = Math.abs(v);
        if (a > far) far = a;
      }
    }
    const a = 3 * i0;
    const b = 3 * i1;
    const d = 3 * i2;
    const ux = positions[b] - positions[a];
    const uy = positions[b + 1] - positions[a + 1];
    const uz = positions[b + 2] - positions[a + 2];
    const wx = positions[d] - positions[a];
    const wy = positions[d + 1] - positions[a + 1];
    const wz = positions[d + 2] - positions[a + 2];
    area += 0.5 * Math.hypot(uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx);
  }
  const extent = triangles === 0 ? 0 : Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  return { triangles, area, reach: extent > 0 ? far / extent : 0 };
}

/** An area change below this share of the area is an imprint's rounding, not a cut. */
const AREA_NOISE = 1e-6;
/** The share grows with the coordinates' reach, as `f32` places vertices more coarsely. */
const AREA_NOISE_PER_REACH = 1e-7;

/**
 * Whether a trim took part of the surface away. An imprint alone never loses
 * triangles, and keeps the area up to `f32` rounding: about 1e-10 of it on
 * a mesh centred on its origin, up to about 1e-6 with coordinates hundreds of
 * extents from it. So a cut shows as fewer triangles, or as area lost beyond
 * that. (A region smaller than both the noise and what the imprint added is
 * not told from none.)
 */
export function surfaceCut(before: SurfaceStats, after: SurfaceStats): boolean {
  if (after.triangles < before.triangles) {
    return true;
  }
  return after.area < before.area * (1 - Math.max(AREA_NOISE, AREA_NOISE_PER_REACH * before.reach));
}

/** The mesh's current {@link SurfaceStats}. */
function statsOf(scene: SceneLike, mesh: MeshHandle): SurfaceStats {
  const views = scene.views(mesh);
  return surfaceStats(views.positions(), views.indices());
}

/**
 * Waits until what was asked of the engine so far has landed and its change
 * records are out: a `WorkerScene` takes a cancelled step back later than
 * the call that asked for it. `mesh` is a live mesh to ask about.
 */
export async function settleChanges(store: SceneStore, mesh: MeshHandle): Promise<void> {
  try {
    await store.scene.inspect?.measure(mesh);
  } catch {
    // Removed meanwhile: nothing more to wait for.
  }
  store.commitFrame();
}

/** What a trim did. */
export interface PathTrimResult {
  keep: TrimKeep;
  /** Triangles the mesh has now. */
  triangles: number;
}

/**
 * Trims `mesh` along the closed `path`, keeping `keep`: one undo step. The
 * inside is the side away from the mesh's open edges, or on a closed mesh
 * the side with the smaller area. A loop that cuts nothing away is undone
 * (see {@link settleChanges}) and throws {@link NO_CUT_MESSAGE}.
 */
export async function trimAlongPath(
  store: SceneStore,
  mesh: MeshHandle,
  path: SurfacePath | null,
  keep: TrimKeep,
): Promise<PathTrimResult> {
  const curve = loopCurve(path);
  const trim = trimCall(store.scene);
  const before = statsOf(store.scene, mesh);
  let after: SurfaceStats;
  store.beginStep();
  try {
    await trim(mesh, curve, { closed: true, deleteInside: keep === "outside" });
    after = statsOf(store.scene, mesh);
    if (!surfaceCut(before, after)) {
      throw new Error(NO_CUT_MESSAGE);
    }
  } catch (err) {
    store.cancelStep();
    throw err;
  }
  store.endStep();
  return { keep, triangles: after.triangles };
}

/** What a split did: the new mesh holding the inside, and both parts' triangles. */
export interface PathSplitResult {
  inside: Item;
  outsideTriangles: number;
  insideTriangles: number;
}

/** The centre of world positions' bounding box (the origin of `[]`). */
function boxCentre(world: ArrayLike<number>): Vec3 {
  if (world.length < 3) {
    return [0, 0, 0];
  }
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i + 2 < world.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      const v = world[i + a];
      if (v < lo[a]) lo[a] = v;
      if (v > hi[a]) hi[a] = v;
    }
  }
  return [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
}

/**
 * Splits `item` along the closed `path` into two meshes, one undo step:
 * `item` keeps the outside, a new mesh named `<name> inside` (listed once
 * the split lands) holds the inside. Undo makes `item` whole again and
 * empties the new mesh, which stays listed (adding a mesh is not an undo
 * step). A loop that cuts nothing away changes nothing: the new mesh is
 * removed again and it throws {@link NO_CUT_MESSAGE}.
 */
export async function splitAlongPath(
  store: SceneStore,
  app: AppState,
  item: Item,
  path: SurfacePath | null,
): Promise<PathSplitResult> {
  const curve = loopCurve(path);
  const trim = trimCall(store.scene);
  const scene = store.scene;
  const snap = extractLiveMeshData(scene, item.mesh);
  const world = localToWorld(scene.matrix4(item.mesh), snap.positions);
  if (snap.indices.length === 0) {
    throw new Error("The mesh has no triangles to split.");
  }
  // Each side against itself: the copy is in world units, the mesh in its own.
  const before = statsOf(scene, item.mesh);
  const copy = store.addMesh(new Float64Array(0), new Uint32Array(0), { origin: boxCentre(world) });
  let outer: SurfaceStats;
  let inner: SurfaceStats;
  store.beginStep();
  try {
    // An edit takes world positions; the copy's triangles follow the ids its new vertices got.
    const added = await store.edit(copy, [{ op: "addVertices", positions: world }]);
    const ids = added.added[0];
    if (ids === undefined || ids.length !== world.length / 3) {
      throw new Error("The edit did not report the added vertices.");
    }
    await store.edit(copy, [{ op: "addTriangles", triangles: remapCorners(snap.indices, ids) }]);
    const copyBefore = statsOf(scene, copy);
    await trim(item.mesh, curve, { closed: true, deleteInside: true });
    outer = statsOf(scene, item.mesh);
    if (!surfaceCut(before, outer)) {
      throw new Error(NO_CUT_MESSAGE);
    }
    await trim(copy, curve, { closed: true, deleteInside: false });
    inner = statsOf(scene, copy);
    if (!surfaceCut(copyBefore, inner)) {
      throw new Error(NO_CUT_MESSAGE);
    }
    store.endStep();
  } catch (err) {
    store.cancelStep();
    store.removeMesh(copy);
    throw err;
  }
  const inside = app.adoptSceneMesh(store, copy, `${item.name} inside`);
  return { inside, outsideTriangles: outer.triangles, insideTriangles: inner.triangles };
}
