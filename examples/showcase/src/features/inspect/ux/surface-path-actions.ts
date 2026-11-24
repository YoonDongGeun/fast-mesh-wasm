// The geodesic and path picks' engine calls (#252): landing a tap on the
// mesh (a source, or a path point with the leg traced to it), closing the
// loop, and cutting the mesh along a closed loop.
//
// Taps and Close loop run one at a time, in order (a leg on a large mesh
// takes a while, on the Worker in the default engine mode), so quick clicks
// queue instead of racing. Undoing or clearing the path cancels the leg
// being traced, and so does Cancel.
//
// Trim deletes one side of the loop (`boolean.trimByCurve`) as one undo
// step. Split keeps both sides: inside one history step it adds a copy of the
// mesh, trims the mesh to the outside and the copy to the inside, and takes
// the parts the loop is not on out of the copy. The copy has the mesh's own
// coordinates, UVs and placement, so both cuts see the same loop, the same
// part and the same inside. Undo puts the mesh back whole and removes the
// copy.
//
// The engine cuts the part the loop is on and leaves the others alone. A loop
// need not cut anything away (once around a ring, say): the engine still
// imprints it, and reports no removed area. Such a loop is undone and
// reported, and the picks stay.
import type { EditOp, MeshHandle, RayHit, SceneLike, TrimReport, Vec3 } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { AppState, Item } from "../../../app/state/app-state";
import { extractLiveMeshData } from "../../../engine/workbench-ops";
import { liveTriangleIds } from "../../mesh-edit/ux/batch-boolean";
import { countLiveTriangles } from "../../select/ux/face-sets";
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

/** The leg being traced per app, if any. */
const tracing = new WeakMap<AppState, AbortController>();

/** Cancels the leg being traced: the path stays as it is. Queued taps still land. */
export function cancelSurfaceLeg(app: AppState): void {
  tracing.get(app)?.abort();
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
  const run = new AbortController();
  tracing.set(app, run);
  app.setSurfacePath({ tracing: start.tracing + 1 });
  try {
    const leg = await traceLeg(scene, mesh, from, to, start.method, run.signal);
    const now = app.getSnapshot().surfacePath;
    if (!run.signal.aborted && now.mesh === mesh && now.path === before) {
      app.setSurfacePath({ path: next(before, leg), error: null });
    }
  } catch (e) {
    const now = app.getSnapshot().surfacePath;
    if (!run.signal.aborted && now.mesh === mesh && now.path === before) {
      app.setSurfacePath({ error: legErrorMessage(e) });
    }
  } finally {
    if (tracing.get(app) === run) {
      tracing.delete(app);
    }
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

/** Deletes one side of the loop `curve` from a mesh; throws when that cuts nothing away. */
type CutAway = (mesh: MeshHandle, curve: readonly Vec3[], side: TrimKeep) => Promise<void>;

/** The scene's `boolean.trimByCurve`, on a local `Scene` or a `WorkerScene`, as a {@link CutAway}. */
function cutAwayCall(scene: SceneLike): CutAway {
  const domain = scene.boolean as { trimByCurve?: TrimCall } | undefined;
  const trim = domain?.trimByCurve;
  if (trim === undefined) {
    throw new Error("This scene cannot trim along a curve.");
  }
  return async (mesh, curve, side) => {
    const report = await trim(mesh, curve, { closed: true, deleteInside: side === "inside" });
    if (report.removedArea === 0) {
      throw new Error(NO_CUT_MESSAGE);
    }
  };
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
  /** Triangles the mesh has now, the parts the loop is not on included. */
  triangles: number;
}

/**
 * Trims `mesh` along the closed `path`, keeping `keep`: one undo step. Of the
 * part the loop is on, the inside is the side away from its open edges, or on
 * a closed part the side with the smaller area. Other parts stay. A loop that
 * cuts nothing away is undone (see {@link settleChanges}) and throws
 * {@link NO_CUT_MESSAGE}.
 */
export async function trimAlongPath(
  store: SceneStore,
  mesh: MeshHandle,
  path: SurfacePath | null,
  keep: TrimKeep,
): Promise<PathTrimResult> {
  const curve = loopCurve(path);
  const cutAway = cutAwayCall(store.scene);
  store.beginStep();
  try {
    await cutAway(mesh, curve, keep === "outside" ? "inside" : "outside");
  } catch (err) {
    store.cancelStep();
    throw err;
  }
  store.endStep();
  return { keep, triangles: countLiveTriangles(store.scene.views(mesh).indices()) };
}

/** A mesh's index buffer before and after a cut. */
export interface CutIndices {
  before: ArrayLike<number>;
  after: ArrayLike<number>;
}

/**
 * The triangles of a mesh's copy that are on the parts the loop is not on:
 * those that neither the cut of the mesh nor the cut of the copy to the other
 * side changed (every triangle of the loop's part is on one side of it, or
 * across it). `slots[k]` is the slot the copy's triangle `k` has in the mesh.
 */
export function uncutTriangles(slots: ArrayLike<number>, mesh: CutIndices, copy: CutIndices): Uint32Array {
  const same = ({ before, after }: CutIndices, t: number) =>
    before[3 * t] === after[3 * t] && before[3 * t + 1] === after[3 * t + 1] && before[3 * t + 2] === after[3 * t + 2];
  const out: number[] = [];
  for (let k = 0; k < slots.length; k++) {
    if (same(copy, k) && same(mesh, slots[k])) {
      out.push(k);
    }
  }
  return Uint32Array.from(out);
}

/** The edit ops that take `triangles` out of a mesh, with the vertices only they use. */
export function removalOps(
  indices: ArrayLike<number>,
  vertexCount: number,
  triangles: ArrayLike<number>,
): EditOp[] {
  if (triangles.length === 0) {
    return [];
  }
  const gone = new Uint8Array(Math.floor(indices.length / 3));
  const free = new Uint8Array(vertexCount);
  for (let i = 0; i < triangles.length; i++) {
    const t = triangles[i];
    gone[t] = 1;
    for (let c = 3 * t; c < 3 * t + 3; c++) {
      free[indices[c]] = 1;
    }
  }
  for (const t of liveTriangleIds(indices)) {
    if (gone[t] === 0) {
      for (let c = 3 * t; c < 3 * t + 3; c++) {
        free[indices[c]] = 0;
      }
    }
  }
  const vertices: number[] = [];
  for (let v = 0; v < vertexCount; v++) {
    if (free[v] === 1) {
      vertices.push(v);
    }
  }
  return [
    { op: "removeTriangles", triangles: Uint32Array.from(triangles) },
    ...(vertices.length > 0 ? [{ op: "removeVertices" as const, vertices: Uint32Array.from(vertices) }] : []),
  ];
}

/** What a split did: the new mesh holding the inside, and both parts' triangles. */
export interface PathSplitResult {
  inside: Item;
  outsideTriangles: number;
  insideTriangles: number;
}

/**
 * Splits `item` along the closed `path` into two meshes, one undo step:
 * `item` keeps the outside (and every part the loop is not on), a new mesh
 * named `<name> inside` (listed once the split lands) holds the inside, with
 * the origin, placement and UVs it had there. Undo makes `item` whole again
 * and removes the new mesh. A loop that cuts nothing away changes nothing:
 * the step is cancelled, the new mesh with it, and it throws
 * {@link NO_CUT_MESSAGE}.
 */
export async function splitAlongPath(
  store: SceneStore,
  app: AppState,
  item: Item,
  path: SurfacePath | null,
): Promise<PathSplitResult> {
  const curve = loopCurve(path);
  const cutAway = cutAwayCall(store.scene);
  const scene = store.scene;
  const indices = (mesh: MeshHandle) => scene.views(mesh).indices();
  const before = Uint32Array.from(indices(item.mesh));
  const whole = extractLiveMeshData(scene, item.mesh);
  if (whole.indices.length === 0) {
    throw new Error("The mesh has no triangles to split.");
  }
  const copied = whole.indices.slice(); // a Worker scene takes the added arrays away
  const { origin } = whole;
  let copy: MeshHandle;
  store.beginStep();
  try {
    copy = store.addMesh(
      whole.positions.map((x, i) => x + origin[i % 3]),
      whole.indices,
      { origin, undoable: true, ...(whole.uvs ? { uvs: whole.uvs } : {}) },
    );
    store.setPlacement(copy, whole.placement);
    await cutAway(item.mesh, curve, "inside");
    await cutAway(copy, curve, "outside");
    const others = uncutTriangles(
      liveTriangleIds(before),
      { before, after: indices(item.mesh) },
      { before: copied, after: indices(copy) },
    );
    const drop = removalOps(indices(copy), Math.floor(scene.views(copy).positions().length / 3), others);
    if (drop.length > 0) {
      await store.edit(copy, drop);
    }
  } catch (err) {
    // Takes the copy out again.
    store.cancelStep();
    throw err;
  }
  store.endStep();
  const cut = extractLiveMeshData(scene, copy);
  const insideTriangles = cut.indices.length / 3;
  const inside = app.listMesh(copy, {
    name: `${item.name} inside`,
    triangles: insideTriangles,
    center: origin,
    radius: cut.radius,
  });
  return { inside, outsideTriangles: countLiveTriangles(indices(item.mesh)), insideTriangles };
}
