// The geodesic map and surface path ux logic (#252) against the real engine:
// picks snap to vertices in world coordinates, path taps start, extend and
// close a path (queued in tap order, stale legs dropped), legs follow the
// surface, the trim curve and the drawn segments, which scene changes clear
// the picks, trim and split along a closed loop (one undo step each), and the
// geodesic map's readouts, legend and iso-lines.
import { init, type MeshChange, type MeshHandle, type RayHit, Scene, type SceneLike } from "fast-mesh-wasm";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { AppState, type Item } from "../src/app/state/app-state";
import { blobbySphere } from "../src/io/demo";
import {
  autoIsoSpacing,
  geodesicLegend,
  geodesicOverlay,
  geodesicReadoutLines,
  isoLevelSpacing,
  isoLineSegments,
  MAX_ISO_LEVELS,
  runGeodesicMap,
  sameGeodesicRequest,
  summarizeDistances,
} from "../src/features/inspect/ux/geodesic-map";
import {
  closeSurfaceLoop,
  landSurfacePick,
  liveTriangleCount,
  NO_CUT_MESSAGE,
  queueSurfaceTap,
  splitAlongPath,
  surfacePathIdle,
  trimAlongPath,
} from "../src/features/inspect/ux/surface-path-actions";
import {
  addSource,
  canClose,
  clearedPicks,
  closePath,
  DEFAULT_SURFACE_PATH_TOOL,
  extendPath,
  legFromSurfacePath,
  type PathLeg,
  pathCurve,
  pathLength,
  pathReadout,
  pathSegments,
  pathTapAction,
  pickFromHit,
  polylineLength,
  startPath,
  type SurfacePath,
  type SurfacePick,
  surfacePicksChanged,
  traceLeg,
  undoPathPoint,
} from "../src/features/inspect/ux/surface-paths";
import { geodesicRequestOf } from "../src/features/inspect/ux/use-geodesic-map";

beforeAll(async () => {
  await init();
});

const cleanups: (() => void)[] = [];
afterEach(() => {
  for (const fn of cleanups.splice(0).reverse()) {
    fn();
  }
});

type V3 = [number, number, number];

/** A unit UV sphere around `c` (32 segments, 16 rings): 482 vertices, 960 triangles. */
function sphere(c: V3 = [0, 0, 0], radius = 1) {
  return blobbySphere(c, radius, 32, 16, 0);
}

function setup(opts: { translate?: V3 } = {}) {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  const data = sphere();
  const item = app.addMesh(store, {
    name: "Ball",
    positions: data.positions,
    indices: data.indices,
    center: [0, 0, 0],
    radius: 1,
    volumeId: null,
    preset: null,
  });
  if (opts.translate !== undefined) {
    store.setPlacement(item.mesh, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: opts.translate });
  }
  store.select(item.mesh);
  store.commitFrame();
  scene.clearHistory();
  cleanups.push(() => {
    store.dispose();
    scene.free();
  });
  return { scene, store, app, item };
}

/** The engine's hit on the ray from outside the sphere (at `centre + 3 dir`) toward its centre. */
function hitToward(scene: Scene, dir: V3, centre: V3 = [0, 0, 0]): RayHit {
  const len = Math.hypot(...dir);
  const d: V3 = [dir[0] / len, dir[1] / len, dir[2] / len];
  const hit = scene.raycast([centre[0] + 3 * d[0], centre[1] + 3 * d[1], centre[2] + 3 * d[2]], [-d[0], -d[1], -d[2]]);
  if (hit === null) {
    throw new Error("missed the sphere");
  }
  return hit;
}

/** Four directions around the +Y pole, `polar` radians down from it. */
function capDirections(polar = 0.7): V3[] {
  return [0, 1, 2, 3].map((k) => {
    const phi = (k * Math.PI) / 2 + 0.2;
    return [Math.sin(polar) * Math.cos(phi), Math.cos(polar), Math.sin(polar) * Math.sin(phi)];
  });
}

/** The world area of a mesh's live triangles. */
function area(scene: Scene, mesh: MeshHandle): number {
  const v = scene.views(mesh);
  const p = v.positions();
  const ix = v.indices();
  const m = scene.matrix4(mesh);
  const w = (i: number): V3 => [
    m[0] * p[3 * i] + m[4] * p[3 * i + 1] + m[8] * p[3 * i + 2] + m[12],
    m[1] * p[3 * i] + m[5] * p[3 * i + 1] + m[9] * p[3 * i + 2] + m[13],
    m[2] * p[3 * i] + m[6] * p[3 * i + 1] + m[10] * p[3 * i + 2] + m[14],
  ];
  let sum = 0;
  for (let k = 0; k + 2 < ix.length; k += 3) {
    if (ix[k] === 0 && ix[k + 1] === 0 && ix[k + 2] === 0) continue;
    const [a, b, c] = [w(ix[k]), w(ix[k + 1]), w(ix[k + 2])];
    const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    const s = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    sum += 0.5 * Math.hypot(u[1] * s[2] - u[2] * s[1], u[2] * s[0] - u[0] * s[2], u[0] * s[1] - u[1] * s[0]);
  }
  return sum;
}

/** A scene whose surface queries answer with promises, as a `WorkerScene` does. */
function asyncScene(scene: Scene): SceneLike {
  const inspect = scene.inspect;
  const later = <T>(v: T) => new Promise<T>((resolve) => setTimeout(() => resolve(v), 0));
  return {
    views: (m: MeshHandle) => scene.views(m),
    matrix4: (m: MeshHandle) => scene.matrix4(m),
    inspect: {
      ...inspect,
      geodesicDistances: (m: MeshHandle, s: number | readonly number[], o?: Parameters<typeof inspect.geodesicDistances>[2]) =>
        later(inspect.geodesicDistances(m, s, o)),
      shortestSurfacePath: (m: MeshHandle, s: number | readonly number[], t: number, o?: Parameters<typeof inspect.shortestSurfacePath>[3]) =>
        later(inspect.shortestSurfacePath(m, s, t, o)),
    },
  } as unknown as SceneLike;
}

/** Arms `mode` on `item` directly (the panel's arm does the same, plus disarming other tools). */
function arm(app: AppState, item: Item, mode: "sources" | "path") {
  app.setSurfacePath({ ...app.getSnapshot().surfacePath, mesh: item.mesh, armed: mode });
}

/** Lands taps toward `dirs` in order (through the queue, as the canvas does) and waits for them. */
async function tap(app: AppState, scene: Scene | SceneLike, real: Scene, dirs: readonly V3[]) {
  for (const d of dirs) {
    void queueSurfaceTap(app, scene as SceneLike, Promise.resolve(hitToward(real, d)));
  }
  await surfacePathIdle(app);
}

/** A closed four-point loop around the +Y cap of the sphere. */
async function closedLoop(app: AppState, scene: Scene, item: Item): Promise<SurfacePath> {
  arm(app, item, "path");
  const dirs = capDirections();
  await tap(app, scene, scene, [...dirs, dirs[0]]);
  const path = app.getSnapshot().surfacePath.path;
  expect(path?.closed).toBe(true);
  return path!;
}

describe("picks", () => {
  it("snap_to_the_hit_triangle's_nearest_vertex_in_world_coordinates", () => {
    const { scene, store, item } = setup({ translate: [10, 0, 0] });
    const hit = hitToward(scene, [0, 1, 0], [10, 0, 0]);
    const pick = pickFromHit(scene, hit);
    const local = scene.views(item.mesh).positions();
    // The +Y pole is vertex 0, at the placement's translation plus (0, 1, 0).
    expect(pick.slot).toBe(0);
    expect(local[0]).toBeCloseTo(0, 6);
    expect(pick.point[0]).toBeCloseTo(10, 5);
    expect(pick.point[1]).toBeCloseTo(1, 5);
    void store;
  });

  it("a_source_is_added_once", () => {
    const a: SurfacePick = { slot: 3, point: [0, 0, 0] };
    const b: SurfacePick = { slot: 4, point: [1, 0, 0] };
    const one = addSource([], a);
    expect(one).toEqual([a]);
    expect(addSource(one, { ...a })).toBe(one);
    expect(addSource(one, b)).toEqual([a, b]);
  });

  it("cleared_picks_keep_only_the_method", () => {
    const s = { ...DEFAULT_SURFACE_PATH_TOOL, method: "heatMethod" as const, armed: "path" as const, sources: [{ slot: 1, point: [0, 0, 0] as V3 }] };
    const c = clearedPicks(s);
    expect(c).toEqual({ ...DEFAULT_SURFACE_PATH_TOOL, method: "heatMethod" });
  });
});

describe("path steps", () => {
  const p = (slot: number): SurfacePick => ({ slot, point: [slot, 0, 0] });
  const leg = (length: number): PathLeg => ({
    positions: Float64Array.from([0, 0, 0, length, 0, 0]),
    normals: Float32Array.from([0, 0, 1, 0, 0, 1]),
    length,
  });

  it("taps_start_extend_close_and_ignore", () => {
    expect(pathTapAction(null, p(1))).toEqual({ kind: "start" });
    let path = startPath(p(1));
    expect(pathTapAction(path, p(1))).toEqual({ kind: "ignore", reason: null });
    expect(pathTapAction(path, p(2))).toEqual({ kind: "extend", from: p(1) });
    path = extendPath(path, p(2), leg(1));
    // Two points cannot close yet.
    expect(pathTapAction(path, p(1)).kind).toBe("ignore");
    expect(canClose(path)).toBe(false);
    path = extendPath(path, p(3), leg(2));
    expect(canClose(path)).toBe(true);
    expect(pathTapAction(path, p(2))).toEqual({ kind: "ignore", reason: "That point is already on the path." });
    expect(pathTapAction(path, p(1))).toEqual({ kind: "close", from: p(3), to: p(1) });
    path = closePath(path, leg(3));
    expect(path.closed).toBe(true);
    expect(canClose(path)).toBe(false);
    expect(pathTapAction(path, p(9)).kind).toBe("ignore");
    expect(pathLength(path)).toBe(6);
  });

  it("undo_reopens_a_loop_then_drops_points", () => {
    let path: SurfacePath | null = closePath(extendPath(extendPath(startPath(p(1)), p(2), leg(1)), p(3), leg(2)), leg(3));
    path = undoPathPoint(path);
    expect(path).toMatchObject({ closed: false });
    expect(path!.points.map((q) => q.slot)).toEqual([1, 2, 3]);
    expect(path!.legs).toHaveLength(2);
    path = undoPathPoint(path!);
    expect(path!.points.map((q) => q.slot)).toEqual([1, 2]);
    expect(path!.legs).toHaveLength(1);
    path = undoPathPoint(path!);
    expect(path!.points.map((q) => q.slot)).toEqual([1]);
    expect(undoPathPoint(path!)).toBeNull();
  });

  it("the_trim_curve_joins_legs_once_and_drops_the_repeated_start", () => {
    const l = (pts: number[][]): PathLeg => ({
      positions: Float64Array.from(pts.flat()),
      normals: new Float32Array(3 * pts.length),
      length: polylineLength(pts.flat()),
    });
    const path: SurfacePath = {
      points: [p(1), p(2), p(3)],
      legs: [l([[0, 0, 0], [0.5, 0, 0], [1, 0, 0]]), l([[1, 0, 0], [1, 1, 0]]), l([[1, 1, 0], [0, 0, 0]])],
      closed: true,
    };
    expect(pathCurve(path)).toEqual([
      [0, 0, 0],
      [0.5, 0, 0],
      [1, 0, 0],
      [1, 1, 0],
    ]);
    expect(pathLength(path)).toBeCloseTo(2 + Math.SQRT2, 12);
  });

  it("drawn_segments_pair_points_and_move_along_the_normals", () => {
    const path: SurfacePath = {
      points: [p(1), p(2)],
      legs: [{ positions: Float64Array.from([0, 0, 0, 1, 0, 0, 2, 0, 0]), normals: Float32Array.from([0, 0, 1, 0, 1, 0, 0, 0, 1]), length: 2 }],
      closed: false,
    };
    expect(Array.from(pathSegments(path, 0))).toEqual([0, 0, 0, 1, 0, 0, 1, 0, 0, 2, 0, 0]);
    expect(Array.from(pathSegments(path, 0.5))).toEqual([0, 0, 0.5, 1, 0.5, 0, 1, 0.5, 0, 2, 0, 0.5]);
    expect(pathSegments(null, 1)).toHaveLength(0);
  });

  it("the_readout_counts_points_and_follows_the_unit", () => {
    expect(pathReadout(null, "mm")).toBeNull();
    expect(pathReadout(startPath(p(1)), "mm")).toBe("1 point · click the next point");
    // 12.7 + 12.7 + 25.4 mm.
    const path = closePath(extendPath(extendPath(startPath(p(1)), p(2), leg(12.7)), p(3), leg(12.7)), leg(25.4));
    expect(pathReadout(path, "mm")).toBe("3 points · 50.8 mm · closed loop");
    expect(pathReadout(path, "in")).toBe("3 points · 2 in · closed loop");
  });
});

describe("legs on the surface", () => {
  it("run_from_vertex_to_vertex_along_the_surface_with_outward_normals", async () => {
    const { scene, item } = setup();
    const [d0, d1] = capDirections();
    const a = pickFromHit(scene, hitToward(scene, d0));
    const b = pickFromHit(scene, hitToward(scene, d1));
    const leg = await traceLeg(scene, item.mesh, a, b, "dijkstra");
    const n = leg.positions.length / 3;
    expect(n).toBeGreaterThan(2);
    for (let k = 0; k < 3; k++) {
      expect(leg.positions[k]).toBeCloseTo(a.point[k], 5);
      expect(leg.positions[3 * (n - 1) + k]).toBeCloseTo(b.point[k], 5);
    }
    const chord = Math.hypot(a.point[0] - b.point[0], a.point[1] - b.point[1], a.point[2] - b.point[2]);
    expect(leg.length).toBeGreaterThan(chord);
    expect(leg.length).toBeLessThan(chord * 1.2);
    for (let i = 0; i < n; i++) {
      const nx = leg.normals[3 * i];
      const ny = leg.normals[3 * i + 1];
      const nz = leg.normals[3 * i + 2];
      expect(Math.hypot(nx, ny, nz)).toBeCloseTo(1, 5);
      // Outward on a sphere around the origin.
      expect(nx * leg.positions[3 * i] + ny * leg.positions[3 * i + 1] + nz * leg.positions[3 * i + 2]).toBeGreaterThan(0.9);
    }
  });

  it("a_promise_answering_scene_gives_the_same_leg", async () => {
    const { scene, item } = setup();
    const [d0, , d2] = capDirections();
    const a = pickFromHit(scene, hitToward(scene, d0));
    const b = pickFromHit(scene, hitToward(scene, d2));
    const local = await traceLeg(scene, item.mesh, a, b, "heatMethod");
    const remote = await traceLeg(asyncScene(scene), item.mesh, a, b, "heatMethod");
    expect(Array.from(remote.positions)).toEqual(Array.from(local.positions));
    expect(remote.length).toBe(local.length);
  });

  it("the_leg_follows_the_placement", () => {
    const { scene, item } = setup({ translate: [5, 0, 0] });
    const raw = scene.inspect.shortestSurfacePath(item.mesh, 0, 40);
    const leg = legFromSurfacePath(scene, item.mesh, raw);
    expect(leg.positions[0]).toBeCloseTo(5, 5);
    expect(leg.length).toBeCloseTo(polylineLength(raw.flatMap((q) => q.position)), 9);
  });
});

describe("taps", () => {
  it("land_sources_on_the_armed_mesh_only", async () => {
    const { scene, store, app, item } = setup();
    const data = sphere([4, 0, 0]);
    const other = app.addMesh(store, { name: "Other", positions: data.positions, indices: data.indices, center: [4, 0, 0], radius: 1, volumeId: null, preset: null });
    // Nothing armed: nothing lands.
    await landSurfacePick(app, scene, hitToward(scene, [0, 1, 0]));
    expect(app.getSnapshot().surfacePath.sources).toHaveLength(0);
    arm(app, item, "sources");
    await tap(app, scene, scene, [[0, 1, 0], [0, 1, 0], [1, 0, 0]]);
    expect(app.getSnapshot().surfacePath.sources.map((s) => s.slot)).toEqual([0, pickFromHit(scene, hitToward(scene, [1, 0, 0])).slot]);
    // A hit on the other mesh does nothing.
    await landSurfacePick(app, scene, hitToward(scene, [0, 1, 0], [4, 0, 0]));
    expect(app.getSnapshot().surfacePath.sources).toHaveLength(2);
    void other;
  });

  it("draw_a_path_in_tap_order_and_close_it_on_the_first_point", async () => {
    const { scene, app, item } = setup();
    arm(app, item, "path");
    const worker = asyncScene(scene);
    const dirs = capDirections();
    await tap(app, worker, scene, dirs);
    let path = app.getSnapshot().surfacePath.path!;
    expect(path.points.map((q) => q.slot)).toEqual(dirs.map((d) => pickFromHit(scene, hitToward(scene, d)).slot));
    expect(path.legs).toHaveLength(3);
    expect(path.closed).toBe(false);
    expect(app.getSnapshot().surfacePath.tracing).toBe(0);
    await tap(app, worker, scene, [dirs[0]]);
    path = app.getSnapshot().surfacePath.path!;
    expect(path.closed).toBe(true);
    expect(path.legs).toHaveLength(4);
    // Every leg ends where the next begins.
    for (let i = 0; i < 4; i++) {
      const a = path.legs[i].positions;
      const b = path.legs[(i + 1) % 4].positions;
      expect([a[a.length - 3], a[a.length - 2], a[a.length - 1]]).toEqual([b[0], b[1], b[2]]);
    }
    // Taps on a closed loop say what to do.
    await tap(app, worker, scene, [[0, 1, 0]]);
    expect(app.getSnapshot().surfacePath.error).toMatch(/closed/);
  });

  it("close_loop_traces_back_to_the_first_point", async () => {
    const { scene, app, item } = setup();
    arm(app, item, "path");
    await tap(app, scene, scene, capDirections().slice(0, 3));
    await closeSurfaceLoop(app, scene);
    const path = app.getSnapshot().surfacePath.path!;
    expect(path.closed).toBe(true);
    expect(path.legs).toHaveLength(3);
    const last = path.legs[2].positions;
    expect([last[last.length - 3], last[last.length - 2], last[last.length - 1]]).toEqual(path.points[0].point);
  });

  it("a_leg_landing_after_its_path_was_cleared_is_dropped", async () => {
    const { scene, app, item } = setup();
    arm(app, item, "path");
    const worker = asyncScene(scene);
    const dirs = capDirections();
    await tap(app, worker, scene, [dirs[0]]);
    void queueSurfaceTap(app, worker, Promise.resolve(hitToward(scene, dirs[1])));
    // The leg is tracing now: Clear wins.
    await Promise.resolve();
    await Promise.resolve();
    app.setSurfacePath({ path: null });
    await surfacePathIdle(app);
    expect(app.getSnapshot().surfacePath.path).toBeNull();
    expect(app.getSnapshot().surfacePath.tracing).toBe(0);
  });

  it("a_failed_raycast_drops_only_its_tap", async () => {
    const { scene, app, item } = setup();
    arm(app, item, "sources");
    void queueSurfaceTap(app, scene, Promise.reject(new Error("gone")));
    await tap(app, scene, scene, [[0, 1, 0]]);
    expect(app.getSnapshot().surfacePath.sources.map((s) => s.slot)).toEqual([0]);
  });
});

describe("scene changes", () => {
  const mesh = { id: "a" } as unknown as MeshHandle;
  const other = { id: "b" } as unknown as MeshHandle;
  const change = (m: MeshHandle, patch: Partial<MeshChange> = {}): MeshChange => ({
    mesh: m,
    added: false,
    removed: false,
    placementChanged: false,
    capacityChanged: false,
    positions: [],
    normals: [],
    colors: [],
    indices: [],
    fieldsChanged: [],
    sectionsChanged: [],
    ...patch,
  });

  it("shape_placement_and_removal_clear_the_picks", () => {
    expect(surfacePicksChanged([change(mesh, { positions: [[0, 3]] })], mesh)).toBe(true);
    expect(surfacePicksChanged([change(mesh, { indices: "all" })], mesh)).toBe(true);
    expect(surfacePicksChanged([change(mesh, { placementChanged: true })], mesh)).toBe(true);
    expect(surfacePicksChanged([change(mesh, { capacityChanged: true })], mesh)).toBe(true);
    expect(surfacePicksChanged([change(mesh, { removed: true })], mesh)).toBe(true);
  });

  it("colours_and_other_meshes_do_not", () => {
    expect(surfacePicksChanged([], mesh)).toBe(false);
    expect(surfacePicksChanged([change(mesh, { colors: "all", normals: "all" })], mesh)).toBe(false);
    expect(surfacePicksChanged([change(other, { positions: "all", removed: true })], mesh)).toBe(false);
  });

  it("the_queries_record_no_change", async () => {
    const { scene, app, item } = setup();
    arm(app, item, "path");
    await tap(app, scene, scene, capDirections());
    await runGeodesicMap(scene, { mesh: item.mesh, sources: [0], method: "heatMethod" });
    expect(surfacePicksChanged(scene.commitFrame(), item.mesh)).toBe(false);
  });
});

describe("trim and split", () => {
  it("trim_keeps_either_side_as_one_undo_step", async () => {
    const { scene, store, app, item } = setup();
    const path = await closedLoop(app, scene, item);
    const whole = area(scene, item.mesh);
    const before = liveTriangleCount(scene.views(item.mesh).indices());

    const inside = await trimAlongPath(store, item.mesh, path, "inside");
    expect(inside.keep).toBe("inside");
    expect(inside.removed).toBeGreaterThan(0);
    const insideArea = area(scene, item.mesh);
    expect(insideArea).toBeLessThan(whole / 2);
    expect(inside.triangles).toBe(liveTriangleCount(scene.views(item.mesh).indices()));
    // One undo step puts the whole mesh back.
    store.undo();
    expect(scene.canUndo()).toBe(false);
    expect(liveTriangleCount(scene.views(item.mesh).indices())).toBe(before);
    expect(area(scene, item.mesh)).toBeCloseTo(whole, 9);

    await trimAlongPath(store, item.mesh, path, "outside");
    const outsideArea = area(scene, item.mesh);
    expect(outsideArea).toBeGreaterThan(whole / 2);
    // The two sides make up the whole surface.
    expect(insideArea + outsideArea).toBeCloseTo(whole, 4);
    store.undo();
    expect(scene.canUndo()).toBe(false);
  });

  it("trim_needs_a_closed_loop", async () => {
    const { store, item } = setup();
    await expect(trimAlongPath(store, item.mesh, null, "inside")).rejects.toThrow(/Close the loop/);
    const open = startPath({ slot: 0, point: [0, 1, 0] });
    await expect(trimAlongPath(store, item.mesh, open, "inside")).rejects.toThrow(/Close the loop/);
  });

  it("a_loop_that_cuts_nothing_away_changes_nothing", async () => {
    const { scene, store, app, item } = setup();
    const path = await closedLoop(app, scene, item);
    const calls: string[] = [];
    const stub = {
      scene: {
        boolean: { trimByCurve: () => Promise.resolve({ kept: item.mesh, trianglesAdded: 4, trianglesRemoved: 0, verticesAdded: 2, seamEdges: 4 }) },
        views: (m: MeshHandle) => scene.views(m),
      },
      beginStep: () => calls.push("begin"),
      endStep: () => calls.push("end"),
      cancelStep: () => calls.push("cancel"),
    } as unknown as SceneStore;
    await expect(trimAlongPath(stub, item.mesh, path, "inside")).rejects.toThrow(NO_CUT_MESSAGE);
    expect(calls).toEqual(["begin", "cancel"]);
    void store;
  });

  it("split_makes_two_meshes_as_one_undo_step", async () => {
    const { scene, store, app, item } = setup({ translate: [0.5, -0.25, 0] });
    store.commitFrame();
    scene.clearHistory();
    // The picks follow the placement: taps toward the moved sphere.
    app.setSurfacePath({ ...app.getSnapshot().surfacePath, mesh: item.mesh, armed: "path" });
    const dirs = capDirections();
    for (const d of [...dirs, dirs[0]]) {
      void queueSurfaceTap(app, scene, Promise.resolve(hitToward(scene, d, [0.5, -0.25, 0])));
    }
    await surfacePathIdle(app);
    const path = app.getSnapshot().surfacePath.path!;
    expect(path.closed).toBe(true);
    const whole = area(scene, item.mesh);
    const before = liveTriangleCount(scene.views(item.mesh).indices());

    const r = await splitAlongPath(store, app, item, path);
    expect(r.inside.name).toBe("Ball inside");
    expect(app.getSnapshot().items.map((i) => i.name)).toEqual(["Ball", "Ball inside"]);
    const outerArea = area(scene, item.mesh);
    const innerArea = area(scene, r.inside.mesh);
    expect(innerArea).toBeLessThan(outerArea);
    expect(innerArea + outerArea).toBeCloseTo(whole, 3);
    expect(r.outsideTriangles).toBe(liveTriangleCount(scene.views(item.mesh).indices()));
    expect(r.insideTriangles).toBe(liveTriangleCount(scene.views(r.inside.mesh).indices()));
    // The inside sits where it was cut from: world placement identity, world positions.
    expect(scene.placement(r.inside.mesh).translation).toEqual([0, 0, 0]);
    // One undo makes the mesh whole and empties the new one.
    store.undo();
    expect(scene.canUndo()).toBe(false);
    expect(liveTriangleCount(scene.views(item.mesh).indices())).toBe(before);
    expect(liveTriangleCount(scene.views(r.inside.mesh).indices())).toBe(0);
    store.redo();
    expect(area(scene, r.inside.mesh)).toBeCloseTo(innerArea, 9);
  });

  it("a_failed_split_removes_the_new_mesh", async () => {
    const { scene, store, app, item } = setup();
    const path = await closedLoop(app, scene, item);
    const meshes = () => store.getSnapshot().meshes.size;
    const n = meshes();
    const broken: SurfacePath = { ...path, legs: path.legs.map((l) => ({ ...l, positions: l.positions.map(() => Number.NaN) })) };
    await expect(splitAlongPath(store, app, item, broken)).rejects.toThrow();
    expect(meshes()).toBe(n);
    expect(app.getSnapshot().items).toHaveLength(1);
    expect(scene.canUndo()).toBe(false);
  });
});

describe("geodesic map", () => {
  it("reads_the_SDK's_distances_out", async () => {
    const { scene, item } = setup();
    const sdk = scene.inspect.geodesicDistances(item.mesh, [0]);
    const report = await runGeodesicMap(scene, { mesh: item.mesh, sources: [0], method: "dijkstra" });
    expect(Array.from(report.values)).toEqual(Array.from(sdk));
    expect(report.values[0]).toBe(0);
    // Pole to pole along the edges: a little over half the circumference.
    expect(report.max).toBeGreaterThan(Math.PI * 0.99);
    expect(report.max).toBeLessThan(Math.PI * 1.1);
    expect(report.reached).toBe(482);
    expect(report.unreached).toBe(0);
    const heat = await runGeodesicMap(asyncScene(scene), { mesh: item.mesh, sources: [0], method: "heatMethod" });
    expect(Array.from(heat.values)).toEqual(Array.from(scene.inspect.geodesicDistances(item.mesh, [0], { method: "heatMethod" })));
  });

  it("more_sources_bring_every_vertex_closer", async () => {
    const { scene, item } = setup();
    const one = await runGeodesicMap(scene, { mesh: item.mesh, sources: [0], method: "dijkstra" });
    const two = await runGeodesicMap(scene, { mesh: item.mesh, sources: [0, 481], method: "dijkstra" });
    expect(two.max).toBeLessThan(one.max * 0.6);
    for (let i = 0; i < one.values.length; i++) {
      expect(two.values[i]).toBeLessThanOrEqual(one.values[i]);
    }
  });

  it("counts_vertices_no_source_reaches", async () => {
    const scene = new Scene();
    cleanups.push(() => scene.free());
    const a = sphere();
    const b = sphere([3, 0, 0]);
    const positions = new Float64Array(a.positions.length + b.positions.length);
    positions.set(a.positions);
    positions.set(b.positions, a.positions.length);
    const nv = a.positions.length / 3;
    const indices = new Uint32Array(a.indices.length + b.indices.length);
    indices.set(a.indices);
    indices.set(b.indices.map((v) => v + nv), a.indices.length);
    const mesh = scene.addMesh(positions, indices, { origin: [0, 0, 0] });
    const report = await runGeodesicMap(scene, { mesh, sources: [0], method: "dijkstra" });
    expect(report.reached).toBe(nv);
    expect(report.unreached).toBe(nv);
    expect(geodesicLegend(report, "mm").band?.label).toBe("not reached");
    expect(geodesicReadoutLines(report, "mm")[2]).toBe(`${nv} on parts no source reaches (grey)`);
  });

  it("styles_legends_and_readouts_follow_the_unit", () => {
    const mesh = { id: "m" } as unknown as MeshHandle;
    const report = {
      request: { mesh, sources: [1, 2], method: "dijkstra" as const },
      values: Float32Array.from([0, 25.4, 50.8]),
      max: 50.8,
      mean: 25.4,
      reached: 3,
      unreached: 0,
    };
    expect(geodesicOverlay(report)).toEqual({ mesh, values: report.values, map: "turbo", min: 0, max: 50.8, band: null });
    expect(geodesicOverlay({ ...report, max: 0 }).max).toBe(1);
    expect(geodesicLegend(report, "in")).toEqual({ map: "turbo", lo: "0 in", hi: "2 in", band: null });
    expect(geodesicReadoutLines(report, "in")).toEqual(["2 sources · farthest 2 in · mean 1 in", "edge graph · 3 vertices reached"]);
    expect(geodesicReadoutLines({ ...report, request: { ...report.request, method: "heatMethod" } }, "mm")[1]).toMatch(/^heat method/);
  });

  it("summaries_skip_dead_slots_and_unused_vertices", () => {
    // Vertex 3 is unused; the second triangle slot is dead.
    const s = summarizeDistances(Float32Array.from([0, 1, 2, 9]), Uint32Array.from([0, 1, 2, 0, 0, 0]));
    expect(s).toEqual({ max: 2, mean: 1, reached: 3, unreached: 0 });
  });

  it("requests_compare_by_mesh_sources_and_method", () => {
    const mesh = { id: "m" } as unknown as MeshHandle;
    const other = { id: "n" } as unknown as MeshHandle;
    const r = { mesh, sources: [1, 2], method: "dijkstra" as const };
    expect(sameGeodesicRequest(r, { ...r, sources: [1, 2] })).toBe(true);
    expect(sameGeodesicRequest(r, { ...r, sources: [2, 1] })).toBe(false);
    expect(sameGeodesicRequest(r, { ...r, method: "heatMethod" })).toBe(false);
    expect(sameGeodesicRequest(r, { ...r, mesh: other })).toBe(false);
    expect(sameGeodesicRequest(null, null)).toBe(true);
    expect(sameGeodesicRequest(r, null)).toBe(false);
    expect(geodesicRequestOf(DEFAULT_SURFACE_PATH_TOOL)).toBeNull();
    expect(
      geodesicRequestOf({ ...DEFAULT_SURFACE_PATH_TOOL, mesh, sources: [{ slot: 5, point: [0, 0, 0] }] }),
    ).toEqual({ mesh, sources: [5], method: "dijkstra" });
  });
});

describe("iso-lines", () => {
  const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

  it("a_round_spacing_of_about_a_tenth_and_a_cap_on_the_count", () => {
    expect(autoIsoSpacing(10)).toBe(1);
    expect(autoIsoSpacing(31)).toBe(2);
    expect(autoIsoSpacing(45)).toBe(5);
    expect(autoIsoSpacing(80)).toBe(10);
    expect(autoIsoSpacing(0.37)).toBeCloseTo(0.05, 12);
    expect(autoIsoSpacing(0)).toBe(1);
    expect(isoLevelSpacing(1, 10)).toBe(1);
    expect(isoLevelSpacing(0.001, 10)).toBe(10 / MAX_ISO_LEVELS);
    expect(isoLevelSpacing(0, 0)).toBe(1);
  });

  it("cross_each_triangle_between_its_lowest_and_highest_value", () => {
    // One triangle in the XY plane, 0 at the origin and 2 at both other corners: level 1 crosses midway.
    const positions = [0, 0, 0, 2, 0, 0, 0, 2, 0];
    const segs = isoLineSegments(positions, [0, 1, 2], identity, [0, 2, 2], 1);
    expect(Array.from(segs)).toEqual([1, 0, 0, 0, 1, 0]);
    // Levels at the corner values themselves add nothing; level 0 never draws.
    expect(isoLineSegments(positions, [0, 1, 2], identity, [0, 2, 2], 2)).toHaveLength(0);
    expect(isoLineSegments(positions, [0, 1, 2], identity, [0, 0, 0], 1)).toHaveLength(0);
    // A non-finite value (an unreached vertex) and a dead slot draw nothing.
    expect(isoLineSegments(positions, [0, 1, 2], identity, [0, Infinity, 2], 1)).toHaveLength(0);
    expect(isoLineSegments(positions, [0, 0, 0], identity, [0, 2, 2], 1)).toHaveLength(0);
    // Two levels, two segments.
    expect(isoLineSegments(positions, [0, 1, 2], identity, [0, 3, 3], 1)).toHaveLength(12);
  });

  it("map_through_the_matrix_and_draw_twice_off_the_surface", () => {
    const positions = [0, 0, 0, 2, 0, 0, 0, 2, 0];
    const moved = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 20, 30, 1];
    expect(Array.from(isoLineSegments(positions, [0, 1, 2], moved, [0, 2, 2], 1))).toEqual([11, 20, 30, 10, 21, 30]);
    // Counter-clockwise in XY: the normal is +Z.
    expect(Array.from(isoLineSegments(positions, [0, 1, 2], identity, [0, 2, 2], 1, 0.25))).toEqual([
      1, 0, 0.25, 0, 1, 0.25, 1, 0, -0.25, 0, 1, -0.25,
    ]);
  });

  it("ring_a_sphere_around_its_source", async () => {
    const { scene, item } = setup();
    const report = await runGeodesicMap(scene, { mesh: item.mesh, sources: [0], method: "heatMethod" });
    const v = scene.views(item.mesh);
    const segs = isoLineSegments(v.positions(), v.indices(), scene.matrix4(item.mesh), report.values, 1);
    expect(segs.length).toBeGreaterThan(0);
    // The unit sphere is pi from pole to pole, so there are three rings, at 1, 2 and 3 along the
    // surface from the source pole: at heights cos(1), cos(2) and cos(3).
    const heights = [Math.cos(1), Math.cos(2), Math.cos(3)];
    const seen = [0, 0, 0];
    for (let i = 1; i < segs.length; i += 3) {
      const off = heights.map((h) => Math.abs(segs[i] - h));
      const ring = off.indexOf(Math.min(...off));
      expect(off[ring]).toBeLessThan(0.08);
      seen[ring]++;
    }
    // Each ring goes once around the 32 segments: one line through every triangle it crosses.
    expect(seen.every((n) => n >= 2 * 32)).toBe(true);
  });
});
