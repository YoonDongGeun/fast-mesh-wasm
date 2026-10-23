// `contactHeatmap` (`fast-mesh-wasm/store`): the preset's stops, range and
// validation (pure), and over both hosts (the synchronous `Scene`, and
// `WorkerScene` against the Worker engine in the Node `worker_threads`
// harness, built dist/, `serve` mode): a tolerance change reuses the store's
// field, `SceneStore.fetchColoringValues` reads it, and `exactDepth` colours
// deep penetration.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it, vi } from "vitest";
import { type MeshHandle, type Rgba, Scene } from "../src/node.js";
import { contactHeatmap, createSceneStore, type DistanceColoring, type SceneStore } from "../src/store/index.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";
import type { Request } from "../src/worker/protocol.js";

const DEEP: Rgba = [0x7f, 0x00, 0x00, 255];
const SHALLOW: Rgba = [0xff, 0x6b, 0x6b, 255];
const WHITE: Rgba = [0xff, 0xff, 0xff, 255];
const NEAR: Rgba = [0x4c, 0xaf, 0x50, 255];
const FAR: Rgba = [0x1e, 0x3a, 0x8a, 255];
/** The scene's default base colour (above the range). */
const BASE: Rgba = [200, 200, 200, 255];

/** The colour of `v` under a heatmap colouring, as the engine computes it (f32 stops, rounded lerp). */
function colorAt(c: Omit<DistanceColoring, "distanceTo">, v: number): Rgba {
  const [lo, hi] = c.range;
  if (v > hi) {
    return BASE;
  }
  const stops = (c.map as { stops: readonly (readonly [number, Rgba])[] }).stops;
  const t = Math.min(1, Math.max(0, (v - lo) / (hi - lo)));
  for (let k = 0; k + 1 < stops.length; k++) {
    const [t0, c0] = stops[k];
    const [t1, c1] = stops[k + 1];
    if (t <= t1) {
      const f = (t - t0) / (t1 - t0);
      return c0.map((x, i) => Math.round(x + (c1[i] - x) * f)) as unknown as Rgba;
    }
  }
  return stops[stops.length - 1][1];
}

describe("contactHeatmap", () => {
  it("heatmap_defaults", () => {
    const fast = contactHeatmap({ depth: 1, gap: 2 });
    expect(fast.range).toEqual([-1, 2]);
    // Fast mode bounds the field at 2 · max(depth, gap): values just beyond
    // the range end stay finite, so the outer boundary is a value boundary.
    expect(fast.maxDistance).toBe(4);
    expect(fast.below).toBe("clamp");
    expect(fast.above).toBe("base");
    expect(fast.sign).toBe("auto");
    expect(contactHeatmap({ depth: 3, gap: 2 }).maxDistance).toBe(6);
    expect(contactHeatmap({ depth: 1, gap: 2, exactDepth: true }).maxDistance).toBe(Infinity);
    expect(contactHeatmap({ depth: 1, gap: 2, exactDepth: false }).maxDistance).toBe(4);
    expect(contactHeatmap({ depth: 1, gap: 2, sign: "winding" }).sign).toBe("winding");

    // Tolerance 0: the band is one stop at t(0).
    const stops = (fast.map as { stops: readonly (readonly [number, Rgba])[] }).stops;
    expect(stops.map((s) => s[1])).toEqual([DEEP, SHALLOW, WHITE, NEAR, FAR]);
    expect(stops[0][0]).toBe(0);
    expect(stops[2][0]).toBeCloseTo(1 / 3, 12);
    expect(stops[4][0]).toBe(1);
    for (let k = 1; k < stops.length; k++) {
      expect(Math.fround(stops[k][0])).toBeGreaterThan(Math.fround(stops[k - 1][0]));
    }
    expect(colorAt(fast, -1)).toEqual(DEEP);
    expect(colorAt(fast, 0)).toEqual(WHITE);
    expect(colorAt(fast, 2)).toEqual(FAR);
    expect(colorAt(fast, -5)).toEqual(DEEP);
    expect(colorAt(fast, 2.5)).toEqual(BASE);

    // A tolerance adds the band's second edge.
    const banded = contactHeatmap({ depth: 1, gap: 2, tolerance: 0.1 });
    const bandStops = (banded.map as { stops: readonly (readonly [number, Rgba])[] }).stops;
    expect(bandStops.map((s) => s[1])).toEqual([DEEP, SHALLOW, WHITE, WHITE, NEAR, FAR]);
    for (let k = 1; k < bandStops.length; k++) {
      expect(Math.fround(bandStops[k][0])).toBeGreaterThan(Math.fround(bandStops[k - 1][0]));
    }

    // Colour overrides.
    const custom = contactHeatmap({ depth: 1, gap: 1, colors: { deep: [1, 2, 3, 255], far: [4, 5, 6, 255] } });
    const customStops = (custom.map as { stops: readonly (readonly [number, Rgba])[] }).stops;
    expect(customStops.map((s) => s[1])).toEqual([[1, 2, 3, 255], SHALLOW, WHITE, NEAR, [4, 5, 6, 255]]);
  });

  it("heatmap_band", () => {
    const c = contactHeatmap({ depth: 1, gap: 1, tolerance: 0.1 });
    for (const v of [-0.1, -0.05, 0, 0.05, 0.1]) {
      expect(colorAt(c, v)).toEqual(WHITE);
    }
    // Just past the band's edge stops (1e-4 of the range away): the shallow and near colours.
    expect(colorAt(c, -0.1003)).toEqual(SHALLOW);
    expect(colorAt(c, 0.1003)).toEqual(NEAR);
  });

  it("heatmap_penetration_only", () => {
    // gap 0: colour penetration only; above the contact band is the base colour.
    const c = contactHeatmap({ depth: 1, gap: 0 });
    expect(c.range).toEqual([-1, 0]);
    expect(c.maxDistance).toBe(2);
    expect(contactHeatmap({ depth: 1, gap: 0, exactDepth: true }).maxDistance).toBe(Infinity);
    const stops = (c.map as { stops: readonly (readonly [number, Rgba])[] }).stops;
    expect(stops.map((s) => s[1])).toEqual([DEEP, SHALLOW, WHITE]);
    expect(stops[0][0]).toBe(0);
    expect(stops[stops.length - 1][0]).toBe(1);
    for (let k = 1; k < stops.length; k++) {
      expect(Math.fround(stops[k][0])).toBeGreaterThan(Math.fround(stops[k - 1][0]));
    }
    expect(colorAt(c, -1)).toEqual(DEEP);
    expect(colorAt(c, -3)).toEqual(DEEP);
    expect(colorAt(c, -0.0003)).toEqual(SHALLOW);
    expect(colorAt(c, 0)).toEqual(WHITE);
    expect(colorAt(c, 0.001)).toEqual(BASE);

    // With a tolerance the white band reaches +tolerance, and only beyond it is base.
    const banded = contactHeatmap({ depth: 1, gap: 0, tolerance: 0.1 });
    expect(banded.range).toEqual([-1, 0.1]);
    const bandStops = (banded.map as { stops: readonly (readonly [number, Rgba])[] }).stops;
    expect(bandStops.map((s) => s[1])).toEqual([DEEP, SHALLOW, WHITE, WHITE]);
    expect(bandStops[bandStops.length - 1][0]).toBe(1);
    for (let k = 1; k < bandStops.length; k++) {
      expect(Math.fround(bandStops[k][0])).toBeGreaterThan(Math.fround(bandStops[k - 1][0]));
    }
    for (const v of [-0.1, 0, 0.1]) {
      expect(colorAt(banded, v)).toEqual(WHITE);
    }
    expect(colorAt(banded, -0.1003)).toEqual(SHALLOW);
    expect(colorAt(banded, 0.1001)).toEqual(BASE);
  });

  it("heatmap_validation", () => {
    const bad = [
      { depth: 0, gap: 1 },
      { depth: -1, gap: 1 },
      { depth: Number.NaN, gap: 1 },
      { depth: Infinity, gap: 1 },
      { depth: 0, gap: 0 },
      { depth: 1, gap: 0, tolerance: 1 },
      { depth: 1, gap: 0, tolerance: -0.1 },
      { depth: 1, gap: -2 },
      { depth: 1, gap: -1e-9 },
      { depth: 1, gap: Number.NaN },
      { depth: 1, gap: Infinity },
      { depth: 1, gap: 1, tolerance: -0.1 },
      { depth: 1, gap: 2, tolerance: 1 },
      { depth: 2, gap: 1, tolerance: 1.5 },
      { depth: 1, gap: 1, tolerance: Number.NaN },
    ];
    for (const o of bad) {
      expect(() => contactHeatmap(o)).toThrow(expect.objectContaining({ code: "INVALID_ARGUMENT" }));
    }
    // Just inside the limits: valid, and the stops stay strictly increasing in f32.
    for (const o of [
      { depth: 1, gap: 1, tolerance: 0.999999999 },
      { depth: 1e-9, gap: 1e9, tolerance: 0 },
      { depth: 1e9, gap: 1e-9, tolerance: 0 },
      { depth: 1, gap: 0, tolerance: 0.999999999 },
      { depth: 1e-9, gap: 0, tolerance: 0 },
      { depth: 1e9, gap: 0, tolerance: 1e-9 },
    ]) {
      const stops = (contactHeatmap(o).map as { stops: readonly (readonly [number, Rgba])[] }).stops;
      expect(stops.length).toBeGreaterThanOrEqual(2);
      expect(stops[0][0]).toBe(0);
      expect(stops[stops.length - 1][0]).toBe(1);
      for (let k = 1; k < stops.length; k++) {
        expect(Math.fround(stops[k][0])).toBeGreaterThan(Math.fround(stops[k - 1][0]));
      }
    }
  });
});

// ---- Over both hosts ----

/** A closed tetrahedron scaled by `s` and offset by `d`. */
function tetra(s = 1, d: [number, number, number] = [0, 0, 0]) {
  const corners = [0, 0, 0, s, 0, 0, 0, s, 0, 0, 0, s];
  const positions = Float64Array.from(corners, (v, i) => v + d[i % 3]);
  const indices = Uint32Array.from([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]);
  return { positions, indices };
}

/** The harness Worker as a `WorkerLike`; it records every batch sent. */
class NodeWorker implements WorkerLike {
  readonly worker: Worker;
  readonly sent: Request[][] = [];
  readonly #message: ((e: { data: unknown }) => void)[] = [];
  readonly #error: ((e: unknown) => void)[] = [];

  constructor() {
    this.worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), { workerData: { serve: true } });
    this.worker.on("message", (data) => {
      for (const cb of this.#message) {
        cb({ data });
      }
    });
    this.worker.on("error", (e) => {
      for (const cb of this.#error) {
        cb(e);
      }
    });
  }

  postMessage(msg: unknown, transfer?: Transferable[]): void {
    if (Array.isArray(msg)) {
      this.sent.push(msg as Request[]);
    }
    this.worker.postMessage(msg, transfer as never);
  }

  addEventListener(type: "message" | "error", cb: (e: any) => void): void {
    (type === "message" ? this.#message : this.#error).push(cb);
  }

  terminate(): void {
    void this.worker.terminate();
  }
}

const cleanup: (() => void)[] = [];
afterEach(() => {
  for (const f of cleanup.splice(0)) {
    f();
  }
  vi.restoreAllMocks();
});

/** Commits frames until nothing reads `"computing"`, then one more frame. */
async function settle(store: SceneStore): Promise<void> {
  for (let k = 0; k < 2000; k++) {
    store.commitFrame();
    if (store.mode === "local") {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 2));
    if (store.getSnapshot().status !== "computing") {
      store.commitFrame();
      return;
    }
  }
  throw new Error("the store did not settle");
}

interface Fixture {
  store: SceneStore;
  fieldAdds(): number;
  colorings(): number;
  /** The values of a field added straight to the scene (on `a`, to `b`), then removed. */
  freshValues(a: MeshHandle, b: MeshHandle, maxDistance: number): Promise<Float32Array>;
}

async function makeLocal(): Promise<Fixture> {
  const scene = new Scene();
  const store = createSceneStore(scene);
  cleanup.push(() => {
    store.dispose();
    scene.free();
  });
  const add = vi.spyOn(Scene.prototype, "addDistanceField");
  const color = vi.spyOn(Scene.prototype, "setColoring");
  return {
    store,
    fieldAdds: () => add.mock.calls.length,
    colorings: () => color.mock.calls.length,
    freshValues: async (a, b, maxDistance) => {
      const field = scene.addDistanceField({ name: "fresh", on: a, to: b, maxDistance });
      const values = scene.fieldValues(field).slice();
      scene.removeField(field);
      return values;
    },
  };
}

async function makeWorker(): Promise<Fixture> {
  let worker!: NodeWorker;
  const scene: WorkerScene = await createWorkerScene({
    createWorker: () => {
      worker = new NodeWorker();
      return worker;
    },
  });
  const store = createSceneStore(scene);
  cleanup.push(() => {
    store.dispose();
    scene.terminate();
  });
  const reqs = () => worker.sent.flat();
  return {
    store,
    fieldAdds: () => reqs().filter((r) => r.kind === "field" && r.op === "add").length,
    colorings: () => reqs().filter((r) => r.kind === "coloring").length,
    freshValues: async (a, b, maxDistance) => {
      const field = scene.addDistanceField({ name: "fresh", on: a, to: b, maxDistance });
      const p = scene.fetchFieldValues(field);
      scene.commitFrame();
      const values = await p;
      scene.removeField(field);
      scene.commitFrame();
      return values;
    },
  };
}

/** A small tetrahedron `a` deep inside a large one `b` (its corner at (1, 1, 1) is 1 deep), and `c` beside them. */
function nested(store: SceneStore) {
  const small = tetra(1, [1, 1, 1]);
  const big = tetra(10);
  const side = tetra(1, [20, 0, 0]);
  return {
    a: store.addMesh(small.positions, small.indices),
    b: store.addMesh(big.positions, big.indices),
    c: store.addMesh(side.positions, side.indices),
  };
}

/** Mesh `mesh`'s vertex colour `v` (RGBA). */
function vertexColor(store: SceneStore, mesh: MeshHandle, v: number): number[] {
  const colors = store.scene.views(mesh).colors();
  expect(colors).not.toBeNull();
  return Array.from(colors!.subarray(4 * v, 4 * v + 4));
}

describe.each([
  ["local", makeLocal],
  ["worker", makeWorker],
] as const)("contactHeatmap in a SceneStore (%s)", (mode, make) => {
  it("tolerance_change_reuses_field", async () => {
    const f = await make();
    const { store } = f;
    expect(store.mode).toBe(mode);
    const { a, b } = nested(store);
    await settle(store);
    for (const exactDepth of [false, true]) {
      store.setDisplay(a, { coloring: { distanceTo: b, ...contactHeatmap({ depth: 0.5, gap: 1, exactDepth }) } });
      await settle(store);
      const adds = f.fieldAdds();
      const colorings = f.colorings();
      for (const tolerance of [0.1, 0.2, 0]) {
        const next = { distanceTo: b, ...contactHeatmap({ depth: 0.5, gap: 1, tolerance, exactDepth }) };
        store.setDisplay(a, { coloring: next });
        await settle(store);
        const st = store.getSnapshot().meshes.get(a)!;
        expect(st.display.coloring).toEqual(next);
        expect(st.status).toBe("ready");
      }
      expect(f.fieldAdds()).toBe(adds);
      expect(f.colorings()).toBe(colorings + 3);
    }
  });

  it("fetch_coloring_values", async () => {
    const f = await make();
    const { store } = f;
    const { a, b, c } = nested(store);
    await settle(store);
    // Not coloured: null.
    await expect(store.fetchColoringValues(a)).resolves.toBeNull();
    const heat = contactHeatmap({ depth: 0.5, gap: 1, exactDepth: true });
    store.setDisplay(a, { coloring: { distanceTo: b, ...heat } });
    await settle(store);
    const values = await store.fetchColoringValues(a);
    expect(values).toBeInstanceOf(Float32Array);
    expect(Array.from(values!)).toEqual(Array.from(await f.freshValues(a, b, Infinity)));
    // The deep corner (vertex 0) is 1 inside b.
    expect(values![0]).toBeCloseTo(-1, 5);
    // A copy: the caller owns it.
    values![0] = 123;
    expect((await store.fetchColoringValues(a))![0]).toBeCloseTo(-1, 5);
    // Colouring off: null again (the field is kept, but shows nothing).
    store.setDisplay(a, { coloring: null });
    await settle(store);
    await expect(store.fetchColoringValues(a)).resolves.toBeNull();
    // A mesh never coloured, and one not in the store.
    await expect(store.fetchColoringValues(c)).resolves.toBeNull();
    store.removeMesh(c);
    await expect(store.fetchColoringValues(c)).rejects.toMatchObject({ code: "UNKNOWN_MESH" });
  });

  it("exact_depth_colours_deep_interior", async () => {
    const f = await make();
    const { store } = f;
    const { a, b } = nested(store);
    await settle(store);
    // Vertex 0 of a is 1 deep; the depth slider is 0.2.
    store.setDisplay(a, { coloring: { distanceTo: b, ...contactHeatmap({ depth: 0.2, gap: 0.4, exactDepth: true }) } });
    await settle(store);
    expect(vertexColor(store, a, 0)).toEqual([...DEEP]);
    // The fast path bounds the field at 2 · max(depth, gap) = 0.8: the deep
    // vertex is beyond it, base grey.
    store.setDisplay(a, { coloring: { distanceTo: b, ...contactHeatmap({ depth: 0.2, gap: 0.4 }) } });
    await settle(store);
    expect(vertexColor(store, a, 0)).toEqual([...BASE]);
    // Within the bound (depth 0.3, gap 0.5: bound 1, inclusive) it is known, and clamps to deep.
    store.setDisplay(a, { coloring: { distanceTo: b, ...contactHeatmap({ depth: 0.3, gap: 0.5 }) } });
    await settle(store);
    expect(vertexColor(store, a, 0)).toEqual([...DEEP]);
  });
});
