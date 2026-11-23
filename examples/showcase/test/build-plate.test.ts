// The build plate on the real engine (#266): the footprint of a turned mesh
// is its own projection, Arrange puts 20 demo meshes on the plate without
// overlaps in one undo step and reports what it leaves, and Multi-Clone makes
// its copies in one undo step, on the sync `Scene` and on a Worker.
import { Worker } from "node:worker_threads";
import { Scene } from "fast-mesh-wasm";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { createWorkerScene, type WorkerLike } from "fast-mesh-wasm/worker";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AppState, type Item } from "../src/app/state/app-state";
import {
  arrangeMeshes,
  type ArrangeReport,
  arrangeTargets,
  describeArrange,
  planArrange,
  quarterTurn,
} from "../src/features/build-plate/ux/arrange";
import { readWorldBoxes } from "../src/features/build-plate/ux/footprint";
import {
  type CloneLayout,
  cloneColumns,
  cloneMesh,
  cloneNames,
  cloneOffsets,
  cloneProblem,
  DEFAULT_CLONE_LAYOUT,
} from "../src/features/build-plate/ux/multi-clone";
import { type BuildPlate, DEFAULT_BUILD_PLATE, parseBuildPlate, plateAxes, plateProblem } from "../src/features/build-plate/ux/plate";
import { plateGrid } from "../src/features/build-plate/ux/plate-grid";
import { readPlacedMesh } from "../src/features/transform/ux/placement-helpers";
import { mulMat3Vec, worldBounds, type WorldBounds } from "../src/features/transform/ux/placement-math";
import { eulerDegToMat3 } from "../src/features/transform/ux/transform-math";
import { blobbySphere, openShellWithPiece, roundedBox } from "../src/io/demo";
import type { MeshData, Vec3 } from "../src/io/mesh-data";
import { bounds } from "../src/io/mesh-data";

const cleanups: (() => void | Promise<void>)[] = [];
afterEach(async () => {
  while (cleanups.length > 0) {
    await cleanups.pop()?.();
  }
});

/** The SDK's Node worker-thread stand-in for the browser engine Worker. */
class NodeWorker implements WorkerLike {
  readonly worker = new Worker(new URL("../../../packages/fast-mesh-wasm/test/harness/node-worker.mjs", import.meta.url), {
    workerData: { serve: true },
  });

  postMessage(msg: unknown, transfer?: Transferable[]): void {
    this.worker.postMessage(msg, transfer as never);
  }

  addEventListener(type: "message" | "error", cb: (e: never) => void): void {
    this.worker.on(type, (data: unknown) => cb((type === "message" ? { data } : data) as never));
  }

  terminate(): void {
    void this.worker.terminate();
  }
}

interface Harness {
  store: SceneStore;
  app: AppState;
  settle: () => Promise<void>;
}

async function local(): Promise<Harness> {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  cleanups.push(app.follow(store));
  cleanups.push(() => {
    store.dispose();
    scene.free();
  });
  return {
    store,
    app,
    settle: async () => {
      store.commitFrame();
    },
  };
}

async function worker(): Promise<Harness> {
  const scene = await createWorkerScene({ createWorker: () => new NodeWorker() });
  const store = createSceneStore(scene);
  const app = new AppState();
  cleanups.push(app.follow(store));
  // The page's frame loop.
  const frames = setInterval(() => store.commitFrame(), 4);
  cleanups.push(() => {
    clearInterval(frames);
    store.dispose();
    scene.terminate();
  });
  return {
    store,
    app,
    settle: async () => {
      for (let k = 0; k < 2; k++) {
        const barrier = scene.raycast([0, 0, 1e6], [0, 0, 1]);
        store.commitFrame();
        await barrier;
        store.commitFrame();
      }
    },
  };
}

const HOSTS = [
  ["Scene", local],
  ["WorkerScene", worker],
] as const;

function addData(h: Harness, name: string, data: MeshData): Item {
  const { center, radius } = bounds(data.positions);
  return h.app.addMesh(h.store, { name, ...data, center, radius, volumeId: null, preset: null });
}

function boxData(half: Vec3, c: Vec3): MeshData {
  const positions: number[] = [];
  for (const z of [-1, 1]) {
    for (const [x, y] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      positions.push(c[0] + x * half[0], c[1] + y * half[1], c[2] + z * half[2]);
    }
  }
  return {
    positions: Float64Array.from(positions),
    indices: Uint32Array.from([0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5]),
  };
}

function world(h: Harness, item: Item): WorldBounds {
  return worldBounds(readPlacedMesh(h.store.scene, item.mesh))!;
}

/** The boxes rest on the plate, inside it and `spacing` apart. */
function expectOnPlate(boxes: readonly WorldBounds[], plate: BuildPlate): void {
  const { u, v, up } = plateAxes(plate.up);
  const size = Math.max(plate.width, plate.depth);
  const eps = size * 1.1e-6;
  for (const b of boxes) {
    expect(Math.abs(b.min[up])).toBeLessThan(1e-9 * size);
    expect(b.min[u]).toBeGreaterThanOrEqual(plate.origin[0] - eps);
    expect(b.min[v]).toBeGreaterThanOrEqual(plate.origin[1] - eps);
    expect(b.max[u]).toBeLessThanOrEqual(plate.origin[0] + plate.width + eps);
    expect(b.max[v]).toBeLessThanOrEqual(plate.origin[1] + plate.depth + eps);
  }
  let least = Infinity;
  boxes.forEach((a, i) => {
    for (const b of boxes.slice(0, i)) {
      least = Math.min(
        least,
        Math.max(a.min[u] - b.max[u], b.min[u] - a.max[u], a.min[v] - b.max[v], b.min[v] - a.max[v]),
      );
    }
  });
  expect(least).toBeGreaterThanOrEqual(plate.spacing - eps);
}

/** Twenty meshes from the three demo shapes, in all sorts of placements. */
function addTwentyDemoMeshes(h: Harness): Item[] {
  const shapes: [string, MeshData][] = [
    ["A", blobbySphere([-0.4, 0.2, 0.1], 1, 24, 12)],
    ["B", roundedBox([0.9, 0.3, -0.2], [0.9, 0.45, 0.6], 0.15, 6)],
    ["C", openShellWithPiece([3, 0, 0], 0.8, 24, 12)],
  ];
  return Array.from({ length: 20 }, (_, i) => {
    const [name, data] = shapes[i % 3];
    const item = addData(h, `${name} ${i}`, { positions: data.positions.slice(), indices: data.indices.slice() });
    h.store.setPlacement(item.mesh, {
      rotation: eulerDegToMat3([37 * i, 61 * i, 13 * i]),
      translation: [Math.sin(i) * 2, Math.cos(3 * i), 0.3 * i - 2],
      scale: i % 4 === 0 ? [1 + i / 10, 1, 0.5] : [1, 1, 1],
    });
    return item;
  });
}

describe("footprints", () => {
  it("are the projection of the geometry, not the box around the turned local box", async () => {
    const h = await local();
    // A sphere turned by 45° projects as wide as before; its turned local box would be 1.41 times that.
    const item = addData(h, "Ball", roundedBox([5, -3, 2], [1, 1, 1], 0.999, 20));
    const before = (await readWorldBoxes(h.store.scene, [item.mesh])).get(item.mesh)!;
    h.store.setPlacement(item.mesh, { rotation: eulerDegToMat3([45, 0, 45]), translation: [0, 0, 0] });
    const after = (await readWorldBoxes(h.store.scene, [item.mesh])).get(item.mesh)!;
    for (const k of [0, 1, 2]) {
      expect(before.max[k] - before.min[k]).toBeCloseTo(2, 2);
      expect(after.max[k] - after.min[k]).toBeLessThan(2.02);
    }
    expect(after).toEqual(world(h, item));
  });

  it("skip dead triangles and unused vertices, and are null for a mesh without any", async () => {
    const h = await local();
    const item = addData(h, "Box", boxData([1, 1, 1], [0, 0, 0]));
    // Only the top face is left.
    await h.store.edit(item.mesh, [{ op: "removeTriangles", triangles: [0, 1, 4, 5, 6, 7, 8, 9, 10, 11] }]);
    expect((await readWorldBoxes(h.store.scene, [item.mesh])).get(item.mesh)).toEqual({ min: [-1, -1, 1], max: [1, 1, 1] });
    await h.store.edit(item.mesh, [{ op: "removeTriangles", triangles: [2, 3] }]);
    expect((await readWorldBoxes(h.store.scene, [item.mesh])).get(item.mesh)).toBeNull();
  });

  it("yield while they are read, and read a mesh that moves or changes meanwhile again", async () => {
    const h = await local();
    const [a, b] = ["A", "B"].map((n, i) => addData(h, n, boxData([1, 2, 3], [5 * i, 0, 0])));
    h.store.commitFrame();
    // Every slice is at its end.
    let now = 0;
    const clock = vi.spyOn(performance, "now").mockImplementation(() => (now += 100));
    cleanups.push(() => clock.mockRestore());
    const at = (x: number) => ({ rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1] as never, translation: [x, 0, 0] as Vec3 });
    let pauses = 0;
    const boxes = await readWorldBoxes(h.store.scene, [a.mesh, b.mesh], async () => {
      pauses++;
      // While B is read: A moves, B keeps its +X face only.
      if (pauses === 3) {
        h.store.setPlacement(a.mesh, at(50));
        await h.store.edit(b.mesh, [{ op: "removeTriangles", triangles: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] }]);
        h.store.commitFrame();
      }
    });
    expect(pauses).toBeGreaterThanOrEqual(8);
    expect(boxes.get(a.mesh)).toEqual({ min: [49, -2, -3], max: [51, 2, 3] });
    // Only B's +X face is left.
    expect(boxes.get(b.mesh)).toEqual({ min: [6, -2, -3], max: [6, 2, 3] });
    // One that never rests is given up on.
    let x = 0;
    await expect(
      readWorldBoxes(h.store.scene, [a.mesh], async () => h.store.setPlacement(a.mesh, at(x++))),
    ).rejects.toThrow(/changed/);
  });
});

describe("planArrange", () => {
  it("turns a part by 90° about the up axis, and only that", () => {
    const part = {
      box: { min: [10, 20, 30] as Vec3, max: [14, 60, 35] as Vec3 },
      placement: { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [1, 2, 3] as Vec3, scale: [2, 1, 1] as Vec3 },
    };
    const z: BuildPlate = { ...DEFAULT_BUILD_PLATE, width: 50, depth: 10, origin: [-5, 7] };
    const plan = planArrange([part], z);
    expect(plan.placed).toHaveLength(1);
    expect(plan.placed[0].turned).toBe(true);
    const { rotation, translation, scale } = plan.placed[0].placement;
    expect(rotation).toEqual([0, -1, 0, 1, 0, 0, 0, 0, 1]);
    expect(scale).toEqual([2, 1, 1]);
    // The box, turned: 40 x 4, centred on the 50 x 10 plate, its lowest point on it.
    const t = translation as Vec3;
    const corner = (p: Vec3): Vec3 => {
      // The old placement undone (a pure move), the new one applied.
      const local: Vec3 = [p[0] - 1, p[1] - 2, p[2] - 3];
      const w = mulMat3Vec(rotation as never, local);
      return [w[0] + t[0], w[1] + t[1], w[2] + t[2]];
    };
    const a = corner(part.box.min);
    const b = corner(part.box.max);
    expect([Math.min(a[0], b[0]), Math.max(a[0], b[0])]).toEqual([0, 40].map((x) => expect.closeTo(x, 3)));
    expect([Math.min(a[1], b[1]), Math.max(a[1], b[1])]).toEqual([10, 14].map((y) => expect.closeTo(y, 3)));
    expect(Math.min(a[2], b[2])).toBeCloseTo(0, 9);
    // With Y up the floor is X and Z.
    const y = planArrange([{ ...part, box: { min: [10, 30, 20], max: [14, 35, 60] } }], { ...z, up: "y" });
    expect(y.placed[0].placement.rotation).toEqual([0, 0, -1, 0, 1, 0, 1, 0, 0]);
    for (const up of ["z", "y"] as const) {
      const m = quarterTurn(up);
      const { u, v, up: k } = plateAxes(up);
      const axis = (i: number) => [0, 1, 2].map((j) => (j === i ? 1 : 0)) as Vec3;
      expect(mulMat3Vec(m, axis(u)).map((c) => c + 0)).toEqual(axis(v));
      expect(mulMat3Vec(m, axis(k)).map((c) => c + 0)).toEqual(axis(k));
    }
  });
});

describe.each(HOSTS)("Arrange on %s", (_name, host) => {
  it.each(["z", "y"] as const)("puts 20 demo meshes on the plate without overlaps, one undo step (%s up)", async (up) => {
    const h = await host();
    const items = addTwentyDemoMeshes(h);
    await h.settle();
    const placements = items.map((i) => h.store.scene.placement(i.mesh));
    const plate: BuildPlate = { ...DEFAULT_BUILD_PLATE, width: 30, depth: 20, origin: [-7, 4], spacing: 0.25, up };
    const report = await arrangeMeshes({ store: h.store, app: h.app, items, plate });
    await h.settle();
    expect(report).toMatchObject({ placed: 20, moved: 20, oversized: [], noRoom: [], locked: [], empty: [] });
    expectOnPlate(items.map((i) => world(h, i)), plate);
    const arranged = items.map((i) => h.store.scene.placement(i.mesh));
    // Scales stay, and only turns about the up axis are made.
    const k = plateAxes(up).up;
    arranged.forEach((p, i) => {
      expect(p.scale).toEqual(placements[i].scale);
      for (const j of [0, 1, 2]) {
        expect(p.rotation[3 * k + j]).toBeCloseTo(placements[i].rotation[3 * k + j], 12);
      }
    });

    // Arranged again, nothing moves and nothing is recorded.
    const again = await arrangeMeshes({ store: h.store, app: h.app, items, plate });
    expect(again).toMatchObject({ placed: 20, moved: 0, turned: 0 });
    expect(describeArrange(again).text).toBe("20 meshes already in place");

    h.store.undo();
    await h.settle();
    expect(items.map((i) => h.store.scene.placement(i.mesh))).toEqual(placements);
    h.store.redo();
    await h.settle();
    expect(items.map((i) => h.store.scene.placement(i.mesh))).toEqual(arranged);
  });

  it("leaves locked, oversized and empty meshes where they are, keeps clear of the others and reports them", async () => {
    const h = await host();
    const fits = [0, 1, 2].map((i) => addData(h, `Fits ${i}`, boxData([4, 4, 4], [50 * i, 80, 9])));
    const locked = addData(h, "Locked", boxData([5, 5, 5], [0, 0, 20]));
    const big = addData(h, "Big", boxData([30, 2, 2], [0, 70, 0]));
    const empty = addData(h, "Empty", boxData([1, 1, 1], [0, 0, 0]));
    const bystander = addData(h, "Bystander", boxData([3, 3, 3], [-11, -11, 0]));
    const hidden = addData(h, "Hidden", boxData([20, 20, 3], [0, 0, 0]));
    await h.store.edit(empty.mesh, [{ op: "removeTriangles", triangles: Array.from({ length: 12 }, (_, t) => t) }]);
    h.app.setMeshLocked(locked.mesh, true);
    h.store.setDisplay(hidden.mesh, { visible: false });
    await h.settle();
    const placed = [...fits, locked, big, empty].map((i) => h.app.getSnapshot().items.find((x) => x.mesh === i.mesh)!);
    const plate: BuildPlate = { ...DEFAULT_BUILD_PLATE, width: 40, depth: 40, origin: [-20, -20], spacing: 1 };
    const before = new Map([locked, big, empty, bystander, hidden].map((i) => [i, h.store.scene.placement(i.mesh)]));
    const report = await arrangeMeshes({ store: h.store, app: h.app, items: placed, plate });
    await h.settle();
    expect(report).toEqual<ArrangeReport>({
      placed: 3,
      turned: 0,
      moved: 3,
      oversized: ["Big"],
      noRoom: [],
      locked: ["Locked"],
      empty: ["Empty"],
    });
    expect(describeArrange(report)).toEqual({
      text: "Arranged 3 of 6 meshes",
      left: ["Too big for the plate: Big", "Locked: Locked", "No triangles: Empty"],
    });
    for (const [item, placement] of before) {
      expect(h.store.scene.placement(item.mesh)).toEqual(placement);
    }
    expectOnPlate(fits.map((i) => world(h, i)), plate);
    const clear = (a: WorldBounds, b: WorldBounds) =>
      Math.max(a.min[0] - b.max[0], b.min[0] - a.max[0], a.min[1] - b.max[1], b.min[1] - a.max[1]);
    for (const i of fits) {
      expect(clear(world(h, i), world(h, locked))).toBeGreaterThanOrEqual(1 - 1e-4);
      expect(clear(world(h, i), world(h, bystander))).toBeGreaterThanOrEqual(1 - 1e-4);
    }
  });

  it("reports the meshes the plate has no room for", async () => {
    const h = await host();
    const items = Array.from({ length: 5 }, (_, i) => addData(h, `Part ${i}`, boxData([4.5, 4.5, 1], [0, 0, 0])));
    await h.settle();
    const plate: BuildPlate = { ...DEFAULT_BUILD_PLATE, width: 20, depth: 20, origin: [0, 0], spacing: 2 };
    const report = await arrangeMeshes({ store: h.store, app: h.app, items, plate });
    await h.settle();
    expect(report).toMatchObject({ placed: 4, noRoom: ["Part 4"] });
    expectOnPlate(items.slice(0, 4).map((i) => world(h, i)), plate);
    h.store.undo();
    await h.settle();
    expect(world(h, items[0]).min[2]).toBe(-1);
    await expect(
      arrangeMeshes({ store: h.store, app: h.app, items, plate: { ...plate, width: 0 } }),
    ).rejects.toThrow(/above zero/);
  });
});

describe.each(HOSTS)("Multi-clone on %s", (_name, host) => {
  it("adds N copies in one undo step, keeping geometry, origin and placement, and selects them", async () => {
    const h = await host();
    const item = addData(h, "Part", boxData([1, 2, 3], [10, 20, 30]));
    addData(h, "Part copy 2", boxData([1, 1, 1], [0, 0, 0]));
    h.store.setPlacement(item.mesh, { rotation: eulerDegToMat3([0, 0, 90]), translation: [5, 0, 1], scale: [1, 2, 1] });
    await h.settle();
    const source = world(h, item);
    const layout: CloneLayout = { ...DEFAULT_CLONE_LAYOUT, copies: 5, columns: 3, spacing: 0.5 };
    const copies = await cloneMesh({ store: h.store, app: h.app, item, layout, up: "z" });
    await h.settle();
    expect(copies.map((c) => c.name)).toEqual(["Part copy", "Part copy 3", "Part copy 4", "Part copy 5", "Part copy 6"]);
    expect(h.app.getSnapshot().items).toHaveLength(7);
    expect(h.store.getSnapshot().selection).toBe(item.mesh);
    expect(h.app.getSnapshot().multi).toEqual([item.mesh, ...copies.map((c) => c.mesh)]);
    expect(h.store.scene.placement(item.mesh).translation).toEqual([5, 0, 1]);

    const size = [0, 1, 2].map((k) => source.max[k] - source.min[k]) as Vec3;
    expect(size.map((s) => Math.round(s * 1e6) / 1e6)).toEqual([8, 2, 6]);
    copies.forEach((copy, i) => {
      const k = i + 1;
      const at = [(k % 3) * 8.5, Math.floor(k / 3) * 2.5, 0];
      const b = world(h, copy);
      for (const a of [0, 1, 2]) {
        expect(b.min[a]).toBeCloseTo(source.min[a] + at[a], 9);
        expect(b.max[a]).toBeCloseTo(source.max[a] + at[a], 9);
      }
      // The same local geometry around the same origin.
      expect(Array.from(h.store.scene.views(copy.mesh).positions())).toEqual(
        Array.from(h.store.scene.views(item.mesh).positions()),
      );
      expect(Array.from(h.store.scene.views(copy.mesh).indices())).toEqual(
        Array.from(h.store.scene.views(item.mesh).indices()),
      );
      expect(copy.center).toEqual([10, 20, 30]);
      expect(h.store.scene.placement(copy.mesh).scale).toEqual([1, 2, 1]);
    });

    h.store.undo();
    await h.settle();
    expect(h.app.getSnapshot().items.map((i) => i.name)).toEqual(["Part", "Part copy 2"]);
    expect(copies.every((c) => !h.store.getSnapshot().meshes.has(c.mesh))).toBe(true);
    expect(h.store.scene.placement(item.mesh).translation).toEqual([5, 0, 1]);
    h.store.redo();
    await h.settle();
    expect(h.app.getSnapshot().items).toHaveLength(7);
    expect(world(h, copies[4]).min[1]).toBeCloseTo(source.min[1] + 2.5, 9);
  });

  it("clones a line of copies, which Arrange then packs", async () => {
    const h = await host();
    const item = addData(h, "Part", roundedBox([1, 2, 3], [1.5, 0.5, 0.25], 0.2, 8));
    await h.settle();
    const layout: CloneLayout = { pattern: "linear", copies: 19, columns: null, axis: "-y", spacing: 0 };
    const copies = await cloneMesh({ store: h.store, app: h.app, item, layout, up: "z" });
    // No settling: Arrange waits for what is still being added.
    const items = [item, ...copies];
    const plate: BuildPlate = { ...DEFAULT_BUILD_PLATE, width: 16, depth: 12, origin: [0, 0], spacing: 0.5 };
    const report = await arrangeMeshes({ store: h.store, app: h.app, items, plate });
    await h.settle();
    expect(report).toMatchObject({ placed: 20, empty: [] });
    expectOnPlate(items.map((i) => world(h, i)), plate);
    h.store.undo();
    await h.settle();
    copies.forEach((copy, i) => {
      expect(world(h, copy).max[1]).toBeCloseTo(2.5 - (i + 1), 5);
    });
    expect(h.app.getSnapshot().items).toHaveLength(20);
    h.store.undo();
    await h.settle();
    expect(h.app.getSnapshot().items).toEqual([item]);
  });
});

describe("multi-clone layout", () => {
  it("fills rows along X, then the floor's other axis, or runs along any axis", () => {
    const grid: CloneLayout = { ...DEFAULT_CLONE_LAYOUT, copies: 3, columns: 2, spacing: 1 };
    expect(cloneOffsets([2, 3, 4], grid, "z")).toEqual([
      [3, 0, 0],
      [0, 4, 0],
      [3, 4, 0],
    ]);
    expect(cloneOffsets([2, 3, 4], grid, "y")).toEqual([
      [3, 0, 0],
      [0, 0, 5],
      [3, 0, 5],
    ]);
    expect(cloneOffsets([2, 3, 4], { ...grid, pattern: "linear", axis: "-z", copies: 2 }, "z")).toEqual([
      [0, 0, -5],
      [0, 0, -10],
    ]);
    // Near square by default.
    expect([1, 3, 4, 5, 8, 9].map((copies) => cloneColumns({ copies, columns: null }))).toEqual([2, 2, 3, 3, 3, 4]);
  });

  it("names copies uniquely and refuses bad values", () => {
    expect(cloneNames("A", 3, ["A", "A copy 2"])).toEqual(["A copy", "A copy 3", "A copy 4"]);
    const ok = DEFAULT_CLONE_LAYOUT;
    expect(cloneProblem(ok, 1000)).toBeNull();
    expect(cloneProblem({ ...ok, copies: 0 }, 10)).toMatch(/Copies/);
    expect(cloneProblem({ ...ok, copies: 2.5 }, 10)).toMatch(/Copies/);
    expect(cloneProblem({ ...ok, copies: 1001 }, 10)).toMatch(/1000/);
    expect(cloneProblem({ ...ok, columns: 0 }, 10)).toMatch(/Columns/);
    expect(cloneProblem({ ...ok, pattern: "linear", columns: 0 }, 10)).toBeNull();
    expect(cloneProblem({ ...ok, spacing: -1 }, 10)).toMatch(/spacing/);
    expect(cloneProblem({ ...ok, copies: 30 }, 1_000_000)).toMatch(/at most/);
  });
});

describe("the plate", () => {
  it("is read tolerantly from storage", () => {
    expect(parseBuildPlate(undefined)).toEqual(DEFAULT_BUILD_PLATE);
    expect(parseBuildPlate({ width: 0, depth: "x", origin: [1], up: "x", spacing: -2, turn: 0 })).toEqual(DEFAULT_BUILD_PLATE);
    const plate: BuildPlate = {
      visible: true,
      width: 300,
      depth: 120.5,
      origin: [0, -20],
      up: "y",
      spacing: 0,
      turn: false,
      keepClear: false,
    };
    expect(parseBuildPlate(JSON.parse(JSON.stringify(plate)))).toEqual(plate);
    expect(plateProblem(plate)).toBeNull();
    expect(plateProblem({ ...plate, depth: 0 })).toMatch(/above zero/);
    expect(plateProblem({ ...plate, spacing: Number.NaN })).toMatch(/spacing/);
  });

  it("is a grid from its corner, on the floor of either up axis", () => {
    const plate = { width: 200, depth: 100, origin: [-100, 10] as [number, number], up: "z" as const };
    const z = plateGrid(plate);
    expect(z.step).toBe(10);
    // 19 lines across the width, 9 across the depth.
    expect(z.lines).toHaveLength((19 + 9) * 6);
    expect(Array.from(z.lines.slice(0, 6))).toEqual([-90, 10, 0, -90, 110, 0]);
    expect(Array.from(z.corners)).toEqual([-100, 10, 0, 100, 10, 0, 100, 110, 0, -100, 110, 0]);
    const y = plateGrid({ ...plate, up: "y" });
    expect(Array.from(y.lines.slice(0, 6))).toEqual([-90, 0, 10, -90, 0, 110]);
    // Counter-clockwise seen from above, so the face looks up.
    const normal = (c: Float32Array) => {
      const e = (i: number) => [0, 1, 2].map((k) => c[3 * i + k] - c[k]);
      const [a, b] = [e(1), e(2)];
      return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]].map((n) => Math.sign(n));
    };
    expect(normal(z.corners).map((n) => n + 0)).toEqual([0, 0, 1]);
    expect(normal(y.corners).map((n) => n + 0)).toEqual([0, 1, 0]);
  });

  it("arranges the selection, or every visible mesh, in list order", async () => {
    const h = await local();
    const items = ["A", "B", "C"].map((n) => addData(h, n, boxData([1, 1, 1], [0, 0, 0])));
    const [a, b, c] = items;
    const visible = (m: unknown) => m !== b.mesh;
    expect(arrangeTargets("selected", items, [], null, visible)).toEqual([]);
    expect(arrangeTargets("selected", items, [], b.mesh, visible)).toEqual([b]);
    expect(arrangeTargets("selected", items, [c.mesh, a.mesh], c.mesh, visible)).toEqual([a, c]);
    // A multi-selection that no longer starts at the primary is stale.
    expect(arrangeTargets("selected", items, [c.mesh, a.mesh], a.mesh, visible)).toEqual([a]);
    expect(arrangeTargets("all", items, [], null, visible)).toEqual([a, c]);
  });
});
