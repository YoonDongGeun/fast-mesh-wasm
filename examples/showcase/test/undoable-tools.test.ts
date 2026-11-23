// Every tool that makes or takes a mesh is one undo step (#271): undo takes
// the new row out of the list (or brings the joined one back) and redo puts
// the same item back, with its name, material and geometry, on the sync
// `Scene` and on a Worker. The panels' buttons: undoable-tools-panel.test.tsx;
// Create, hulls and booleans: undo-add-remove.test.ts.
import { Worker } from "node:worker_threads";
import { type MeshHandle, Scene } from "fast-mesh-wasm";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "fast-mesh-wasm/worker";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AppState, type Item } from "../src/app/state/app-state";
import { runWrap } from "../src/engine/workbench-ops";
import { splitAlongPath } from "../src/features/inspect/ux/surface-path-actions";
import { closePath, extendPath, type SurfacePath, startPath, traceLeg } from "../src/features/inspect/ux/surface-paths";
import { executeDrainHoleAtHit, executePlaceStampAtHit } from "../src/features/mesh-edit/ux/print-prep-ops";
import { joinMeshes } from "../src/features/mesh-repair/ux/join";
import { createSyntheticSphereCloud } from "../src/features/points/ux/point-cloud";
import { countLiveTriangles } from "../src/features/select/ux/face-sets";
import { blobbySphere } from "../src/io/demo";

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
  scene: Scene | WorkerScene;
  /** Commits frames until `work` (or what was asked so far) is done. */
  settle: <T>(work?: Promise<T>) => Promise<T>;
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
    scene,
    settle: async (work) => {
      const value = await work;
      store.commitFrame();
      return value as never;
    },
  };
}

async function worker(): Promise<Harness> {
  const scene = await createWorkerScene({ createWorker: () => new NodeWorker() });
  const store = createSceneStore(scene);
  const app = new AppState();
  cleanups.push(app.follow(store));
  cleanups.push(() => {
    store.dispose();
    scene.terminate();
  });
  return {
    store,
    app,
    scene,
    settle: async (work) => {
      let done = false;
      const finished = Promise.resolve(work).finally(() => {
        done = true;
      });
      finished.catch(() => {});
      for (let k = 0; k < 2 || !done; k++) {
        const barrier = scene.raycast([0, 0, 1e6], [0, 0, 1]);
        store.commitFrame();
        await barrier;
        store.commitFrame();
      }
      return (await finished) as never;
    },
  };
}

type V3 = [number, number, number];

function boxGeometry(h: number, c: Readonly<V3>) {
  const [cx, cy, cz] = c;
  const [x0, x1, y0, y1, z0, z1] = [cx - h, cx + h, cy - h, cy + h, cz - h, cz + h];
  return {
    positions: Float64Array.from([x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1]),
    indices: Uint32Array.from([0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5]),
  };
}

function add(h: Harness, name: string, data: { positions: Float64Array; indices: Uint32Array }, center: V3 = [0, 0, 0]): Item {
  return h.app.addMesh(h.store, { name, ...data, center, radius: 1, volumeId: null, preset: null });
}

/** What a mesh is on screen: its live triangles' world positions, rounded, in order. */
function shape(h: Harness, mesh: MeshHandle): number[] {
  const views = h.store.scene.views(mesh);
  const positions = views.positions();
  const indices = views.indices();
  const m = h.store.scene.matrix4(mesh);
  const out: number[] = [];
  for (let k = 0; k + 2 < indices.length; k += 3) {
    if (indices[k] === 0 && indices[k + 1] === 0 && indices[k + 2] === 0) {
      continue;
    }
    for (let c = 0; c < 3; c++) {
      const v = 3 * indices[k + c];
      for (let a = 0; a < 3; a++) {
        const w = m[a] * positions[v] + m[4 + a] * positions[v + 1] + m[8 + a] * positions[v + 2] + m[12 + a];
        out.push(Math.round(w * 1e4));
      }
    }
  }
  return out;
}

/**
 * Checks the one undo step that added `added` to `before`'s items: undo takes
 * exactly its row out (and gives the other meshes the shapes of `wereShapes`),
 * redo lists the same item in its place, with the mesh it had.
 */
async function expectAddUndoes(h: Harness, before: readonly Item[], added: Item, wereShapes: readonly number[][]): Promise<void> {
  const { store, app } = h;
  const listed = [...before, added];
  expect(app.getSnapshot().items).toEqual(listed);
  expect(store.getSnapshot().history.canRedo).toBe(false);
  const disposed = vi.fn();
  added.material.addEventListener("dispose", disposed);
  const { name, color, material, triangles } = added;
  added.color = 0x123456;
  material.color.setHex(0x123456);
  const shapes = listed.map((i) => shape(h, i.mesh));
  expect(shapes[before.length].length).toBeGreaterThan(0);

  for (let k = 0; k < 2; k++) {
    store.undo();
    await h.settle();
    expect(app.getSnapshot().items).toEqual(before);
    expect(store.getSnapshot().meshes.has(added.mesh)).toBe(false);
    expect(store.getSnapshot().meshes.size).toBe(before.length);
    expect((h.scene as Scene).has(added.mesh)).toBe(false);
    expect(store.isRestorable(added.mesh)).toBe(true);
    expect(store.getSnapshot().history.canUndo).toBe(false);
    expect(disposed).toHaveBeenCalledTimes(k + 1);
    expect(before.map((i) => shape(h, i.mesh))).toEqual(wereShapes);

    store.redo();
    await h.settle();
    expect(app.getSnapshot().items).toEqual(listed);
    expect(app.getSnapshot().items[before.length]).toBe(added);
    expect(added).toMatchObject({ name, color: 0x123456, triangles });
    expect(added.material).toBe(material);
    expect(store.getSnapshot().meshes.has(added.mesh)).toBe(true);
    expect(store.getSnapshot().history).toEqual({ canUndo: true, canRedo: false });
    expect(listed.map((i) => shape(h, i.mesh))).toEqual(shapes);
  }
  expect(color).not.toBe(0x123456);
}

/** The vertex nearest to `target`, as a path pick. */
function pickNear(h: Harness, mesh: MeshHandle, target: Readonly<V3>): { slot: number; point: V3 } {
  const p = h.store.scene.views(mesh).positions();
  const m = h.store.scene.matrix4(mesh);
  let best = { slot: 0, point: [0, 0, 0] as V3, d: Infinity };
  for (let v = 0; 3 * v + 2 < p.length; v++) {
    const point = [0, 1, 2].map(
      (a) => m[a] * p[3 * v] + m[4 + a] * p[3 * v + 1] + m[8 + a] * p[3 * v + 2] + m[12 + a],
    ) as V3;
    const d = Math.hypot(point[0] - target[0], point[1] - target[1], point[2] - target[2]);
    if (d < best.d) {
      best = { slot: v, point, d };
    }
  }
  return { slot: best.slot, point: best.point };
}

/** A closed loop on `mesh` through the vertices nearest to `targets`. */
async function loopThrough(h: Harness, mesh: MeshHandle, targets: readonly V3[]): Promise<SurfacePath> {
  const picks = targets.map((t) => pickNear(h, mesh, t));
  let path = startPath(picks[0]);
  for (let k = 1; k <= picks.length; k++) {
    const to = picks[k % picks.length];
    const leg = await h.settle(traceLeg(h.store.scene as never, mesh, picks[k - 1], to, "dijkstra"));
    path = k === picks.length ? closePath(path, leg) : extendPath(path, to, leg);
  }
  return path;
}

for (const [mode, open] of [
  ["sync Scene", local],
  ["WorkerScene", worker],
] as const) {
  describe(`tools that make or take a mesh (${mode})`, () => {
    it("wrap", async () => {
      const h = await open();
      const a = add(h, "A", boxGeometry(1, [0, 0, 0]));
      await h.settle();
      const were = [shape(h, a.mesh)];
      const r = await h.settle(
        runWrap({ store: h.store, app: h.app, name: "A wrap", sources: [a], options: { gap: 0.2, detail: 16, project: false } }),
      );
      await h.settle();
      expect(h.store.getSnapshot().selection).toBe(r.item.mesh);
      expect(countLiveTriangles(h.store.scene.views(r.item.mesh).indices())).toBe(r.triangles);
      await expectAddUndoes(h, [a], r.item, were);
      // The wrap was selected, and is again when it is back.
      expect(h.store.getSnapshot().selection).toBe(r.item.mesh);
    });

    it("split along a closed surface path", async () => {
      const h = await open();
      const ball = add(h, "Ball", blobbySphere([0, 0, 0], 1, 32, 16, 0));
      h.store.setPlacement(ball.mesh, { rotation: [0, -1, 0, 1, 0, 0, 0, 0, 1], translation: [5, -2, 1], scale: [1, 1, 2] });
      await h.settle();
      await h.settle();
      const z = 2 * Math.cos(0.7);
      const ring = Math.sin(0.7);
      const loop = await loopThrough(
        h,
        ball.mesh,
        [0, 1, 2, 3].map((k): V3 => [5 + ring * Math.cos((k * Math.PI) / 2), -2 + ring * Math.sin((k * Math.PI) / 2), 1 + z]),
      );
      // The split is the only step from here on.
      (h.scene as Scene).clearHistory();
      await h.settle();
      const were = [shape(h, ball.mesh)];
      const r = await h.settle(splitAlongPath(h.store, h.app, ball, loop));
      await h.settle();
      expect(r.inside.name).toBe("Ball inside");
      expect(r.insideTriangles).toBe(countLiveTriangles(h.store.scene.views(r.inside.mesh).indices()));
      await expectAddUndoes(h, [ball], r.inside, were);
    });

    it("extracted and split faces", async () => {
      const h = await open();
      const a = add(h, "A", boxGeometry(1, [0, 0, 0]));
      await h.settle();
      const were = [shape(h, a.mesh)];
      const extracted = await h.settle(Promise.resolve(h.scene.repair.extractSelectedFaces(a.mesh, [0, 1])));
      const copy = h.app.adoptSceneMesh(h.store, extracted, "A (Extract)");
      await h.settle();
      await expectAddUndoes(h, [a], copy, were);
      h.store.undo();
      await h.settle();

      const r = await h.settle(Promise.resolve(h.scene.repair.splitSelectedFaces(a.mesh, [0, 1])));
      const part = h.app.adoptSceneMesh(h.store, r.detached, "A (Split)");
      await h.settle();
      expect(countLiveTriangles(h.store.scene.views(a.mesh).indices())).toBe(10);
      await expectAddUndoes(h, [a], part, were);
    });

    it("a drain hole's plug", async () => {
      const h = await open();
      const a = add(h, "A", boxGeometry(5, [0, 0, 0]));
      await h.settle();
      const were = [shape(h, a.mesh)];
      const r = await h.settle(
        executeDrainHoleAtHit({
          store: h.store,
          app: h.app,
          target: a.mesh,
          targetName: "A",
          hitPoint: [0, 0, 5],
          hitNormal: [0, 0, 1],
          drainParams: { diameter: 2, depth: 3, outerChamfer: 0.2, taperDeg: 2, generatePlug: true },
        }),
      );
      await h.settle();
      expect(r.plugs).toHaveLength(1);
      const plug = h.app.getSnapshot().items[1];
      expect(plug.name).toBe("A plug");
      expect(plug.mesh).toBe(r.plugs[0]);
      await expectAddUndoes(h, [a], plug, were);
    });

    it.each(["peg", "sceneMesh"] as const)("a placed stamp (%s)", async (stampSource) => {
      const h = await open();
      const a = add(h, "A", boxGeometry(5, [0, 0, 0]));
      const stamp = add(h, "Stamp", boxGeometry(0.5, [0, 0, 20]), [0, 0, 20]);
      await h.settle();
      const were = [a, stamp].map((i) => shape(h, i.mesh));
      const r = await h.settle(
        executePlaceStampAtHit({
          store: h.store,
          app: h.app,
          target: a.mesh,
          targetName: "A",
          hitPoint: [5, 0, 0],
          hitNormal: [1, 0, 0],
          placeParams: {
            ...h.app.getSnapshot().wave6.printPrep.placeParams,
            stampSource,
            stampMeshId: stampSource === "sceneMesh" ? stamp.id : null,
            mode: "keepSeparate",
            uniformScale: 1,
            embedDepth: 0.25,
            twistDeg: 0,
            conformToSurface: false,
          },
        }),
      );
      await h.settle();
      const placed = h.app.getSnapshot().items[2];
      expect(placed).toMatchObject({ mesh: r.added, name: "A stamp" });
      await expectAddUndoes(h, [a, stamp], placed, were);
    });

    it("a surface reconstructed from a point cloud", async () => {
      const h = await open();
      const a = add(h, "A", boxGeometry(1, [8, 0, 0]));
      await h.settle();
      const were = [shape(h, a.mesh)];
      const mesh = await h.settle(
        Promise.resolve(
          h.scene.inspect.reconstructPointCloud(createSyntheticSphereCloud(200, 1), {
            method: "implicitMlsMarchingCubes",
            voxelSize: 0.25,
            kNeighbors: 12,
            normals: true,
          }),
        ),
      );
      const item = h.app.adoptSceneMesh(h.store, mesh, "Reconstructed");
      await h.settle();
      await expectAddUndoes(h, [a], item, were);
    });

    it("join takes its source, and undo brings it back where it was", async () => {
      const h = await open();
      const { store, app } = h;
      const quad = (z: number, up: boolean) => ({
        positions: Float64Array.from([0, 0, z, 1, 0, z, 1, 1, z, 0, 1, z]),
        indices: Uint32Array.from(up ? [0, 1, 2, 0, 2, 3] : [0, 2, 1, 0, 3, 2]),
      });
      const a = add(h, "A", quad(0, false));
      const source = add(h, "Source", quad(1, true));
      const c = add(h, "C", boxGeometry(1, [8, 0, 0]));
      await h.settle();
      app.setCollisionMode("pair");
      expect(app.getSnapshot().collision).toEqual({ a: a.mesh, b: source.mesh });
      store.setDisplay(a.mesh, { coloring: { distanceTo: source.mesh, maxDistance: 5, range: [0, 1], map: "turbo" } });
      app.toggleCheck(a.mesh, true);
      app.toggleCheck(source.mesh, true);
      app.createGroup("Both");
      (h.scene as Scene).setCollisionRule(a.mesh, source.mesh, { mode: "rigid", contactMargin: 0.25 });
      await h.settle();
      source.material.color.setHex(0xabcdef);
      const { collision, meshGroups, group } = app.getSnapshot();
      const were = [a, source, c].map((i) => shape(h, i.mesh));
      const loopOf = async (mesh: MeshHandle) => (await h.settle(Promise.resolve(h.scene.analyze(mesh)))).loops[0].id;
      const loops = { loopA: await loopOf(a.mesh), loopB: await loopOf(source.mesh) };

      // A join that fails takes nothing.
      const stale = { ...loops, loopB: { ...loops.loopB, hash: loops.loopB.hash ^ 1n } };
      await expect(h.settle(joinMeshes(store, app, a.mesh, source, stale))).rejects.toThrow();
      await h.settle();
      expect(app.getSnapshot().items).toEqual([a, source, c]);
      expect(store.getSnapshot().history.canUndo).toBe(false);

      const r = await h.settle(joinMeshes(store, app, a.mesh, source, loops));
      expect(r).toEqual({ verticesAdded: 4, trianglesAdded: 2, bridgeTrianglesAdded: 8 });
      await h.settle();
      const joined = shape(h, a.mesh);
      for (let k = 0; k < 2; k++) {
        expect(app.getSnapshot().items).toEqual([a, c]);
        expect(store.getSnapshot().meshes.has(source.mesh)).toBe(false);
        expect(store.isRestorable(source.mesh)).toBe(true);
        expect(app.getSnapshot().collision).toEqual({ a: a.mesh, b: null });
        expect(app.getSnapshot().meshGroups[0].members).toEqual([a.mesh]);
        expect(shape(h, a.mesh)).toEqual(joined);

        store.undo();
        await h.settle();
        expect(app.getSnapshot().items).toEqual([a, source, c]);
        expect(app.getSnapshot().items[1]).toBe(source);
        expect(source.material.color.getHex()).toBe(0xabcdef);
        expect([a, source, c].map((i) => shape(h, i.mesh))).toEqual(were);
        expect(store.getSnapshot().history).toEqual({ canUndo: false, canRedo: true });
        // Its pair, groups and collision rule come back with it.
        expect(app.getSnapshot().collision).toEqual(collision);
        expect(new Set(app.getSnapshot().group.members)).toEqual(new Set(group.members));
        expect(app.getSnapshot().meshGroups).toEqual(meshGroups);
        expect((h.scene as Scene).collisionRule(a.mesh, source.mesh)).toMatchObject({ mode: "rigid", contactMargin: 0.25 });

        store.redo();
        await h.settle();
      }
      expect(app.getSnapshot().items).toEqual([a, c]);
      expect(shape(h, a.mesh)).toEqual(joined);
    });
  });
}
