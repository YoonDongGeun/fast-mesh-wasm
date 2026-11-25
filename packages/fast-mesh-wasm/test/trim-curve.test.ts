// trimByCurve's report and component rules (#273), on Scene and WorkerScene.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it } from "vitest";
import { generatePrimitive, type MeshHandle, Scene, type TrimReport, type Vec3 } from "../src/node.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";

type Geometry = { positions: Float64Array; indices: Uint32Array };

/** A primitive, moved by `at`. */
function primitive(shape: Parameters<typeof generatePrimitive>[0], at: Vec3 = [0, 0, 0]): Geometry {
  const p = generatePrimitive(shape);
  return { positions: Float64Array.from(p.positions, (x, i) => x + at[i % 3]), indices: p.indices };
}

const ball = (at: Vec3, radius: number) => primitive({ kind: "uvSphere", radius, segments: 32, rings: 16 }, at);
const torus = () => primitive({ kind: "torus", majorRadius: 1, minorRadius: 0.35, segments: 48, tubeSegments: 24 });

/** Several meshes' triangles in one, sharing no vertex. */
function merged(...parts: Geometry[]): Geometry {
  const positions: number[] = [];
  const indices: number[] = [];
  for (const part of parts) {
    const at = positions.length / 3;
    positions.push(...part.positions);
    indices.push(...Array.from(part.indices, (v) => v + at));
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

/** A loop around the top of the ball at `c`. */
const cap = (c: Vec3, radius: number): Vec3[] =>
  Array.from({ length: 8 }, (_, k) => {
    const t = (k * Math.PI) / 4 + 0.1;
    const [s, h] = [radius * Math.sin(0.7), radius * Math.cos(0.7)];
    return [c[0] + s * Math.cos(t), c[1] + h, c[2] + s * Math.sin(t)];
  });

/** A loop once around the torus's tube, off its mesh lines: it separates nothing. */
const aroundTube: Vec3[] = [0, 0.3, 0, -0.3, 0.2].map((a, k) => {
  const t = (k * 2 * Math.PI) / 5;
  const r = 1 + 0.35 * Math.cos(t);
  return [r * Math.cos(a), 0.35 * Math.sin(t), r * Math.sin(a)];
});

class NodeWorker implements WorkerLike {
  readonly worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), { workerData: { serve: true } });

  postMessage(msg: unknown, transfer?: Transferable[]): void {
    this.worker.postMessage(msg, transfer as never);
  }

  addEventListener(type: "message" | "error", listener: (e: never) => void): void {
    this.worker.on(type, (data: unknown) => listener((type === "message" ? { data } : data) as never));
  }

  terminate(): void {
    void this.worker.terminate();
  }
}

const cleanups: (() => void)[] = [];
afterEach(() => {
  for (const fn of cleanups.splice(0)) fn();
});

async function open(kind: "Scene" | "WorkerScene"): Promise<Scene | WorkerScene> {
  if (kind === "Scene") {
    const scene = new Scene();
    cleanups.push(() => scene.free());
    return scene;
  }
  const scene = await createWorkerScene({ createWorker: () => new NodeWorker() });
  cleanups.push(() => scene.terminate());
  return scene;
}

async function trim(
  scene: Scene | WorkerScene,
  mesh: MeshHandle,
  curve: Vec3[],
  deleteInside = true,
): Promise<TrimReport> {
  return scene.boolean.trimByCurve(mesh, curve, { closed: true, deleteInside });
}

/** Sends the frame and waits for every result so far. */
async function settle(scene: Scene | WorkerScene): Promise<void> {
  if (scene instanceof Scene) {
    scene.commitFrame();
    return;
  }
  const barrier = scene.raycast([0, 0, 1e6], [0, 0, 1]);
  scene.commitFrame();
  await barrier;
  scene.commitFrame();
}

/** The live triangles' corner positions, by slot. */
function triangles(scene: Scene | WorkerScene, mesh: MeshHandle): Map<number, string> {
  const views = scene.views(mesh);
  const p = views.positions();
  const ix = views.indices();
  const out = new Map<number, string>();
  for (let t = 0; 3 * t < ix.length; t++) {
    const corners = [ix[3 * t], ix[3 * t + 1], ix[3 * t + 2]];
    if (corners.some((c) => c !== 0)) {
      out.set(t, corners.map((c) => `${c}:${p[3 * c]},${p[3 * c + 1]},${p[3 * c + 2]}`).join(" "));
    }
  }
  return out;
}

describe.each(["Scene", "WorkerScene"] as const)("trimByCurve on a %s (#273)", (kind) => {
  it("reports no cut for a loop that separates nothing, and imprints it", async () => {
    const scene = await open(kind);
    for (const data of [torus(), merged(ball([4, 0, 0], 0.3), torus())]) {
      for (const inside of [true, false]) {
        const mesh = scene.addMesh(data.positions.slice(), data.indices.slice());
        await settle(scene);
        const r = await trim(scene, mesh, aroundTube, inside);
        expect(r).toMatchObject({ kept: mesh, cutTriangles: 0, removedArea: 0 });
        expect(r.imprintTriangles).toBeGreaterThan(40);
        expect(r.trianglesRemoved).toBe(r.imprintTriangles);
        await settle(scene);
        expect(triangles(scene, mesh).size).toBe(data.indices.length / 3 + r.trianglesAdded - r.trianglesRemoved);
      }
    }
  });

  it("cuts only the component the loop is on, inside and outside", async () => {
    const scene = await open(kind);
    const data = merged(ball([0, 0, 0], 1), ball([3, 0, 0], 0.3));
    const half = data.indices.length / 6;
    const cases = [
      { curve: cap([0, 0, 0], 1), own: (t: number) => t < half, radius: 1 },
      { curve: cap([3, 0, 0], 0.3), own: (t: number) => t >= half, radius: 0.3 },
    ];
    for (const { curve, own, radius } of cases) {
      const removed: number[] = [];
      for (const inside of [true, false]) {
        const mesh = scene.addMesh(data.positions.slice(), data.indices.slice());
        await settle(scene);
        const before = triangles(scene, mesh);
        expect(before.size).toBe(2 * half);
        const r = await trim(scene, mesh, curve, inside);
        await settle(scene);
        expect(r.cutTriangles).toBeGreaterThan(0);
        expect(r.trianglesRemoved).toBe(r.imprintTriangles + r.cutTriangles);
        removed.push(r.removedArea);
        const after = triangles(scene, mesh);
        for (const [t, corners] of before) {
          if (!own(t)) {
            expect(after.get(t)).toBe(corners);
          }
        }
      }
      // The cap, and the rest of the ball (a little less than a sphere).
      const sphere = 4 * Math.PI * radius * radius;
      expect(removed[0]).toBeGreaterThan(0.08 * sphere);
      expect(removed[0]).toBeLessThan(0.15 * sphere);
      expect(removed[0] + removed[1]).toBeGreaterThan(0.97 * sphere);
      expect(removed[0] + removed[1]).toBeLessThan(sphere);
    }
  });

  it("reports the removed area in world units", async () => {
    const scene = await open(kind);
    const sheet = primitive({ kind: "plane", size: 2, segments: 10 });
    const mesh = scene.addMesh(sheet.positions, sheet.indices);
    scene.setPlacement(mesh, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0, 5, 0], scale: [3, 1, 2] });
    await settle(scene);
    const curve: Vec3[] = [
      [-1.5, 5, -1],
      [1.5, 5, -1],
      [1.5, 5, 1],
      [-1.5, 5, 1],
    ];
    const r = await trim(scene, mesh, curve);
    expect(r.removedArea).toBeCloseTo(6, 4);
  });

  it("refuses a curve with points on several components, and changes nothing", async () => {
    const scene = await open(kind);
    const data = merged(ball([0, 0, 0], 1), ball([3, 0, 0], 0.3));
    const mesh = scene.addMesh(data.positions, data.indices);
    await settle(scene);
    const before = triangles(scene, mesh);
    const curve: Vec3[] = [
      [0.8, 0.6, 0],
      [2.8, 0.2, 0],
      [0.8, 0, 0.6],
    ];
    for (const closed of [true, false]) {
      await expect((async () => scene.boolean.trimByCurve(mesh, curve, { closed }))()).rejects.toMatchObject({
        code: "INVALID_ARGUMENT",
        message: expect.stringContaining("several components"),
      });
    }
    await settle(scene);
    expect(triangles(scene, mesh)).toEqual(before);
    expect(typeof scene.canUndo === "function" ? scene.canUndo() : scene.canUndo).toBe(false);
  });
});
