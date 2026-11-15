// Wave 5 SDK & Worker RPC tests for Issues #210 (continuity hole fill & partial bridge)
// and #212 (dynamic topology sculpt & undo coalescing).
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it } from "vitest";
import {
  type DyntopoDabReport,
  type DyntopoOptions,
  type FillContinuity,
  type MeshHandle,
  Scene,
  type Vec3,
} from "../src/node.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";

function openCylinderRings(
  slices = 12,
  radius = 1,
  zBottom = -0.5,
  zTop = 0.5,
) {
  const positions: number[] = [];
  for (let ring = 0; ring < 2; ring++) {
    const z = ring === 0 ? zBottom : zTop;
    for (let j = 0; j < slices; j++) {
      const theta = (2 * Math.PI * j) / slices;
      positions.push(radius * Math.cos(theta), radius * Math.sin(theta), z);
    }
  }
  const indices: number[] = [];
  for (let j = 0; j < slices; j++) {
    const a = j;
    const b = (j + 1) % slices;
    const c = slices + ((j + 1) % slices);
    const d = slices + j;
    indices.push(a, b, c, a, c, d);
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

function twoSeparatedRings(slices = 10) {
  const positions: number[] = [];
  const indices: number[] = [];
  // Band A at z in [-1.2, -0.6]
  for (let r = 0; r < 2; r++) {
    const z = r === 0 ? -1.2 : -0.6;
    for (let j = 0; j < slices; j++) {
      const a = (2 * Math.PI * j) / slices;
      positions.push(Math.cos(a), Math.sin(a), z);
    }
  }
  for (let j = 0; j < slices; j++) {
    const a = j;
    const b = (j + 1) % slices;
    const c = slices + b;
    const d = slices + a;
    indices.push(a, b, c, a, c, d);
  }
  // Band B at z in [0.6, 1.2]
  const off = slices * 2;
  for (let r = 0; r < 2; r++) {
    const z = r === 0 ? 0.6 : 1.2;
    for (let j = 0; j < slices; j++) {
      const a = (2 * Math.PI * j) / slices;
      positions.push(Math.cos(a), Math.sin(a), z);
    }
  }
  for (let j = 0; j < slices; j++) {
    const a = off + j;
    const b = off + ((j + 1) % slices);
    const c = off + slices + ((j + 1) % slices);
    const d = off + slices + j;
    indices.push(a, b, c, a, c, d);
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

function gridMesh(nx = 8, ny = 8, size = 2) {
  const positions: number[] = [];
  for (let y = 0; y < ny; y++) {
    for (let x = 0; x < nx; x++) {
      positions.push(
        (x / (nx - 1) - 0.5) * size,
        (y / (ny - 1) - 0.5) * size,
        0,
      );
    }
  }
  const indices: number[] = [];
  for (let y = 0; y < ny - 1; y++) {
    for (let x = 0; x < nx - 1; x++) {
      const a = y * nx + x;
      const b = a + 1;
      const c = a + nx + 1;
      const d = a + nx;
      indices.push(a, b, c, a, c, d);
    }
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

class NodeWorker implements WorkerLike {
  readonly worker: Worker;
  readonly #message: ((e: { data: unknown }) => void)[] = [];
  readonly #error: ((e: unknown) => void)[] = [];

  constructor() {
    this.worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), {
      workerData: { serve: true },
    });
    this.worker.on("message", (data) => {
      for (const l of [...this.#message]) l({ data });
    });
    this.worker.on("error", (err) => {
      const ev = { message: err instanceof Error ? err.message : String(err) };
      for (const l of [...this.#error]) l(ev);
    });
  }

  postMessage(msg: unknown, transfer?: Transferable[]): void {
    this.worker.postMessage(msg, transfer as never);
  }

  addEventListener(type: "message" | "error", listener: (e: never) => void): void {
    (type === "message" ? this.#message : this.#error).push(listener as never);
  }

  removeEventListener(type: "message" | "error", listener: (e: never) => void): void {
    const list = type === "message" ? this.#message : this.#error;
    const i = list.indexOf(listener as never);
    if (i >= 0) list.splice(i, 1);
  }

  terminate(): void {
    void this.worker.terminate();
  }
}

describe("Wave 5 bindings (#210 & #212) on Scene & WorkerScene", () => {
  const cleanups: (() => void)[] = [];

  afterEach(() => {
    for (const c of cleanups.splice(0)) c();
  });

  function openSync() {
    const scene = new Scene();
    cleanups.push(() => scene.free());
    return scene;
  }

  async function openWorker(): Promise<WorkerScene> {
    const worker = await createWorkerScene({
      createWorker: () => new NodeWorker(),
    });
    cleanups.push(() => worker.terminate());
    return worker;
  }

  it("#210: fillHoleContinuity (flat/tangent/curvature), fillAllHolesContinuity, and undo/redo work on Scene & WorkerScene", async () => {
    const continuities: FillContinuity[] = ["flat", "tangent", "curvature"];
    const triCountSync = (s: Scene, id: MeshHandle) => s.inspect.triangleQuality(id).liveTriangleCount;
    const triCountWorker = async (w: WorkerScene, id: MeshHandle) =>
      (await w.inspect.triangleQuality(id)).liveTriangleCount;

    // Sync scene tests across all three continuity levels + fillAllHolesContinuity
    {
      const s = openSync();
      for (const continuity of continuities) {
        const cyl = openCylinderRings(12);
        const id = s.addMesh(cyl.positions, cyl.indices);
        const beforeTris = triCountSync(s, id);
        const stats = s.repair.fillHoleContinuity(id, 0, {
          continuity,
          refinePatch: true,
          smoothIterations: 4,
          detectIslands: true,
        });
        expect(stats.continuity).toBe(continuity);
        expect(stats.filledTriangles).toBeGreaterThan(0);
        expect(stats.converged).toBe(true);
        expect(triCountSync(s, id)).toBe(beforeTris + stats.filledTriangles);

        s.undo();
        expect(triCountSync(s, id)).toBe(beforeTris);
        s.redo();
        expect(triCountSync(s, id)).toBe(beforeTris + stats.filledTriangles);
      }

      const cylAll = openCylinderRings(10);
      const allId = s.addMesh(cylAll.positions, cylAll.indices);
      const beforeAll = triCountSync(s, allId);
      const allStats = s.fillAllHolesContinuity(allId, {
        continuity: "curvature",
        refinePatch: true,
      });
      expect(allStats.continuity).toBe("curvature");
      expect(allStats.filledTriangles).toBeGreaterThanOrEqual(16);
      expect(triCountSync(s, allId)).toBe(beforeAll + allStats.filledTriangles);
      s.undo();
      expect(triCountSync(s, allId)).toBe(beforeAll);
    }

    // WorkerScene tests
    {
      const w = await openWorker();
      const cyl = openCylinderRings(12);
      const id = w.addMesh(cyl.positions, cyl.indices);
      w.commitFrame();
      const before = await triCountWorker(w, id);

      const one = await w.repair.fillHoleContinuity(id, 0, {
        continuity: "tangent",
        refinePatch: true,
      });
      expect(one.continuity).toBe("tangent");
      expect(one.filledTriangles).toBeGreaterThan(0);
      expect(await triCountWorker(w, id)).toBe(before + one.filledTriangles);

      await w.undo();
      expect(await triCountWorker(w, id)).toBe(before);

      const all = await w.fillAllHolesContinuity(id, {
        continuity: "curvature",
        refinePatch: true,
      });
      expect(all.continuity).toBe("curvature");
      expect(all.filledTriangles).toBeGreaterThan(0);
      expect(await triCountWorker(w, id)).toBe(before + all.filledTriangles);
    }
  });

  it("#210: bridgePartialArcs bridges facing boundary arcs with undo/redo on Scene & WorkerScene", async () => {
    const syncRings = twoSeparatedRings(10);
    const s = openSync();
    const sid = s.addMesh(syncRings.positions, syncRings.indices);
    const sBefore = s.inspect.triangleQuality(sid).liveTriangleCount;
    const sAnalysis = s.analyze(sid);
    expect(sAnalysis.loops.length).toBeGreaterThanOrEqual(2);
    const sPos = s.views(sid).positions();
    const loopMidA = sAnalysis.loops.find(
      (loop) => Math.abs(sPos[loop.vertices[0]! * 3 + 2]! + 0.6) < 0.05,
    )!;
    const loopMidB = sAnalysis.loops.find(
      (loop) => Math.abs(sPos[loop.vertices[0]! * 3 + 2]! - 0.6) < 0.05,
    )!;
    expect(loopMidA.vertices.length).toBe(10);
    expect(loopMidB.vertices.length).toBe(10);

    const sStats = s.repair.bridgePartialArcs(sid, loopMidA.vertices, loopMidB.vertices, {
      arcARange: [0, 4],
      arcBRange: [0, 4],
      twistOffset: 0,
      subdivisions: 2,
    });
    expect(sStats.trianglesAdded).toBeGreaterThan(0);
    expect(sStats.arcAVertices).toBeGreaterThan(0);
    expect(sStats.arcBVertices).toBeGreaterThan(0);
    expect(s.inspect.triangleQuality(sid).liveTriangleCount).toBe(sBefore + sStats.trianglesAdded);
    s.undo();
    expect(s.inspect.triangleQuality(sid).liveTriangleCount).toBe(sBefore);

    const w = await openWorker();
    const wRings = twoSeparatedRings(10);
    const wid = w.addMesh(wRings.positions, wRings.indices);
    w.commitFrame();
    const wBefore = (await w.inspect.triangleQuality(wid)).liveTriangleCount;
    const wAnalysis = await w.analyze(wid);
    const wPos = w.views(wid).positions();
    const wA = wAnalysis.loops.find(
      (l) => Math.abs(wPos[l.vertices[0]! * 3 + 2]! + 0.6) < 0.05,
    )!;
    const wB = wAnalysis.loops.find(
      (l) => Math.abs(wPos[l.vertices[0]! * 3 + 2]! - 0.6) < 0.05,
    )!;

    const wStats = await w.bridgePartialArcs(wid, wA.vertices, wB.vertices, {
      arcARange: [0, 4],
      arcBRange: [0, 4],
      subdivisions: 1,
    });
    expect(wStats.trianglesAdded).toBeGreaterThan(0);
    expect((await w.inspect.triangleQuality(wid)).liveTriangleCount).toBe(
      wBefore + wStats.trianglesAdded,
    );
    await w.undo();
    expect((await w.inspect.triangleQuality(wid)).liveTriangleCount).toBe(wBefore);
  });

  it("#212: applyDyntopo, dyntopoDab, SculptDomain dyntopo toggle, and beginStep/endStep undo coalescing on Scene & WorkerScene", async () => {
    const dyntopoOpts: DyntopoOptions = {
      mode: "split-only",
      detailSize: 0.07,
      equalizeValence: true,
      smoothTangent: true,
    };

    // Sync tests
    {
      const s = openSync();
      const grid = gridMesh(8, 8, 2);
      const id = s.addMesh(grid.positions, grid.indices);
      const beforeTris = s.inspect.triangleQuality(id).liveTriangleCount;
      const beforePinnedPos = Array.from(s.views(id).positions().subarray(0, 3));

      const applied = s.applyDyntopo(id, [0, 0, 0] as Vec3, 0.55, dyntopoOpts, [0]);
      expect(applied.splits).toBeGreaterThan(0);
      expect(applied.createdVertices).toBeInstanceOf(Uint32Array);
      expect(applied.removedVertices).toBeInstanceOf(Uint32Array);
      expect(applied.createdVertices.length).toBeGreaterThan(0);
      expect(s.inspect.triangleQuality(id).liveTriangleCount).toBeGreaterThan(beforeTris);
      // Pinned vertex 0 stayed put
      expect(Array.from(s.views(id).positions().subarray(0, 3))).toEqual(beforePinnedPos);

      s.undo();
      expect(s.inspect.triangleQuality(id).liveTriangleCount).toBe(beforeTris);

      // Stroke coalescing via beginStep / endStep + SculptDomain.setDyntopo
      expect(s.sculpt.getDyntopo()).toBeNull();
      s.sculpt.setDyntopo({
        mode: "brush-relative",
        detailSize: 0.06,
      });
      expect(s.sculpt.getDyntopo()?.mode).toBe("brush-relative");

      const preStrokeTris = s.inspect.triangleQuality(id).liveTriangleCount;
      const dab = {
        op: "inflate" as const,
        center: [0, 0, 0] as Vec3,
        radius: 0.5,
        falloff: "smooth" as const,
        strength: 0.15,
        protect: { vertices: [0] },
      };
      s.beginStep();
      const r1 = s.sculpt.dab(id, dab) as DyntopoDabReport;
      const r2 = s.dyntopoDab(
        id,
        { op: "inflate", center: [0.1, 0, 0], radius: 0.5, falloff: "smooth", strength: 0.15 },
        { mode: "constant-detail", detailSize: 0.06 },
        { vertices: [0] },
      );
      s.endStep();

      expect(r1.splits).toBeGreaterThanOrEqual(0);
      expect(r2.createdVertices).toBeInstanceOf(Uint32Array);
      expect(r2.moved).toBeGreaterThan(0);
      expect(s.inspect.triangleQuality(id).liveTriangleCount).toBeGreaterThan(preStrokeTris);
      const postStrokeTris = s.inspect.triangleQuality(id).liveTriangleCount;

      // Single undo reverts BOTH dyntopo dabs in the stroke!
      s.undo();
      expect(s.inspect.triangleQuality(id).liveTriangleCount).toBe(preStrokeTris);
      s.redo();
      expect(s.inspect.triangleQuality(id).liveTriangleCount).toBe(postStrokeTris);
    }

    // WorkerScene tests
    {
      const w = await openWorker();
      const grid = gridMesh(8, 8, 2);
      const id = w.addMesh(grid.positions, grid.indices);
      w.commitFrame();
      const before = (await w.inspect.triangleQuality(id)).liveTriangleCount;

      const applied = await w.applyDyntopo(id, [0, 0, 0], 0.55, dyntopoOpts, [0]);
      expect(applied.splits).toBeGreaterThan(0);
      expect(applied.createdVertices).toBeInstanceOf(Uint32Array);
      expect((await w.inspect.triangleQuality(id)).liveTriangleCount).toBeGreaterThan(before);

      await w.undo();
      expect((await w.inspect.triangleQuality(id)).liveTriangleCount).toBe(before);

      w.sculpt.setDyntopo({ mode: "split-collapse", detailSize: 0.07 });
      expect(w.sculpt.getDyntopo()?.mode).toBe("split-collapse");

      await w.beginStep();
      const wd1 = (await w.sculpt.dab(id, {
        op: "inflate",
        center: [0, 0, 0],
        radius: 0.5,
        falloff: "smooth",
        strength: 0.15,
      })) as DyntopoDabReport;
      const wd2 = await w.dyntopoDab(
        id,
        { op: "inflate", center: [0.1, 0, 0], radius: 0.5, falloff: "smooth", strength: 0.15 },
        { mode: "split-only", detailSize: 0.06 },
        { vertices: [0] },
      );
      await w.endStep();

      expect(wd1.moved).toBeGreaterThan(0);
      expect(wd2.createdVertices).toBeInstanceOf(Uint32Array);
      const postStroke = (await w.inspect.triangleQuality(id)).liveTriangleCount;
      expect(postStroke).toBeGreaterThan(before);

      await w.undo();
      expect((await w.inspect.triangleQuality(id)).liveTriangleCount).toBe(before);
      await w.redo();
      expect((await w.inspect.triangleQuality(id)).liveTriangleCount).toBe(postStroke);
    }
  });
});
