// Sculpt domain API (`Scene.sculpt`, `WorkerScene.sculpt`, `SceneStore.sculpt`,
// `verticesInSphere`, brush algorithms, surface filters, extra brushes, symmetry,
// and soft-band vertex protection) across synchronous `Scene`, `WorkerScene`,
// and `SceneStore`.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it } from "vitest";
import {
  type ClayProfile,
  type ClayReport,
  type CreaseReport,
  FastMeshError,
  type FlattenMode,
  type FlattenReport,
  type GrabMethod,
  type InflateMethod,
  type MeshHandle,
  Scene,
  type SculptOp,
  type SculptReport,
  type SmoothMethod,
  type SurfaceFilter,
  type Symmetry,
  type TwistMethod,
  type TwistReport,
  type Vec3,
} from "../src/node.js";
import { createSceneStore, type SceneStore } from "../src/store/index.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";

class NodeWorker implements WorkerLike {
  readonly worker: Worker;
  readonly #message: ((e: { data: unknown }) => void)[] = [];
  readonly #error: ((e: unknown) => void)[] = [];

  constructor() {
    this.worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), {
      workerData: { serve: true },
    });
    this.worker.on("message", (data: unknown) => {
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
});

function makeSync(): Scene {
  const s = new Scene();
  cleanup.push(() => s.free());
  return s;
}

async function makeWorker(): Promise<WorkerScene> {
  const w = await createWorkerScene({ createWorker: () => new NodeWorker() });
  cleanup.push(() => w.terminate());
  return w;
}

function makeStore(scene: Scene | WorkerScene): SceneStore {
  const store = createSceneStore(scene);
  cleanup.push(() => store.dispose());
  return store;
}

/** Symmetric (2*half+1)^2 vertex grid on [-half, half]^2, z(x, y) given by `height`. */
function gridMesh(
  half: number,
  step = 1,
  height: (x: number, y: number) => number = () => 0,
): { positions: Float64Array; indices: Uint32Array; n: number } {
  const n = 2 * half + 1;
  const positions = new Float64Array(n * n * 3);
  for (let iy = 0; iy < n; iy++) {
    const y = (iy - half) * step;
    for (let ix = 0; ix < n; ix++) {
      const x = (ix - half) * step;
      const k = (iy * n + ix) * 3;
      positions[k] = x;
      positions[k + 1] = y;
      positions[k + 2] = height(x, y);
    }
  }
  const indices = new Uint32Array((n - 1) * (n - 1) * 6);
  let at = 0;
  for (let iy = 0; iy < n - 1; iy++) {
    for (let ix = 0; ix < n - 1; ix++) {
      const a = iy * n + ix;
      indices.set([a, a + 1, a + n + 1, a, a + n + 1, a + n], at);
      at += 6;
    }
  }
  return { positions, indices, n };
}

/** Two parallel flat sheets separated by an air gap `dz`, origin fixed at `[0, 0, 0]`. */
function twoSheets(dz: number): { positions: Float64Array; indices: Uint32Array; sheetVertices: number } {
  const g = gridMesh(3, 1);
  const sheetVertices = g.positions.length / 3;
  const positions = new Float64Array(g.positions.length * 2);
  positions.set(g.positions, 0);
  for (let v = 0; v < sheetVertices; v++) {
    positions[sheetVertices * 3 + v * 3] = g.positions[v * 3];
    positions[sheetVertices * 3 + v * 3 + 1] = g.positions[v * 3 + 1];
    positions[sheetVertices * 3 + v * 3 + 2] = dz;
  }
  const indices = new Uint32Array(g.indices.length * 2);
  indices.set(g.indices, 0);
  for (let i = 0; i < g.indices.length; i++) {
    indices[g.indices.length + i] = g.indices[i] + sheetVertices;
  }
  return { positions, indices, sheetVertices };
}

function syncCodeOf(fn: () => unknown): string {
  try {
    fn();
  } catch (e) {
    expect(e).toBeInstanceOf(FastMeshError);
    return (e as FastMeshError).code;
  }
  throw new Error("expected FastMeshError");
}

async function asyncCodeOf(fn: () => Promise<unknown>): Promise<string> {
  try {
    await fn();
  } catch (e) {
    expect(e).toBeInstanceOf(FastMeshError);
    return (e as FastMeshError).code;
  }
  throw new Error("expected FastMeshError rejection");
}

interface HostHarness {
  label: "sync" | "worker" | "store-sync" | "store-worker";
  create(): Promise<{
    addMesh(positions: Float64Array, indices: Uint32Array): Promise<MeshHandle>;
    sculpt<const O extends SculptOp>(mesh: MeshHandle, op: O): Promise<SculptReport>;
    sculptList(mesh: MeshHandle, ops: readonly SculptOp[]): Promise<SculptReport[]>;
    repair(mesh: MeshHandle, op: SculptOp): Promise<SculptReport>;
    verticesInSphere(mesh: MeshHandle, center: Vec3, radius: number): Promise<Uint32Array>;
    positions(mesh: MeshHandle): Float32Array;
  }>;
}

const HARNESSES: readonly HostHarness[] = [
  {
    label: "sync",
    async create() {
      const s = makeSync();
      return {
        addMesh: async (p, i) => s.addMesh(p, i, { origin: [0, 0, 0] }),
        sculpt: async (m, op) => s.sculpt(m, op),
        sculptList: async (m, ops) => s.sculpt(m, ops),
        repair: async (m, op) => s.repair(m, op),
        verticesInSphere: async (m, c, r) => s.verticesInSphere(m, c, r),
        positions: (m) => Float32Array.from(s.views(m).positions()),
      };
    },
  },
  {
    label: "worker",
    async create() {
      const w = await makeWorker();
      return {
        addMesh: async (p, i) => {
          const m = w.addMesh(p, i, { origin: [0, 0, 0] });
          await w.measure(m);
          return m;
        },
        sculpt: (m, op) => w.sculpt(m, op),
        sculptList: (m, ops) => w.sculpt(m, ops),
        repair: (m, op) => w.repair(m, op),
        verticesInSphere: (m, c, r) => w.verticesInSphere(m, c, r),
        positions: (m) => Float32Array.from(w.views(m).positions()),
      };
    },
  },
  {
    label: "store-sync",
    async create() {
      const s = makeSync();
      const store = makeStore(s);
      return {
        addMesh: async (p, i) => store.addMesh(p, i, { origin: [0, 0, 0] }),
        sculpt: (m, op) => store.sculpt(m, op),
        sculptList: (m, ops) => store.sculpt(m, ops),
        repair: (m, op) => store.repair(m, op) as Promise<SculptReport>,
        verticesInSphere: (m, c, r) => store.verticesInSphere(m, c, r),
        positions: (m) => Float32Array.from(s.views(m).positions()),
      };
    },
  },
  {
    label: "store-worker",
    async create() {
      const w = await makeWorker();
      const store = makeStore(w);
      return {
        addMesh: async (p, i) => {
          const m = store.addMesh(p, i, { origin: [0, 0, 0] });
          await w.measure(m);
          return m;
        },
        sculpt: (m, op) => store.sculpt(m, op),
        sculptList: (m, ops) => store.sculpt(m, ops),
        repair: (m, op) => store.repair(m, op) as Promise<SculptReport>,
        verticesInSphere: (m, c, r) => store.verticesInSphere(m, c, r),
        positions: (m) => Float32Array.from(w.views(m).positions()),
      };
    },
  },
];

describe.each(HARNESSES)("sculpt domain ($label)", (harness) => {
  it("runs inflate, smooth, and grab methods (including strokeNormal, geodesic, kelvinlet, fromStroke + adaptiveReach)", async () => {
    const h = await harness.create();

    // 1. Inflate methods: vertexNormal, averageNormal, strokeNormal with captureStrokeState.
    const ridge = gridMesh(4, 0.5, (x) => Math.max(0, 1 - Math.abs(x)));
    const inflateResults: Float32Array[] = [];
    for (const method of ["vertexNormal", "averageNormal", "strokeNormal"] as const satisfies readonly InflateMethod[]) {
      const m = await h.addMesh(ridge.positions, ridge.indices);
      if (method === "strokeNormal") {
        expect(await h.sculpt(m, { op: "captureStrokeState" })).toEqual({});
      }
      const r1 = await h.sculpt(m, {
        op: "inflate",
        center: [0, 0, 1],
        radius: 1.5,
        falloff: "smooth",
        strength: 0.3,
        method,
      });
      expect(r1).toHaveProperty("moved");
      expect((r1 as { moved: number }).moved).toBeGreaterThan(0);
      if (method === "strokeNormal") {
        // On a flat sheet, captured +Z stroke normal stays strictly vertical across repeated off-center dabs!
        const flat = gridMesh(4, 0.5);
        const mf = await h.addMesh(flat.positions, flat.indices);
        await h.sculpt(mf, { op: "captureStrokeState" });
        await h.sculpt(mf, { op: "inflate", center: [0, 0, 0], radius: 1.2, falloff: "smooth", strength: 0.8, method: "strokeNormal" });
        await h.sculpt(mf, { op: "inflate", center: [0.4, 0, 0.5], radius: 1.2, falloff: "smooth", strength: 0.8, method: "strokeNormal" });
        const pf = h.positions(mf);
        for (let v = 0; v < pf.length / 3; v++) {
          expect(pf[v * 3]).toBeCloseTo(flat.positions[v * 3], 5);
          expect(pf[v * 3 + 1]).toBeCloseTo(flat.positions[v * 3 + 1], 5);
        }
      }
      inflateResults.push(h.positions(m));
    }
    // vertexNormal and averageNormal produce distinct profiles on a sharp ridge.
    expect(Array.from(inflateResults[0])).not.toEqual(Array.from(inflateResults[1]));

    // 2. Smooth methods: taubin, laplacian, hcLaplacian, cotangent.
    const bumpy = gridMesh(4, 0.5, (x, y) => (x === 0 && y === 0 ? 1.2 : 0.1 * Math.sin(2 * x)));
    for (const method of ["taubin", "laplacian", "hcLaplacian", "cotangent"] as const satisfies readonly SmoothMethod[]) {
      const m = await h.addMesh(bumpy.positions, bumpy.indices);
      const r = await h.sculpt(m, {
        op: "smooth",
        center: [0, 0, 1.2],
        radius: 2.0,
        falloff: "smooth",
        strength: 0.8,
        cycles: 2,
        method,
      });
      expect((r as { moved: number }).moved).toBeGreaterThan(0);
      const centerIdx = (4 * 9 + 4) * 3 + 2;
      expect(h.positions(m)[centerIdx]).toBeLessThan(1.2);
    }

    // 3. Grab methods: sphere vs geodesic across separated sheets, kelvinlet with poisson, and fromStroke + adaptiveReach + splitStretched.
    const sheets = twoSheets(0.4);
    const mSphere = await h.addMesh(sheets.positions, sheets.indices);
    const mGeo = await h.addMesh(sheets.positions, sheets.indices);
    const grabBase = {
      op: "grab",
      center: [0, 0, 0] as Vec3,
      radius: 1.2,
      falloff: "smooth" as const,
      strength: 1.0,
      direction: [0, 0, 0.5] as Vec3,
    };
    await h.sculpt(mSphere, { ...grabBase, method: "sphere" satisfies GrabMethod });
    await h.sculpt(mGeo, { ...grabBase, method: "geodesic" satisfies GrabMethod });
    const pSphere = h.positions(mSphere);
    const pGeo = h.positions(mGeo);
    let upperMovedSphere = 0;
    let upperMovedGeo = 0;
    for (let v = sheets.sheetVertices; v < sheets.sheetVertices * 2; v++) {
      if (Math.abs(pSphere[v * 3 + 2] - 0.4) > 1e-5) upperMovedSphere++;
      if (Math.abs(pGeo[v * 3 + 2] - 0.4) > 1e-5) upperMovedGeo++;
    }
    expect(upperMovedSphere).toBeGreaterThan(0);
    expect(upperMovedGeo).toBe(0);

    // Kelvinlet with poisson ratio induces cross-axis coupling & varies with poisson.
    const flat = gridMesh(4, 0.5);
    const mKelvHigh = await h.addMesh(flat.positions, flat.indices);
    const mKelvZero = await h.addMesh(flat.positions, flat.indices);
    const rk = await h.sculpt(mKelvHigh, {
      op: "grab",
      center: [0, 0, 0.5],
      radius: 2.0,
      falloff: "smooth",
      strength: 0.6,
      direction: [0, 0, 1],
      method: "kelvinlet",
      poisson: 0.49,
    });
    await h.sculpt(mKelvZero, {
      op: "grab",
      center: [0, 0, 0.5],
      radius: 2.0,
      falloff: "smooth",
      strength: 0.6,
      direction: [0, 0, 1],
      method: "kelvinlet",
      poisson: 0.0,
    });
    expect((rk as { moved: number }).moved).toBeGreaterThan(1);
    const pkHigh = h.positions(mKelvHigh);
    const pkZero = h.positions(mKelvZero);
    let lateralShift = 0;
    let poissonDiff = 0;
    for (let v = 0; v < pkHigh.length / 3; v++) {
      lateralShift = Math.max(lateralShift, Math.abs(pkHigh[v * 3] - flat.positions[v * 3]));
      poissonDiff = Math.max(poissonDiff, Math.abs(pkHigh[v * 3 + 2] - pkZero[v * 3 + 2]));
    }
    expect(lateralShift).toBeGreaterThan(1e-4);
    expect(poissonDiff).toBeGreaterThan(1e-4);

    // fromStroke + adaptiveReach + splitStretched preserves non-cumulative stroke rest positions!
    const mStroke = await h.addMesh(flat.positions, flat.indices);
    await h.sculpt(mStroke, { op: "captureRestSizes" });
    await h.sculpt(mStroke, { op: "captureStrokeState" });
    await h.sculpt(mStroke, {
      op: "grab",
      center: [0, 0, 0],
      radius: 0.8,
      falloff: "smooth",
      strength: 1.0,
      direction: [1.5, 0, 0],
      fromStroke: true,
      adaptiveReach: true,
    });
    const splitRep = await h.sculpt(mStroke, { op: "splitStretched", center: [0, 0, 0], radius: 4.0 });
    expect((splitRep as { splits: number }).splits).toBeGreaterThan(0);
    await h.sculpt(mStroke, {
      op: "grab",
      center: [0, 0, 0],
      radius: 0.8,
      falloff: "smooth",
      strength: 1.0,
      direction: [0, 0, 1.2],
      fromStroke: true,
      adaptiveReach: true,
    });
    const centerVid = 4 * 9 + 4;
    const ps = h.positions(mStroke);
    // X snapped back to rest 0 while Z moved to 1.2!
    expect(ps[centerVid * 3]).toBeCloseTo(0, 4);
    expect(ps[centerVid * 3 + 2]).toBeCloseTo(1.2, 4);
  });

  it("supports SurfaceFilter, extra brushes (flatten, crease, clay, twist), and Symmetry via both sculpt() and repair()", async () => {
    const h = await harness.create();

    // SurfaceFilter: peaksOnly vs valleysOnly on a mesh with both a peak and a pit.
    const relief = gridMesh(4, 0.5, (x, y) => {
      if (x === -0.5 && y === 0) return 0.6;
      if (x === 0.5 && y === 0) return -0.6;
      return 0;
    });
    const peakVid = 4 * 9 + 3;
    const pitVid = 4 * 9 + 5;

    const mPeaks = await h.addMesh(relief.positions, relief.indices);
    const peaksFilter: SurfaceFilter = { kind: "peaksOnly", threshold: 0.05, contrast: 0.8 };
    await h.sculpt(mPeaks, {
      op: "smooth",
      center: [0, 0, 0],
      radius: 2.5,
      falloff: "linear",
      strength: 0.8,
      cycles: 2,
      method: "laplacian",
      filter: peaksFilter,
    });
    const pp = h.positions(mPeaks);
    expect(pp[peakVid * 3 + 2]).toBeLessThan(0.55);
    expect(pp[pitVid * 3 + 2]).toBeCloseTo(-0.6, 5);

    const mValleys = await h.addMesh(relief.positions, relief.indices);
    const valleysFilter: SurfaceFilter = { kind: "valleysOnly", threshold: 0.05, contrast: 0.8 };
    await h.repair(mValleys, {
      op: "smooth",
      center: [0, 0, 0],
      radius: 2.5,
      falloff: "linear",
      strength: 0.8,
      cycles: 2,
      method: "laplacian",
      filter: valleysFilter,
    });
    const pv = h.positions(mValleys);
    expect(pv[peakVid * 3 + 2]).toBeCloseTo(0.6, 5);
    expect(pv[pitVid * 3 + 2]).toBeGreaterThan(-0.55);

    // NormalAngle filter masks steep side walls while moving +Z-facing top.
    const stepWall = gridMesh(4, 0.5, (x) => (x > 0 ? 2.0 * x : 0));
    const mNorm = await h.addMesh(stepWall.positions, stepWall.indices);
    await h.sculpt(mNorm, {
      op: "inflate",
      center: [0, 0, 0],
      radius: 3.0,
      falloff: "linear",
      strength: 0.4,
      filter: { kind: "normalAngle", normal: [0, 0, 1], maxAngleDeg: 20 },
    });
    const pn = h.positions(mNorm);
    const flatVid = 4 * 9 + 1; // x = -1.5, normal = (0, 0, 1)
    const steepVid = 4 * 9 + 7; // x = +1.5, normal tilted ~63.4 deg > 20 deg
    expect(pn[flatVid * 3 + 2]).toBeGreaterThan(0.05);
    expect(pn[steepVid * 3 + 2]).toBeCloseTo(stepWall.positions[steepVid * 3 + 2], 5);

    // Extra brushes: flatten (both/peaks/valleys), crease, clay (round/strips), twist (vortex/kelvinlet).
    for (const mode of ["both", "peaks", "valleys"] as const satisfies readonly FlattenMode[]) {
      const m = await h.addMesh(relief.positions, relief.indices);
      const rep: FlattenReport = (await h.sculpt(m, {
        op: "flatten",
        center: [0, 0, 0],
        radius: 2.0,
        falloff: "smooth",
        strength: 0.9,
        mode,
        planeOrigin: [0, 0, 0],
        planeNormal: [0, 0, 1],
      })) as FlattenReport;
      expect(rep.moved).toBeGreaterThan(0);
    }

    const flat = gridMesh(4, 0.5);
    const mCrease = await h.addMesh(flat.positions, flat.indices);
    const creaseRep = (await h.sculpt(mCrease, {
      op: "crease",
      center: [0, 0, 0],
      radius: 1.5,
      falloff: "smooth",
      strength: -0.4,
      pinch: 0.6,
    })) as CreaseReport;
    expect(creaseRep.moved).toBeGreaterThan(0);
    expect(h.positions(mCrease)[(4 * 9 + 4) * 3 + 2]).toBeLessThan(0);

    for (const profile of ["round", "strips"] as const satisfies readonly ClayProfile[]) {
      const mClay = await h.addMesh(flat.positions, flat.indices);
      const clayRep = (await h.sculpt(mClay, {
        op: "clay",
        center: [0, 0, 0],
        radius: 1.5,
        falloff: "smooth",
        strength: 0.8,
        profile,
        tangent: profile === "strips" ? [1, 0, 0] : undefined,
      })) as ClayReport;
      expect(clayRep.moved).toBeGreaterThan(0);
      expect(h.positions(mClay)[(4 * 9 + 4) * 3 + 2]).toBeGreaterThan(0);
    }

    for (const method of ["vortex", "kelvinlet"] as const satisfies readonly TwistMethod[]) {
      const mTwist = await h.addMesh(flat.positions, flat.indices);
      const twistRep = (await h.repair(mTwist, {
        op: "twist",
        center: [0, 0, 0],
        radius: 1.8,
        falloff: "smooth",
        strength: 0.9,
        method,
        axis: [0, 0, 1],
      })) as TwistReport;
      expect(twistRep.moved).toBeGreaterThan(0);
    }

    // Symmetry { x: true, y: true, z: false } via both sculpt() and repair() mirrors across all 4 XY quadrants!
    const sym: Symmetry = { x: true, y: true, z: false };
    const mSymSculpt = await h.addMesh(flat.positions, flat.indices);
    const mSymRepair = await h.addMesh(flat.positions, flat.indices);
    const symOp: SculptOp = {
      op: "inflate",
      center: [1.5, 1.5, 0],
      radius: 0.9,
      falloff: "smooth",
      strength: 0.5,
      symmetry: sym,
    };
    await h.sculpt(mSymSculpt, symOp);
    await h.repair(mSymRepair, symOp);
    const pSym = h.positions(mSymSculpt);
    expect(Array.from(pSym)).toEqual(Array.from(h.positions(mSymRepair)));
    const qPP = (7 * 9 + 7) * 3 + 2; // (+1.5, +1.5)
    const qNP = (7 * 9 + 1) * 3 + 2; // (-1.5, +1.5)
    const qPN = (1 * 9 + 7) * 3 + 2; // (+1.5, -1.5)
    const qNN = (1 * 9 + 1) * 3 + 2; // (-1.5, -1.5)
    expect(pSym[qPP]).toBeGreaterThan(0.4);
    expect(pSym[qNP]).toBeCloseTo(pSym[qPP], 5);
    expect(pSym[qPN]).toBeCloseTo(pSym[qPP], 5);
    expect(pSym[qNN]).toBeCloseTo(pSym[qPP], 5);
  });

  it("applies soft-band vertex protection and returns identical verticesInSphere", async () => {
    const h = await harness.create();
    const g = gridMesh(5, 0.5);
    const m = await h.addMesh(g.positions, g.indices);

    // verticesInSphere picks exact local-space sphere vertices in ascending order.
    const picked = await h.verticesInSphere(m, [0, 0, 0], 0.6);
    expect(picked).toBeInstanceOf(Uint32Array);
    // Center (5,5)=60 plus its 4 axial neighbors at distance 0.5: 49, 59, 61, 71.
    expect(Array.from(picked)).toEqual([49, 59, 60, 61, 71]);

    // Soft-band protect: hold center vertex 60 at weight 0 with band = 1.2!
    // Unprotected control mesh for comparison:
    const mFree = await h.addMesh(g.positions, g.indices);
    const brush = {
      op: "grab" as const,
      center: [0, 0, 0] as Vec3,
      radius: 2.5,
      falloff: "linear" as const,
      strength: 1.0,
      direction: [0, 0, 1] as Vec3,
    };
    await h.sculpt(mFree, brush);
    await h.sculpt(m, {
      ...brush,
      protect: { vertices: Uint32Array.of(60), weights: Float32Array.of(0), band: 1.2 },
    });

    const pBand = h.positions(m);
    const pFree = h.positions(mFree);
    // Center vertex 60 is held at strictly 0 displacement.
    expect(pBand[60 * 3 + 2]).toBe(0);
    // Immediate 1-ring neighbor 61 (dist 0.5 < band 1.2) moves > 0, feathered less than unmasked mFree,
    // and less than 2-ring neighbor 62 (dist 1.0) relative to its unmasked displacement!
    const ratio1 = pBand[61 * 3 + 2] / pFree[61 * 3 + 2];
    const ratio2 = pBand[62 * 3 + 2] / pFree[62 * 3 + 2];
    const ratio3 = pBand[63 * 3 + 2] / pFree[63 * 3 + 2]; // dist 1.5 > band 1.2 => weight 1.0
    expect(ratio1).toBeGreaterThan(0);
    expect(ratio1).toBeLessThan(ratio2);
    expect(ratio2).toBeLessThan(1);
    expect(ratio3).toBeCloseTo(1, 5);
  });

  it("rejects invalid sculpt options with INVALID_ARGUMENT and zero mesh mutation", async () => {
    const h = await harness.create();
    const g = gridMesh(3, 1);
    const m = await h.addMesh(g.positions, g.indices);
    const before = h.positions(m);

    // 1. poisson out of [0, 0.5]
    expect(
      await asyncCodeOf(() =>
        h.sculpt(m, {
          op: "grab",
          center: [0, 0, 0],
          radius: 2,
          falloff: "smooth",
          strength: 1,
          direction: [0, 0, 1],
          method: "kelvinlet",
          poisson: 0.9,
        }),
      ),
    ).toBe("INVALID_ARGUMENT");

    // 2. maxAngleDeg <= 0
    expect(
      await asyncCodeOf(() =>
        h.sculpt(m, {
          op: "inflate",
          center: [0, 0, 0],
          radius: 2,
          falloff: "smooth",
          strength: 0.5,
          filter: { kind: "normalAngle", maxAngleDeg: 0 },
        }),
      ),
    ).toBe("INVALID_ARGUMENT");

    // 3. captureStrokeState mixed in a multi-op list
    expect(
      await asyncCodeOf(() =>
        h.sculptList(m, [
          { op: "inflate", center: [0, 0, 0], radius: 2, falloff: "smooth", strength: 0.5 },
          { op: "captureStrokeState" },
        ]),
      ),
    ).toBe("INVALID_ARGUMENT");

    // 4. negative radius in verticesInSphere
    expect(await asyncCodeOf(() => h.verticesInSphere(m, [0, 0, 0], -1))).toBe("INVALID_ARGUMENT");

    expect(Array.from(h.positions(m))).toEqual(Array.from(before));
  });
});

describe("store history.canUndo across captureStrokeState and sculpt", () => {
  it.each(["sync", "worker"] as const)(
    "keeps history.canUndo false after captureStrokeState and flips true after sculpt (%s)",
    async (backend) => {
      const s = backend === "sync" ? makeSync() : await makeWorker();
      const store = makeStore(s);
      const g = gridMesh(3, 1);
      const m = store.addMesh(g.positions, g.indices, { origin: [0, 0, 0] });
      if (backend === "worker") {
        await (s as Awaited<ReturnType<typeof makeWorker>>).measure(m);
      }
      expect(store.getSnapshot().history.canUndo).toBe(false);

      await store.sculpt(m, { op: "captureStrokeState" });
      expect(store.getSnapshot().history.canUndo).toBe(false);

      await store.sculpt(m, {
        op: "grab",
        center: [0, 0, 0],
        radius: 2,
        falloff: "smooth",
        strength: 0.6,
        direction: [0, 0, 1],
        fromStroke: true,
      });
      expect(store.getSnapshot().history.canUndo).toBe(true);
    },
  );
});

describe("sync Scene.sculpt synchronous error semantics", () => {
  it("throws FastMeshError synchronously on bad options without mutating or recording undo", () => {
    const s = makeSync();
    const g = gridMesh(2, 1);
    const m = s.addMesh(g.positions, g.indices, { origin: [0, 0, 0] });
    const before = Float32Array.from(s.views(m).positions());

    expect(
      syncCodeOf(() =>
        s.sculpt(m, {
          op: "grab",
          center: [0, 0, 0],
          radius: 1.5,
          falloff: "smooth",
          strength: 1,
          direction: [0, 0, 1],
          method: "kelvinlet",
          poisson: 0.9,
        }),
      ),
    ).toBe("INVALID_ARGUMENT");
    expect(
      syncCodeOf(() =>
        s.sculpt(m, {
          op: "inflate",
          center: [0, 0, 0],
          radius: 1.5,
          falloff: "smooth",
          strength: 0.5,
          filter: { kind: "normalAngle", maxAngleDeg: 0 },
        }),
      ),
    ).toBe("INVALID_ARGUMENT");
    expect(
      syncCodeOf(() =>
        s.sculpt(m, [
          { op: "inflate", center: [0, 0, 0], radius: 1.5, falloff: "smooth", strength: 0.5 },
          { op: "captureStrokeState" },
        ]),
      ),
    ).toBe("INVALID_ARGUMENT");
    expect(s.canUndo()).toBe(false);
    expect(Array.from(s.views(m).positions())).toEqual(Array.from(before));
  });
});
