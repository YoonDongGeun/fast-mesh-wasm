// Contours, fits, point clouds, trims, selections, surface paths and stamps far
// from the world origin (#280): each the same as near it, moved. Every case runs
// on a Scene and on a WorkerScene.
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it } from "vitest";
import {
  type ContourLoop,
  type ContourSlice,
  type FittedPrimitive,
  FastMeshError,
  type MeshHandle,
  packPointCloud,
  type PackedPointCloud,
  parsePointCloudBytes,
  parsePointCloudText,
  type PlacementInput,
  pointCloudEstimateNormals,
  pointCloudRor,
  pointCloudSor,
  pointCloudVoxelDownsample,
  Scene,
  sliceContoursExport,
  unpackPointCloud,
  type Vec3,
} from "../src/node.js";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";

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

type AnyScene = Scene | WorkerScene;

const FAR: Vec3 = [148_000.3, -277_000.7, 91_234.1];
const ZERO: Vec3 = [0, 0, 0];
/** Several `f32` steps at the sizes here (up to 2). `f32` world positions are off by 0.0156 at FAR. */
const TOL = 1e-6;

const add = (a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

/** The largest distance along an axis of `got` from `want` moved by `shift`. */
function off(got: ArrayLike<number>, want: ArrayLike<number>, shift: Readonly<Vec3>): number {
  expect(got.length).toBe(want.length);
  let m = 0;
  for (let i = 0; i < got.length; i++) m = Math.max(m, Math.abs(got[i] - (want[i] + shift[i % 3])));
  return m;
}

const MID: Vec3 = [0.5, 0.4, 0.3];
const RADIUS = 0.6;
const RINGS = 24;
const SEGMENTS = 48;

/** A ball around `at` with small holes at its poles. */
function ball(at: Readonly<Vec3>): { positions: Float64Array; indices: Uint32Array } {
  const positions: number[] = [];
  const indices: number[] = [];
  for (let i = 0; i <= RINGS; i++) {
    const polar = Math.PI * (0.04 + (0.92 * i) / RINGS);
    for (let j = 0; j < SEGMENTS; j++) {
      const turn = (2 * Math.PI * j) / SEGMENTS;
      const r = RADIUS * Math.sin(polar);
      positions.push(at[0] + r * Math.cos(turn), at[1] + r * Math.sin(turn), at[2] - RADIUS * Math.cos(polar));
    }
  }
  for (let i = 0; i < RINGS; i++) {
    for (let j = 0; j < SEGMENTS; j++) {
      const a = i * SEGMENTS + j;
      const b = i * SEGMENTS + ((j + 1) % SEGMENTS);
      indices.push(a, b, b + SEGMENTS, a, b + SEGMENTS, a + SEGMENTS);
    }
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

function box(lo: Readonly<Vec3>, hi: Readonly<Vec3>): { positions: Float64Array; indices: Uint32Array } {
  const positions: number[] = [];
  for (const z of [lo[2], hi[2]]) {
    positions.push(lo[0], lo[1], z, hi[0], lo[1], z, hi[0], hi[1], z, lo[0], hi[1], z);
  }
  const indices = [
    0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 3, 0, 4, 3, 4, 7,
  ];
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

/** Points of the ball around the one at `+x`, `angle` radians from it. */
function around(angle: number, count: number): Vec3[] {
  return Array.from({ length: count }, (_, k) => {
    const turn = (2 * Math.PI * k) / count + 0.05;
    return [
      MID[0] + RADIUS * Math.cos(angle),
      MID[1] + RADIUS * Math.sin(angle) * Math.cos(turn),
      MID[2] + RADIUS * Math.sin(angle) * Math.sin(turn),
    ] as Vec3;
  });
}

/** The mesh's vertices in the world, through its `matrix4`, flat. */
function worldPositions(s: AnyScene, mesh: MeshHandle): number[] {
  const p = s.views(mesh).positions();
  const m = s.matrix4(mesh);
  const out: number[] = [];
  for (let i = 0; i + 2 < p.length; i += 3) {
    for (let r = 0; r < 3; r++) out.push(m[r] * p[i] + m[4 + r] * p[i + 1] + m[8 + r] * p[i + 2] + m[12 + r]);
  }
  return out;
}

/** A placement that scales unevenly, and where it takes a point. */
const UNEVEN: PlacementInput = { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [7, -42.5, 300], scale: [1.5, 0.5, 2] };
const uneven = (p: Readonly<Vec3>): Vec3 => [1.5 * p[0] + 7, 0.5 * p[1] - 42.5, 2 * p[2] + 300];

describe("far from the world origin (#280)", () => {
  const cleanups: (() => void)[] = [];

  afterEach(() => {
    for (const c of cleanups.splice(0)) c();
  });

  /** Runs `body` on a Scene and on a WorkerScene. */
  async function onBoth(body: (s: AnyScene) => Promise<void>): Promise<void> {
    const scene = new Scene();
    cleanups.push(() => scene.free());
    await body(scene);
    const worker = await createWorkerScene({ createWorker: () => new NodeWorker() });
    cleanups.push(() => worker.terminate());
    await body(worker);
  }

  async function addBall(s: AnyScene, at: Readonly<Vec3>): Promise<MeshHandle> {
    const { positions, indices } = ball(add(at, MID));
    const mesh = s.addMesh(positions, indices);
    await s.commitFrame();
    return mesh;
  }

  it("slices contours, with f64 points", async () => {
    await onBoth(async (s) => {
      const slices = async (at: Readonly<Vec3>, normal: Vec3): Promise<ContourSlice[]> => {
        const mesh = await addBall(s, at);
        const origin = add(at, MID);
        return [
          await s.inspect.sliceContours(mesh, origin, normal),
          ...(await s.inspect.sliceContourStack(mesh, origin, normal, 0.05, 3)),
        ];
      };
      for (const normal of [[0, 0, 1], [0.36, 0.48, 0.8]] as Vec3[]) {
        const [near, far] = [await slices(ZERO, normal), await slices(FAR, normal)];
        for (const [i, slice] of far.entries()) {
          expect(slice.loops.length).toBe(1);
          const [got, want] = [slice.loops[0], near[i].loops[0]] as [Required<ContourLoop>, Required<ContourLoop>];
          expect(got.points3d64).toBeInstanceOf(Float64Array);
          expect(off(got.points3d64, want.points3d64, FAR)).toBeLessThan(TOL);
          expect(off(got.centroid3d, want.centroid3d, FAR)).toBeLessThan(TOL);
          expect(off(slice.planeOrigin, near[i].planeOrigin, FAR)).toBeLessThan(TOL);
          expect(off(got.points2d, want.points2d, ZERO)).toBeLessThan(4 * TOL);
          expect(got.perimeter).toBeCloseTo(want.perimeter, 5);
          expect(got.enclosedArea).toBeCloseTo(want.enclosedArea, 5);
          // The f32 points are those, rounded.
          expect(got.points3d).toEqual(Float32Array.from(got.points3d64));
          expect(off(got.points3d, got.points3d64, ZERO)).toBeGreaterThan(1e-3);
          // The plane's axes and (u, v) lead to them too.
          const uv = got.points2d;
          const fromPlane = Float64Array.from(
            got.points3d64,
            (_, k) =>
              slice.planeOrigin[k % 3] +
              uv[2 * Math.floor(k / 3)] * slice.uAxis[k % 3] +
              uv[2 * Math.floor(k / 3) + 1] * slice.vAxis[k % 3],
          );
          expect(off(fromPlane, got.points3d64, ZERO)).toBeLessThan(TOL);
        }
      }
    });
  });

  it("slices along a plane given by a point far from the mesh", async () => {
    await onBoth(async (s) => {
      const normal: Vec3 = [0.6, 0, 0.8];
      const [near, far] = [await addBall(s, ZERO), await addBall(s, FAR)];
      const want = (await s.inspect.sliceContours(near, MID, normal)).loops[0];
      // Along (0.8, 0, -0.6), in the plane: 1e5 away.
      const there: Vec3 = [FAR[0] + MID[0] + 80_000, FAR[1] + MID[1], FAR[2] + MID[2] - 60_000];
      const slice = await s.inspect.sliceContours(far, there, normal);
      expect(off(slice.planeOrigin, there, ZERO)).toBeLessThan(TOL);
      expect(off(slice.loops[0].points3d64!, want.points3d64!, FAR)).toBeLessThan(TOL);
    });
  });

  it("exports the f64 points, or the f32 ones of a loop without them", async () => {
    const scene = new Scene();
    cleanups.push(() => scene.free());
    const mesh = await addBall(scene, FAR);
    const slice = scene.sliceContours(mesh, add(FAR, MID), [0, 0, 1]);
    const points = (s: ContourSlice) =>
      (JSON.parse(sliceContoursExport(s, "json")).slices[0].loops[0].points_3d as number[][]).flat();
    const loop = slice.loops[0];
    expect(off(points(slice), loop.points3d64!, ZERO)).toBeLessThan(1e-6);
    const { points3d64: _, ...plain } = loop;
    expect(off(points({ ...slice, loops: [plain] }), loop.points3d, ZERO)).toBeLessThan(1e-6);
  });

  it("fits primitives", async () => {
    await onBoth(async (s) => {
      const fit = async (at: Readonly<Vec3>) => {
        const mesh = await addBall(s, at);
        const report = await s.inspect.fitPrimitive(mesh, { kind: "sphere" });
        const aligned = await s.inspect.alignToFittedPrimitive(mesh, report.primitive, [0, 0, 1], ZERO);
        return { sphere: report.primitive as Extract<FittedPrimitive, { kind: "sphere" }>, report, aligned };
      };
      const [near, far] = [await fit(ZERO), await fit(FAR)];
      expect(off(far.sphere.center, near.sphere.center, FAR)).toBeLessThan(TOL);
      expect(far.sphere.radius).toBeCloseTo(near.sphere.radius, 6);
      expect(far.report.rmsError).toBeCloseTo(near.report.rmsError, 6);
      expect(off(far.report.residuals, near.report.residuals, ZERO)).toBeLessThan(TOL);
      expect(off(far.aligned.translation as number[], far.sphere.center.map((c) => -c), ZERO)).toBeLessThan(TOL);
    });
  });

  it("keeps a point cloud's origin, and reconstructs around it", async () => {
    const cloudAt = (at: Readonly<Vec3>): { text: string; positions: Float64Array; normals: Float64Array } => {
      const points = [...around(0.3, 24), ...around(0.2, 18), ...around(0.1, 9), [MID[0] + RADIUS, MID[1], MID[2]]];
      const normals = points.map((p) => p.map((c, k) => (c - MID[k]) / RADIUS));
      const moved = points.map((p) => p.map((c, k) => c + at[k]));
      return {
        text: moved.map((p, i) => [...p, ...normals[i]].join(" ")).join("\n"),
        positions: Float64Array.from(moved.flat()),
        normals: Float64Array.from(normals.flat()),
      };
    };
    const { text, positions, normals } = cloudAt(FAR);

    // Parsed around their centre, or packed around a given point.
    const centered = parsePointCloudText(text, "xyz", { center: true });
    const packed = packPointCloud(positions, normals, undefined, add(FAR, MID));
    expect(packed.origin).toEqual(add(FAR, MID));
    const bytes = new TextEncoder().encode(
      `ply\nformat ascii 1.0\nelement vertex ${centered.count}\n${["x", "y", "z", "nx", "ny", "nz"]
        .map((n) => `property double ${n}\n`)
        .join("")}end_header\n${text}\n`,
    );
    const ply = parsePointCloudBytes(bytes, { center: true });
    for (const cloud of [centered, packed, ply]) {
      const { positions: local, origin } = unpackPointCloud(cloud);
      expect(off(Float64Array.from(local, (c, k) => c + origin[k % 3]), positions, ZERO)).toBeLessThan(TOL);
    }
    // Without, they are f32 world positions, as ever.
    const plain = parsePointCloudText(text);
    expect(plain).not.toHaveProperty("origin");
    expect(unpackPointCloud(plain).origin).toEqual(ZERO);
    expect(unpackPointCloud(plain).positions).toEqual(Float32Array.from(positions));
    expect(packPointCloud(positions, normals)).toEqual(plain);
    expect(off(unpackPointCloud(plain).positions, positions, ZERO)).toBeGreaterThan(1e-3);

    // What makes a cloud of one keeps the origin.
    const bare: PackedPointCloud = packPointCloud(positions, undefined, undefined, add(FAR, MID));
    const eye = add(FAR, [9, 0, 0]);
    const made = [
      pointCloudSor(packed, 6, 3).cloud,
      pointCloudRor(packed, 0.5, 1).cloud,
      pointCloudVoxelDownsample(packed, 0.05),
      pointCloudEstimateNormals(bare, 8, eye),
    ];
    for (const cloud of made) expect(cloud.origin).toEqual(packed.origin);
    const toEye = unpackPointCloud(made[3]).normals!;
    for (let i = 0; i < toEye.length; i += 3) expect(toEye[i]).toBeGreaterThan(0);

    const near = cloudAt(ZERO);
    const nearCloud = parsePointCloudText(near.text, "xyz", { center: true });
    await onBoth(async (s) => {
      for (const method of ["implicitMlsMarchingCubes", "ballPivoting"] as const) {
        const want = await s.inspect.reconstructPointCloud(nearCloud, { method });
        const got = await s.inspect.reconstructPointCloud(centered, { method });
        await s.commitFrame();
        expect(s.origin(got)).toEqual(centered.origin);
        expect(off(worldPositions(s, got), worldPositions(s, want), FAR)).toBeLessThan(4 * TOL);
      }
      const hull = await s.convexHullOfPoints(packed);
      await s.commitFrame();
      const corners = worldPositions(s, hull.mesh);
      for (let i = 0; i < corners.length; i += 3) {
        let nearest = Infinity;
        for (let j = 0; j < positions.length; j += 3) {
          nearest = Math.min(nearest, off(corners.slice(i, i + 3), positions.slice(j, j + 3), ZERO));
        }
        expect(nearest).toBeLessThan(TOL);
      }
    });

    for (const origin of [[1, 2], [0, Number.NaN, 0], "x"]) {
      expect(() => packPointCloud(positions, undefined, undefined, origin as never)).toThrow(FastMeshError);
      expect(() => pointCloudVoxelDownsample({ ...packed, origin: origin as never }, 0.1)).toThrow(FastMeshError);
    }
    expect(() => parsePointCloudText(text, "xyz", { center: 1 as never })).toThrow(FastMeshError);
    expect(() => parsePointCloudBytes(bytes, null as never)).toThrow(FastMeshError);
  });

  it("trims, selects and traces paths on a mesh scaled unevenly", async () => {
    await onBoth(async (s) => {
      const run = async (at: Readonly<Vec3>) => {
        const mesh = await addBall(s, at);
        s.setPlacement(mesh, UNEVEN);
        await s.commitFrame();
        const world = (p: Readonly<Vec3>) => uneven(add(at, p));
        const center = world(around(0.2, 5)[1]);
        const faces = [
          await s.repair.selectFacesInSphere(mesh, center, 0.17),
          await s.repair.selectFacesInSphere(mesh, center, 0.43, { normal: [1, 0.1, 0.2], angleRad: 0.5 }),
          await s.repair.selectByCreaseAngle(mesh, 1200, 0.15),
          await s.repair.selectByCreaseAngle(mesh, 1200, 0.3),
          await s.repair.selectSmallFaceIslands(mesh, 1_000_000, 5),
          await s.repair.selectSmallFaceIslands(mesh, 1_000_000, 8),
        ].map((f) => Array.from(f));
        const path = await s.inspect.shortestSurfacePath(mesh, 4 * SEGMENTS + 3, 20 * SEGMENTS + 26, {
          method: "dijkstra",
        });
        const points = around(0.5, 12).map(world);
        const trim = await s.boolean.trimByCurve(mesh, { points, closed: true, deleteInside: true });
        await s.commitFrame();
        return { faces, path: path.flatMap((p) => p.position), trim, trimmed: worldPositions(s, mesh) };
      };
      const [near, far] = [await run(ZERO), await run(FAR)];
      const shift = add(uneven(FAR), uneven(ZERO).map((c) => -c) as Vec3);
      expect(far.faces).toEqual(near.faces);
      expect(near.faces[0].length).toBeGreaterThan(0);
      expect(near.path.length).toBeGreaterThan(60);
      expect(off(far.path, near.path, shift)).toBeLessThan(2 * TOL);
      expect(far.trim.seamEdges).toBe(near.trim.seamEdges);
      expect(far.trim.removedArea).toBeCloseTo(near.trim.removedArea, 5);
      expect(off(far.trimmed, near.trimmed, shift)).toBeLessThan(4 * TOL);
    });
  });

  it("stands a stamp far from its own zero on the hit point", async () => {
    await onBoth(async (s) => {
      const target = box([-1, -1, -1], [1, 1, 0]);
      const place = async (at: Readonly<Vec3>, stampAnchor?: Vec3) => {
        const on = s.addMesh(target.positions, target.indices);
        const { positions, indices } = box(add(at, [-0.1, -0.1, 0]), add(at, [0.1, 0.1, 0.3]));
        const stamp = s.addMesh(positions, indices);
        await s.commitFrame();
        const report = await s.boolean.placeMesh(on, stamp, {
          hitPoint: [0.2, 0.1, 0],
          hitNormal: [0, 0, 1],
          twistAngleRad: 0.4,
          ...(stampAnchor === undefined ? {} : { stampAnchor }),
        });
        await s.commitFrame();
        return { report, positions: worldPositions(s, report.added!), origin: s.origin(report.added!) };
      };
      const near = await place(ZERO);
      const far = await place(FAR);
      expect(off(far.positions, near.positions, ZERO)).toBeLessThan(TOL);
      expect(far.origin).toEqual([0.2, 0.1, 0]);
      const given = await place(FAR, add(FAR, [0, 0, 0.1]));
      expect(off(given.positions, near.positions, [0, 0, -0.1])).toBeLessThan(TOL);
      // The zero of a far stamp, when that is what is asked for.
      const zero = await place(FAR, ZERO);
      const t = zero.report.transform;
      const moved = [0, 1, 2].map((r) => t.rotation.slice(3 * r, 3 * r + 3).reduce((m, c, k) => m + c * FAR[k], 0)) as Vec3;
      expect(off(zero.positions, near.positions, moved)).toBeLessThan(TOL);
      const bad = async () =>
        s.boolean.placeMesh(far.report.added!, near.report.added!, {
          hitPoint: ZERO,
          hitNormal: [0, 0, 1],
          stampAnchor: [1, Number.NaN, 0],
        });
      await expect(bad()).rejects.toThrow(FastMeshError);
    });
  });
});
