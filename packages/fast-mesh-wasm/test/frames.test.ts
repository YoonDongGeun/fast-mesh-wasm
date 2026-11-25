// One world, many ways to store it (#289): the same world shapes, put into the
// scene through mixes of mesh origin, pose and scale, near the world origin and
// far from it, give the same world answers, through the bindings too. Every
// case runs on a Scene and on a WorkerScene. (The engine's own sweep is
// crates/fast-mesh-scene/tests/frames.)
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it } from "vitest";
import * as THREE from "three";
import {
  type FieldHandle,
  type MeshHandle,
  type PlacementInput,
  Scene,
  type Vec3,
} from "../src/node.js";
import { bindMesh, bindSection } from "../src/three/index.js";
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
interface Shape {
  points: Vec3[];
  triangles: number[][];
}

const NEAR: Vec3 = [10, -4, 6];
const FAR: Vec3 = [1.0e5, -2.5e5, 7.3e4];

const add = (a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: Readonly<Vec3>, b: Readonly<Vec3>): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const unit = (a: Readonly<Vec3>): Vec3 => a.map((c) => c / Math.hypot(...a)) as Vec3;

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];
/** A rotation about z, then about x, row-major. */
const TURNED = ((z: number, x: number) => {
  const [sz, cz, sx, cx] = [Math.sin(z), Math.cos(z), Math.sin(x), Math.cos(x)];
  return [cz, -sz, 0, cx * sz, cx * cz, -sx, sx * sz, sx * cz, cx];
})(0.4, -0.7);

/** A closed lump of about 2.4 x 1.8 x 2, without symmetries. */
function lump(): Shape {
  const [rings, n] = [9, 14];
  const points: Vec3[] = [];
  const bend = (p: Vec3): Vec3 => {
    const f = 1 + 0.13 * Math.sin(2.3 * p[0] + 0.4) + 0.09 * Math.cos(1.7 * p[1] - 1.1 * p[2]);
    return [1.15 * f * p[0], 0.85 * f * p[1], f * p[2] + 0.07 * p[0]];
  };
  points.push(bend([0, 0, -1]));
  for (let i = 1; i <= rings; i++) {
    const t = (Math.PI * i) / (rings + 1);
    for (let j = 0; j < n; j++) {
      const u = (2 * Math.PI * j) / n;
      points.push(bend([Math.sin(t) * Math.cos(u), Math.sin(t) * Math.sin(u), -Math.cos(t)]));
    }
  }
  points.push(bend([0, 0, 1]));
  const last = points.length - 1;
  const at = (r: number, j: number) => 1 + r * n + (j % n);
  const triangles: number[][] = [];
  for (let j = 0; j < n; j++) {
    triangles.push([0, at(0, j + 1), at(0, j)]);
    for (let r = 0; r < rings - 1; r++) {
      triangles.push([at(r, j), at(r, j + 1), at(r + 1, j + 1)], [at(r, j), at(r + 1, j + 1), at(r + 1, j)]);
    }
    triangles.push([last, at(rings - 1, j), at(rings - 1, j + 1)]);
  }
  return { points, triangles };
}

function block(lo: Vec3, hi: Vec3): Shape {
  const points = Array.from({ length: 8 }, (_, i) => [0, 1, 2].map((k) => ((i >> k) & 1 ? hi[k] : lo[k])) as Vec3);
  const triangles = [
    [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5], [0, 1, 5], [0, 5, 4],
    [2, 6, 7], [2, 7, 3], [0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6],
  ];
  return { points, triangles };
}

const LUMP = lump();
const CROSSING = block([0.6, -0.31, -0.42], [2.1, 0.53, 0.37]);
const APART: Shape = { ...LUMP, points: LUMP.points.map((p) => add(p, [3.4, 0.37, -0.21])) };

/** How a world shape is put into the scene: `origin` the mesh's own centre, or the placement takes it there. */
interface Put {
  scale: Vec3;
  turned: boolean;
  centred: boolean;
  at: Vec3;
}

const PLAIN: Put = { scale: [1, 1, 1], turned: false, centred: false, at: [0, 0, 0] };
const PUTS: Put[] = [NEAR, FAR].flatMap((at) => [
  { scale: [1, 1, 1], turned: true, centred: true, at },
  { scale: [0.25, 0.25, 0.25], turned: true, centred: false, at },
  { scale: [2, 1.5, 0.5], turned: false, centred: true, at },
  { scale: [2, 1.5, 0.5], turned: true, centred: false, at },
]);

function putMesh(s: AnyScene, put: Put, shape: Shape): MeshHandle {
  const r = put.turned ? TURNED : IDENTITY;
  const back = (d: Vec3): Vec3 =>
    [0, 1, 2].map((k) => (r[k] * d[0] + r[3 + k] * d[1] + r[6 + k] * d[2]) / put.scale[k]) as Vec3;
  const lo = [0, 1, 2].map((k) => Math.min(...shape.points.map((p) => p[k])));
  const hi = [0, 1, 2].map((k) => Math.max(...shape.points.map((p) => p[k])));
  const middle = add(put.at, lo.map((c, k) => (c + hi[k]) / 2) as Vec3);
  const translation: Vec3 = put.centred ? [0, 0, 0] : [...put.at];
  const origin: Vec3 = put.centred ? back(middle) : [0, 0, 0];
  const positions = shape.points.flatMap((p) => back(sub(add(put.at, p), translation)));
  const mesh = s.addMesh(Float64Array.from(positions), Uint32Array.from(shape.triangles.flat()), { origin });
  const placement: PlacementInput = { rotation: r, translation, scale: [...put.scale] };
  s.setPlacement(mesh, placement);
  return mesh;
}

/** The largest difference of two flat results (infinite when they do not match at all). */
function off(got: readonly number[], want: readonly number[]): number {
  if (got.length !== want.length) return Number.POSITIVE_INFINITY;
  let m = 0;
  for (let i = 0; i < got.length; i++) {
    if (Object.is(got[i], want[i]) || got[i] === want[i]) continue;
    const d = Math.abs(got[i] - want[i]);
    m = Number.isNaN(d) ? Number.POSITIVE_INFINITY : Math.max(m, d);
  }
  return m;
}

const asNumber = (b: boolean): number => (b ? 1 : 0);
const numbers = (a: ArrayLike<number>): number[] => Array.from(a);

/** The mesh's vertices in the model, through its `matrix4`, flat. */
function modelPositions(s: AnyScene, mesh: MeshHandle, put: Put): number[] {
  const p = s.views(mesh).positions();
  const m = s.matrix4(mesh);
  const out: number[] = [];
  for (let i = 0; i + 2 < p.length; i += 3) {
    for (let r = 0; r < 3; r++) {
      out.push(m[r] * p[i] + m[4 + r] * p[i + 1] + m[8 + r] * p[i + 2] + (m[12 + r] - put.at[r]));
    }
  }
  return out;
}

/** Volume, area and bounds of the mesh in the model. */
function solid(s: AnyScene, mesh: MeshHandle, put: Put): number[] {
  const p = modelPositions(s, mesh, put);
  const t = s.views(mesh).indices();
  let [volume, area] = [0, 0];
  const [lo, hi] = [[Infinity, Infinity, Infinity], [-Infinity, -Infinity, -Infinity]];
  for (let k = 0; k + 2 < t.length; k += 3) {
    if (t[k] === 0 && t[k + 1] === 0 && t[k + 2] === 0) continue;
    const [a, b, c] = [0, 1, 2].map((j) => p.slice(3 * t[k + j], 3 * t[k + j] + 3) as Vec3);
    const n = cross(sub(b, a), sub(c, a));
    area += Math.hypot(...n) / 2;
    volume += dot(a, cross(b, c)) / 6;
    for (const q of [a, b, c]) {
      for (let j = 0; j < 3; j++) {
        lo[j] = Math.min(lo[j], q[j]);
        hi[j] = Math.max(hi[j], q[j]);
      }
    }
  }
  return [volume, area, ...lo, ...hi];
}

/** Lets what was asked so far land, on the Worker too. */
async function settle(s: AnyScene): Promise<void> {
  for (let k = 0; k < 2; k++) {
    const barrier = s.inspect.raycast([0, 0, 1e7], [0, 0, 1]);
    s.commitFrame();
    await barrier;
    s.commitFrame();
  }
}

/** A camera at `eye` looking at `to`, as the viewer's matrices have it. */
function cameraAt(eye: Vec3, to: Vec3) {
  const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 50);
  camera.up.set(0, 0, 1);
  camera.position.set(...eye);
  camera.lookAt(...to);
  camera.updateMatrixWorld();
  camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
  const viewProj = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  return { viewProj: viewProj.toArray(), cameraOrigin: eye, cameraForward: sub(to, eye) };
}

describe("one world, however it is stored (#289)", { timeout: 60_000 }, () => {
  const cleanups: (() => void)[] = [];

  afterEach(() => {
    for (const c of cleanups.splice(0)) c();
  });

  async function scenes(): Promise<AnyScene[]> {
    const scene = new Scene();
    const worker = await createWorkerScene({ createWorker: () => new NodeWorker() });
    cleanups.push(() => scene.free(), () => worker.terminate());
    return [scene, worker];
  }

  /** Fails unless `run` gives the same numbers for every put as for the plain one, on both kinds of scene. */
  async function agree(tol: number, run: (s: AnyScene, put: Put) => Promise<number[]>): Promise<void> {
    for (const s of await scenes()) {
      const want = await run(s, PLAIN);
      expect(want.length).toBeGreaterThan(0);
      for (const put of PUTS) {
        const d = off(await run(s, put), want);
        expect(d, `${s.constructor.name} ${JSON.stringify(put)}`).toBeLessThanOrEqual(tol);
      }
    }
  }

  /** Runs `body` with fresh meshes of `shapes`, removed again after it. */
  async function withMeshes(
    s: AnyScene,
    shapes: readonly (readonly [Put, Shape])[],
    body: (...meshes: MeshHandle[]) => Promise<number[]>,
  ): Promise<number[]> {
    const meshes = shapes.map(([put, shape]) => putMesh(s, put, shape));
    await settle(s);
    const out = await body(...meshes);
    for (const m of meshes) if (s.has(m)) s.removeMesh(m);
    await settle(s);
    return out;
  }

  /** The second mesh of a pair: put another way than the first. */
  const other = (put: Put): Put =>
    put === PLAIN ? PLAIN : { scale: [0.5, 1, 1.5], turned: !put.turned, centred: !put.centred, at: put.at };

  it("queries", async () => {
    await agree(1e-5, (s, put) =>
      withMeshes(s, [[put, LUMP]], async (mesh) => {
        const out: number[] = [];
        for (const p of [[2.9, 0.4, 0.3], [0.2, -2.6, 0.7], [0.3, 0.1, 0.4]] as Vec3[]) {
          const w = add(put.at, p);
          const ray = await s.inspect.raycast(w, sub([0.1, -0.05, 0.02], p));
          out.push(ray!.distance, ray!.triangle, ...sub(ray!.point, put.at), ...ray!.barycentric);
          const hit = await s.inspect.closestPoint(w);
          out.push(hit!.distance, ...sub(hit!.point, put.at));
        }
        const m = await s.inspect.measure(mesh);
        out.push(m.area, m.volume!);
        if (s instanceof Scene) {
          const brush = s.select(mesh, {
            kind: "brush",
            center: add(put.at, [0.9, 0.3, 0.2]),
            radius: 0.71,
            space: "world",
          });
          out.push(...numbers(brush.vertices));
        }
        return out;
      }),
    );
  });

  it("measures of a mesh", async () => {
    await agree(1e-3, (s, put) =>
      withMeshes(s, [[put, LUMP]], async (mesh) => {
        const over = await s.inspect.overhangs(mesh, { buildDirection: [0.3, -0.5, 0.8], criticalAngleDeg: 37 });
        const quality = await s.inspect.triangleQuality(mesh, "minAngleDeg");
        const geodesic = await s.inspect.geodesicDistances(mesh, 17);
        return [
          over.totalSupportArea,
          over.overhangFaceCount,
          ...numbers(over.vertexAnglesDeg),
          quality.min,
          quality.mean,
          ...numbers(quality.vertexQuality),
          ...numbers(geodesic),
        ];
      }),
    );
  });

  it("pairs", async () => {
    await agree(1e-5, (s, put) =>
      withMeshes(
        s,
        [
          [put, LUMP],
          [other(put), CROSSING],
          [other(put), APART],
        ],
        async (a, hit, far) => {
          const out: number[] = [];
          for (const [x, y] of [[a, far], [far, a]]) {
            const d = (await s.inspect.meshDistance(x, y))!;
            out.push(d.distance, ...sub(d.a, put.at), ...sub(d.b, put.at));
          }
          for (const [x, y] of [[a, hit], [hit, a], [a, far]]) out.push(asNumber(await s.inspect.intersects(x, y)));
          const contacts = await s.inspect.findContacts({ margin: 0.4, pairs: [[a, hit], [a, far]] });
          // (Which mesh of a pair comes first is that of the lower slot.)
          expect(contacts.length).toBe(1);
          for (const c of contacts) {
            const [own, others] = c.a === a ? [c.trianglesA, c.trianglesB] : [c.trianglesB, c.trianglesA];
            out.push(asNumber(c.intersecting), c.distance, ...numbers(own), ...numbers(others));
          }
          const deviation = await s.inspect.meshDeviation(a, hit, { signed: true });
          out.push(deviation.min, deviation.max, deviation.mean, deviation.rms, ...numbers(deviation.vertexDeviations));
          return out;
        },
      ),
    );
  });

  it("fields", async () => {
    await agree(1e-4, (s, put) =>
      withMeshes(
        s,
        [
          [put, LUMP],
          [other(put), CROSSING],
          [put, APART],
        ],
        async (a, hit, far) => {
          const maxDistance = Number.POSITIVE_INFINITY;
          const fields: FieldHandle[] = [s.inspect.addDistanceField({ on: a, to: [hit, far], name: "d", maxDistance })];
          if (s instanceof Scene) {
            // These two are the Scene's alone.
            fields.push(s.addThicknessField({ on: a, name: "t", maxDistance }), s.addCurvatureField({ on: a, name: "c" }));
          }
          await settle(s);
          const out: number[] = [];
          for (const f of fields) out.push(...numbers(await s.inspect.fieldValues(f)));
          out.push(...numbers(await s.inspect.fieldNearest(fields[0])));
          for (const f of fields) s.inspect.removeField(f);
          return out;
        },
      ),
    );
  });

  it("sections, with f64 points", async () => {
    const normal = unit([0.3, -0.5, 0.8]);
    await agree(1e-5, (s, put) =>
      withMeshes(
        s,
        [
          [put, LUMP],
          [other(put), APART],
        ],
        async (a, b) => {
          const origin = add(put.at, [0.1, 0.05, -0.2]);
          const section = s.inspect.addSection({ plane: { origin, normal }, meshes: [a, b], fill: true });
          s.inspect.updateSections();
          await settle(s);
          const out: number[] = [];
          for (const mesh of [a, b]) {
            const r = s.inspect.section(section, mesh);
            const c = r.contours[0];
            expect(c.points64).toBeInstanceOf(Float64Array);
            expect(c.points).toEqual(Float32Array.from(c.points64!));
            const [lo, hi] = [[Infinity, Infinity, Infinity], [-Infinity, -Infinity, -Infinity]];
            let offPlane = 0;
            for (const points of [c.points64!, r.fill!.positions64!]) {
              for (let i = 0; i < points.length; i += 3) {
                const p = sub([points[i], points[i + 1], points[i + 2]], put.at);
                offPlane = Math.max(offPlane, Math.abs(dot(sub(p, [0.1, 0.05, -0.2]), normal)));
                for (let k = 0; k < 3; k++) {
                  lo[k] = Math.min(lo[k], p[k]);
                  hi[k] = Math.max(hi[k], p[k]);
                }
              }
            }
            expect(offPlane).toBeLessThan(1e-6);
            out.push(r.length, r.area, ...lo, ...hi);
          }
          const gap = (await s.inspect.sectionGap(section, a, b))!;
          out.push(gap.distance, ...sub(gap.a, put.at), ...sub(gap.b, put.at));
          s.inspect.removeSection(section);
          return out;
        },
      ),
    );
  });

  it("selections, through a camera in f64", async () => {
    const eye: Vec3 = [4, -3, 2.5];
    await agree(0, (s, put) =>
      withMeshes(s, [[put, LUMP]], async (mesh) => {
        const camera = cameraAt(add(put.at, eye), add(put.at, [0.1, 0, 0]));
        const out: number[] = [];
        for (const visibleOnly of [false, true]) {
          const faces = await s.repair.selectFacesScreen(
            mesh,
            { kind: "rect", minNdc: [-0.31, -0.27], maxNdc: [0.43, 0.39] },
            { ...camera, visibleOnly, rejectBackfaces: visibleOnly },
          );
          out.push(-1, ...numbers(faces));
        }
        out.push(-1, ...numbers(await s.repair.selectFacesInSphere(mesh, add(put.at, [1, 0.3, 0.2]), 0.61)));
        return out;
      }),
    );
  });

  it("edits, cuts, booleans and placed meshes", async () => {
    await agree(3e-5, (s, put) =>
      withMeshes(
        s,
        [
          [put, LUMP],
          [other(put), CROSSING],
          [{ ...put, at: [0, 0, 0], turned: false, centred: false }, block([-0.2, -0.1, -0.1], [0.35, 0.1, 0.2])],
          [put, block([-1, -0.8, -0.7], [1.2, 0.9, 0.7])],
        ],
        async (a, b, stamp, table) => {
          const out: number[] = [];
          // Positions are the world's.
          await s.edit(a, [{ op: "setPositions", vertices: [40], positions: add(put.at, [1.4, 0.3, 0.1]) }]);
          await settle(s);
          out.push(...modelPositions(s, a, put));
          await s.undo();
          await settle(s);
          await s.boolean(a, b, "subtract");
          await settle(s);
          out.push(...solid(s, a, put));
          await s.undo();
          await settle(s);
          out.push(...solid(s, a, put));
          const cut = await s.boolean.cut(a, { origin: add(put.at, [0.1, 0.05, -0.2]), normal: [0.3, -0.5, 0.8] });
          await settle(s);
          out.push(...solid(s, a, put), ...solid(s, cut.removed, put));
          s.removeMesh(cut.removed);
          for (const mode of ["appendShell", "embedUnion"] as const) {
            const r = await s.boolean.placeMesh(table, stamp, {
              hitPoint: add(put.at, [0.1, 0.05, 0.7]),
              hitNormal: [0.2, -0.1, 0.97],
              twistAngleRad: 0.4,
              embedDepth: 0.03,
              mode,
            });
            await settle(s);
            out.push(...solid(s, table, put), ...sub(r.transform.translation as Vec3, put.at));
            await s.undo();
            await settle(s);
          }
          return out;
        },
      ),
    );
  });

  it("collisions with a soft response", async () => {
    await agree(3e-4, (s, put) =>
      withMeshes(
        s,
        [
          [other(put), LUMP],
          [put, { ...LUMP, points: LUMP.points.map((p) => add(p, [0.1, 0.05, 2.1])) }],
        ],
        async (still, moving) => {
          s.setCollisionRule(still, moving, { mode: "softDeform", role: "movingIsCutter", contactMargin: 0.01 });
          const p = s.placement(moving);
          const r = await s.moveWithCollision(moving, { ...p, translation: add(p.translation, [0, 0, -0.35]) });
          await settle(s);
          expect(r.softVerticesDeformed).toBeGreaterThan(0);
          const out = [r.softVerticesDeformed, ...modelPositions(s, still, other(put))];
          await s.undo();
          await settle(s);
          out.push(...modelPositions(s, still, other(put)));
          return out;
        },
      ),
    );
  });

  it("a far section is drawn as offsets from near it, a near one at its coordinates", () => {
    for (const put of [PLAIN, ...PUTS]) {
      const s = new Scene();
      cleanups.push(() => s.free());
      const mesh = putMesh(s, put, LUMP);
      const bound = bindMesh(s, mesh);
      const origin = add(put.at, [0.1, 0.05, -0.2]);
      const section = s.addSection({ plane: { origin, normal: [0.3, -0.5, 0.8] }, meshes: [mesh], fill: true });
      s.updateSections();
      s.commitFrame();
      const drawn = bindSection(s, section, [bound]);
      drawn.group.updateMatrixWorld(true);
      const result = s.section(section, mesh);
      const wanted = [result.contours[0].points64!, result.fill!.positions64!];
      for (const [i, object] of (drawn.group.children as THREE.Mesh[]).entries()) {
        const position = object.geometry.getAttribute("position") as THREE.BufferAttribute;
        const [v, from] = [new THREE.Vector3(), new THREE.Vector3()];
        from.setFromMatrixPosition(object.matrixWorld);
        expect(from.lengthSq() === 0).toBe(put.at !== FAR);
        v.fromBufferAttribute(position, 0);
        expect(Math.max(...v.toArray().map(Math.abs))).toBeLessThan(600);
        v.applyMatrix4(object.matrixWorld);
        expect(Math.max(...sub(v.toArray(), numbers(wanted[i].slice(0, 3)) as Vec3).map(Math.abs))).toBeLessThan(5e-5);
      }
      drawn.dispose();
      bound.dispose();
    }
  });
});
