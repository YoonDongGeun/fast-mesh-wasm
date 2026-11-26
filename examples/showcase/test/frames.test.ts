// One world, many ways to store it (#289): what the tools make of a mesh in the
// world and back does not depend on the mix of mesh origin, pose and scale that
// puts it there, near the world origin or far from it. (The engine's own sweep
// is crates/fast-mesh-scene/tests/frames, the SDK's frames.test.ts.)
import { type MeshHandle, type Placement, Scene, type Vec3 } from "fast-mesh-wasm";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import * as THREE from "three";
import { afterEach, describe, expect, it } from "vitest";
import { AppState, type Item } from "../src/app/state/app-state";
import { mirrorMesh, runIcpAlign } from "../src/engine/workbench-ops";
import { isoLineSegments } from "../src/features/inspect/ux/geodesic-map";
import { fitWorldBox } from "../src/features/inspect/ux/section-clip";
import { legFromSurfacePath, pickFromHit } from "../src/features/inspect/ux/surface-paths";
import { computeHitPointAndWorldNormal } from "../src/features/mesh-edit/ux/surface-orientation";
import { worldExtent } from "../src/features/mesh-repair/ux/wrap";
import { keepSection } from "../src/features/polylines/ux/polyline-actions";
import { grabLocalDelta } from "../src/features/sculpt/ux/sculpt";
import { buildScreenSelectOptions } from "../src/features/select/ux/screen-select";
import { landmarkPairs, landmarkResiduals, planeToLocal, planeToWorld } from "../src/features/transform/ux/align-tools";
import { applyTransform, readPlacedMesh } from "../src/features/transform/ux/placement-helpers";
import { keptTranslation, moveToOrigin, worldBounds } from "../src/features/transform/ux/placement-math";
import { placementToValues, scaleFactor, valuesToPlacement } from "../src/features/transform/ux/transform-math";
import { drawOrigin } from "../src/shared/ux/draw-origin";
import { worldCenter } from "../src/shared/ux/mesh-center";

const cleanups: (() => void)[] = [];
afterEach(() => {
  while (cleanups.length > 0) {
    cleanups.pop()?.();
  }
});

const NEAR: Vec3 = [10, -4, 6];
const FAR: Vec3 = [1.0e5, -2.5e5, 7.3e4];

const add = (a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: Readonly<Vec3>, b: Readonly<Vec3>): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];
/** A rotation about z, then about x, row-major. */
const TURNED = ((z: number, x: number) => {
  const [sz, cz, sx, cx] = [Math.sin(z), Math.cos(z), Math.sin(x), Math.cos(x)];
  return [cz, -sz, 0, cx * sz, cx * cz, -sx, sx * sz, sx * cz, cx];
})(0.4, -0.7);

/** How a world shape is put into the scene: `centred` about its own middle, or the placement takes it there. */
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

const RINGS = 12;
const SEGMENTS = 20;

/** A closed lump about zero, without symmetries. */
function lump(): { points: Vec3[]; indices: number[] } {
  const points: Vec3[] = [];
  const bend = (p: Vec3): Vec3 => {
    const f = 1 + 0.13 * Math.sin(2.3 * p[0] + 0.4) + 0.09 * Math.cos(1.7 * p[1] - 1.1 * p[2]);
    return [1.15 * f * p[0], 0.85 * f * p[1], f * p[2] + 0.07 * p[0]];
  };
  points.push(bend([0, 0, -1]));
  for (let i = 1; i <= RINGS; i++) {
    const t = (Math.PI * i) / (RINGS + 1);
    for (let j = 0; j < SEGMENTS; j++) {
      const u = (2 * Math.PI * j) / SEGMENTS;
      points.push(bend([Math.sin(t) * Math.cos(u), Math.sin(t) * Math.sin(u), -Math.cos(t)]));
    }
  }
  points.push(bend([0, 0, 1]));
  const last = points.length - 1;
  const at = (r: number, j: number) => 1 + r * SEGMENTS + (j % SEGMENTS);
  const indices: number[] = [];
  for (let j = 0; j < SEGMENTS; j++) {
    indices.push(0, at(0, j + 1), at(0, j), last, at(RINGS - 1, j), at(RINGS - 1, j + 1));
    for (let r = 0; r < RINGS - 1; r++) {
      indices.push(at(r, j), at(r, j + 1), at(r + 1, j + 1), at(r, j), at(r + 1, j + 1), at(r + 1, j));
    }
  }
  return { points, indices };
}

const LUMP = lump();
const MIDDLE = [0, 1, 2].map((k) => {
  const c = LUMP.points.map((p) => p[k]);
  return (Math.min(...c) + Math.max(...c)) / 2;
}) as Vec3;

interface Harness {
  scene: Scene;
  store: SceneStore;
  app: AppState;
}

function open(): Harness {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  cleanups.push(app.follow(store));
  cleanups.push(() => {
    store.dispose();
    scene.free();
  });
  return { scene, store, app };
}

/** The lump (moved by `by` in the world) as `put` puts it. */
function putLump({ store }: Harness, put: Put, by: Vec3 = [0, 0, 0]): MeshHandle {
  const r = put.turned ? TURNED : IDENTITY;
  const at = add(put.at, by);
  const back = (d: Vec3): Vec3 =>
    [0, 1, 2].map((k) => (r[k] * d[0] + r[3 + k] * d[1] + r[6 + k] * d[2]) / put.scale[k]) as Vec3;
  const translation: Vec3 = put.centred ? [0, 0, 0] : [...at];
  const origin: Vec3 = put.centred ? back(add(at, MIDDLE)) : [0, 0, 0];
  const positions = LUMP.points.flatMap((p) => back(sub(add(at, p), translation)));
  const mesh = store.addMesh(Float64Array.from(positions), Uint32Array.from(LUMP.indices), { origin });
  store.setPlacement(mesh, { rotation: [...r] as never, translation, scale: [...put.scale] });
  store.commitFrame();
  return mesh;
}

/** The mesh's vertices in the model, flat. */
function model(scene: Scene, mesh: MeshHandle, at: Vec3): number[] {
  const p = scene.views(mesh).positions();
  const m = scene.matrix4(mesh);
  const out: number[] = [];
  for (let i = 0; i + 2 < p.length; i += 3) {
    for (let r = 0; r < 3; r++) {
      out.push(m[r] * p[i] + m[4 + r] * p[i + 1] + m[8 + r] * p[i + 2] + (m[12 + r] - at[r]));
    }
  }
  return out;
}

function off(got: ArrayLike<number>, want: ArrayLike<number>): number {
  expect(got.length).toBe(want.length);
  let m = 0;
  for (let i = 0; i < got.length; i++) m = Math.max(m, Math.abs(got[i] - want[i]));
  return m;
}

function agree(tol: number, run: (h: Harness, put: Put) => number[], puts: readonly Put[] = PUTS): void {
  const want = run(open(), PLAIN);
  expect(want.length).toBeGreaterThan(0);
  for (const put of puts) {
    expect(off(run(open(), put), want), JSON.stringify(put)).toBeLessThanOrEqual(tol);
  }
}

const itemOf = (mesh: MeshHandle): Item => ({ mesh, radius: 1.5, center: [0, 0, 0] }) as unknown as Item;

describe("one world, however it is stored (#289)", () => {
  it("picks, their normals, boxes and extents", () => {
    agree(2e-6, (h, put) => {
      const mesh = putLump(h, put);
      const { scene } = h;
      const hit = scene.raycast(add(put.at, [2.9, 0.4, 0.3]), [-1, -0.15, -0.1])!;
      const oriented = computeHitPointAndWorldNormal(scene, hit)!;
      const pick = pickFromHit(scene, hit);
      const path = scene.inspect.shortestSurfacePath(mesh, 17, 200) as { position: Vec3; triangle: number; barycentric: Vec3 }[];
      const leg = legFromSurfacePath(scene, mesh, path);
      const box = worldBounds(readPlacedMesh(scene, mesh))!;
      return [
        ...sub(oriented.point, put.at),
        ...oriented.normal,
        ...sub(pick.point as Vec3, put.at),
        leg.length,
        ...leg.normals.slice(0, 9),
        worldExtent(scene, [itemOf(mesh)]) > 2 ? 1 : 0,
        ...sub(box.min, put.at),
        ...sub(box.max, put.at),
      ];
    });
  });

  it("the clip box holds the ball of the mesh's radius, turned and stretched", () => {
    for (const put of PUTS) {
      const h = open();
      const mesh = putLump(h, put);
      const item = { ...itemOf(mesh), radius: 2 } as Item;
      const { min, max } = fitWorldBox(h.scene, item)!;
      const m = h.scene.matrix4(mesh);
      const c = worldCenter(h.scene, mesh);
      // The farthest the matrix takes a point of the unit ball along each axis.
      const reach = [0, 0, 0];
      for (let i = 0; i < 4000; i++) {
        const z = 1 - (2 * i + 1) / 4000;
        const [s, t] = [Math.sqrt(1 - z * z), i * 2.399963];
        const p = [s * Math.cos(t), s * Math.sin(t), z];
        for (let k = 0; k < 3; k++) reach[k] = Math.max(reach[k], m[k] * p[0] + m[4 + k] * p[1] + m[8 + k] * p[2]);
      }
      for (let k = 0; k < 3; k++) {
        expect(max[k] - c[k]).toBeGreaterThanOrEqual(reach[k] - 1e-9);
        expect(max[k] - c[k]).toBeLessThan(reach[k] * 1.01);
        expect(c[k] - min[k]).toBeCloseTo(max[k] - c[k], 6);
      }
    }
  });

  it("geodesic iso-lines are offsets from near the mesh", () => {
    const lines = (h: Harness, put: Put): { around: Vec3; world: Float64Array } => {
      const mesh = putLump(h, put);
      const { scene } = h;
      const values = scene.inspect.geodesicDistances(mesh, 17) as Float32Array;
      const views = scene.views(mesh);
      const m = scene.matrix4(mesh);
      const around = drawOrigin([m[12], m[13], m[14]]);
      const segments = isoLineSegments(views.positions(), views.indices(), m, values, 0.37, 0.01, around);
      expect(Math.max(...Array.from(segments, Math.abs))).toBeLessThan(700);
      return { around, world: Float64Array.from(segments, (c, i) => c + (around[i % 3] - put.at[i % 3])) };
    };
    const want = lines(open(), PLAIN);
    expect(want.around).toEqual([0, 0, 0]);
    expect(want.world.length).toBeGreaterThan(100);
    for (const put of PUTS) {
      const got = lines(open(), put);
      expect(got.around.some((c) => c !== 0)).toBe(put.at === FAR);
      // Offsets of a few hundred are f32 to 1.5e-5.
      expect(off(got.world, want.world)).toBeLessThan(8e-4);
    }
  });

  it("section contours are kept in f64", () => {
    const length = (h: Harness, put: Put) => {
      const mesh = putLump(h, put);
      const plane = { origin: add(put.at, [0.1, 0.05, -0.2]), normal: [0.3, -0.5, 0.8] as Vec3 };
      const handle = h.store.addSection({ plane, meshes: [mesh], fill: false });
      h.scene.updateSections();
      h.store.commitFrame();
      const [curve] = keepSection(h.app, h.store, h.scene, { handle, plane });
      const n = Math.hypot(...plane.normal);
      let offPlane = 0;
      for (let i = 0; i < curve.points.length; i += 3) {
        const d = [0, 1, 2].map((k) => curve.points[i + k] - plane.origin[k]);
        offPlane = Math.max(offPlane, Math.abs((d[0] * 0.3 - d[1] * 0.5 + d[2] * 0.8) / n));
      }
      expect(offPlane).toBeLessThan(1e-6);
      const lo = [0, 1, 2].map((k) => Math.min(...curve.points.filter((_, i) => i % 3 === k)) - put.at[k]);
      const hi = [0, 1, 2].map((k) => Math.max(...curve.points.filter((_, i) => i % 3 === k)) - put.at[k]);
      return [...lo, ...hi];
    };
    agree(2e-6, length);
  });

  it("a screen selection through a camera far out", () => {
    const faces = (h: Harness, put: Put) => {
      const mesh = putLump(h, put);
      const camera = new THREE.PerspectiveCamera(45, 1.3, 0.1, 50);
      camera.position.set(...add(put.at, [4, -3, 2.5]));
      camera.lookAt(...add(put.at, [0.1, 0, 0]));
      const options = buildScreenSelectOptions(camera, true, true);
      const shape = { kind: "rect", minNdc: [-0.31, -0.27], maxNdc: [0.43, 0.39] } as const;
      const got = h.scene.selectFacesScreen(mesh, shape, options);
      expect(got.length).toBeGreaterThan(10);
      return Array.from(got);
    };
    agree(0, faces);
  });

  it("landmarks and planes go to a mesh and back", () => {
    agree(2e-6, (h, put) => {
      const a = putLump(h, put);
      const b = putLump(h, PLAIN);
      const m = h.scene.matrix4(a);
      const point = add(put.at, [0.3, -0.2, 0.5]);
      const normal: Vec3 = [0.6, 0, 0.8];
      const local = planeToLocal(m, point, normal);
      const world = planeToWorld(m, local.point, local.normal);
      const p = h.scene.views(a).positions();
      const set = {
        moving: a,
        fixed: b,
        movingPoints: [[p[30], p[31], p[32]] as Vec3],
        fixedPoints: [LUMP.points[10]],
      } as unknown as Parameters<typeof landmarkPairs>[0];
      const [pair] = landmarkPairs(set, m, h.scene.matrix4(b));
      const [residual] = landmarkResiduals(set, m, h.scene.matrix4(b));
      expect(residual).toBeCloseTo(Math.hypot(...sub(pair.source as Vec3, pair.target as Vec3)), 9);
      return [...sub(world.point, put.at), ...world.normal, ...sub(pair.source as Vec3, put.at), ...(pair.target as Vec3)];
    });
  });

  it("mirrors across a world plane and moves to the origin", async () => {
    const run = async (put: Put) => {
      const h = open();
      const mesh = putLump(h, put);
      await mirrorMesh(h.store, mesh, { point: add(put.at, [0.2, 0.1, 0]), normal: [0.6, 0, 0.8] });
      h.store.commitFrame();
      const out = model(h.scene, mesh, put.at);
      h.store.setPlacement(mesh, moveToOrigin(readPlacedMesh(h.scene, mesh), "center"));
      h.store.commitFrame();
      return [...out, ...model(h.scene, mesh, [0, 0, 0])];
    };
    const want = await run(PLAIN);
    for (const put of PUTS) expect(off(await run(put), want), JSON.stringify(put)).toBeLessThan(3e-6);
  });
});

describe("a placement's scale (#289)", () => {
  const SCALED: Placement = { rotation: TURNED as never, translation: [3, -2, 1], scale: [2, 1.5, 0.5] };

  it("a grab follows the pointer on a scaled mesh", () => {
    const object = new THREE.Object3D();
    object.quaternion.setFromEuler(new THREE.Euler(0.3, -0.5, 0.8));
    object.scale.set(2, 1.5, 0.5);
    object.updateMatrixWorld();
    const q = object.getWorldQuaternion(new THREE.Quaternion()).invert();
    const [right, up] = [[0.8, 0, -0.6] as const, [0, 1, 0] as const];
    const local = grabLocalDelta(30, -20, 0.01, right, up, [q.x, q.y, q.z, q.w], [2, 1.5, 0.5]);
    const moved = object.localToWorld(new THREE.Vector3(...local)).sub(object.localToWorld(new THREE.Vector3()));
    expect(off(moved.toArray(), [0.24, 0.2, -0.18])).toBeLessThan(1e-12);
    expect(grabLocalDelta(30, -20, 0.01, right, up, [q.x, q.y, q.z, q.w])).toEqual(
      grabLocalDelta(30, -20, 0.01, right, up, [q.x, q.y, q.z, q.w], [1, 1, 1]),
    );
  });

  it("stays when the transform panel moves or turns the mesh", () => {
    const values = placementToValues(SCALED);
    const next = valuesToPlacement({ ...values, position: [7, 8, 9] }, SCALED);
    expect(next.translation).toEqual([7, 8, 9]);
    expect(next.scale).toEqual([2, 1.5, 0.5]);
    expect(off(next.rotation as number[], TURNED)).toBeLessThan(1e-12);
    expect(valuesToPlacement(values)).not.toHaveProperty("scale");
  });

  it("is not part of what the scale gizmo does", () => {
    expect(scaleFactor({ x: 1, y: 1, z: 1 }, { x: 2, y: 1, z: 0.5 })).toEqual([2, 1, 0.5]);
    expect(scaleFactor({ x: 2, y: 1.5, z: 0.5 }, { x: 4, y: 1.5, z: 0.25 })).toEqual([2, 1, 0.5]);
  });

  it("stays through a registration", async () => {
    const h = open();
    const put: Put = { scale: [2, 1.5, 0.5], turned: true, centred: true, at: FAR };
    const moving = putLump(h, put, [0.04, -0.03, 0.05]);
    const fixed = putLump(h, { ...put, scale: [1, 1, 1], centred: false });
    const r = await runIcpAlign({ store: h.store, moving, fixed, maxIterations: 50, overlap: 1 });
    h.store.commitFrame();
    expect(r.rmse).toBeLessThan(1e-4);
    expect(h.scene.placement(moving).scale).toEqual([2, 1.5, 0.5]);
    expect(off(model(h.scene, moving, FAR), model(h.scene, fixed, FAR))).toBeLessThan(1e-3);
  });
});

describe("apply transform far from a mesh's origin (#289)", () => {
  it("keeps the shape, and a translation when it needs one", async () => {
    for (const put of PUTS) {
      const h = open();
      const mesh = putLump(h, put);
      h.scene.clearHistory();
      const before = h.scene.placement(mesh);
      const world = model(h.scene, mesh, put.at);
      expect(await applyTransform(h.store, mesh)).toBe(true);
      h.store.commitFrame();
      const after = h.scene.placement(mesh);
      expect(after.rotation).toEqual(IDENTITY);
      expect(after.scale).toEqual([1, 1, 1]);
      // Only the placement had it far away.
      const kept = put.at === FAR && (!put.centred || put.turned || put.scale[0] !== 1);
      expect(after.translation.some((c) => c !== 0), JSON.stringify(put)).toBe(kept);
      expect(off(model(h.scene, mesh, put.at), world)).toBeLessThan(2e-6);
      expect(await applyTransform(h.store, mesh)).toBe(false);
      h.store.undo();
      h.store.commitFrame();
      expect(h.scene.placement(mesh)).toEqual(before);
      expect(off(model(h.scene, mesh, put.at), world)).toBe(0);
      expect(h.scene.canUndo()).toBe(false);
    }
  });

  it("keeps none within 1024 times the size of the origin", () => {
    const box = (x: number) => [x, 0, 0, x + 1, 1, 1];
    expect(keptTranslation(box(1000), [0, 0, 0])).toEqual([0, 0, 0]);
    expect(keptTranslation(box(1100), [0, 0, 0])).toEqual([1100.5, 0.5, 0.5]);
    expect(keptTranslation(box(1100), [1000, 0, 0])).toEqual([0, 0, 0]);
    expect(keptTranslation([], [0, 0, 0])).toEqual([0, 0, 0]);
  });
});
