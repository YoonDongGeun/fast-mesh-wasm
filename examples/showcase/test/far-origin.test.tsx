// @vitest-environment jsdom
// Contour slices, fitted primitives, surface paths, point clouds and scene
// stamps far from the world origin (#280), on the real engine: what the tools
// keep and what the overlays draw is as fine there as near it (`f32` world
// positions are off by up to 0.016 at FAR).
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

vi.mock("fast-mesh-wasm", async () => {
  return await import("../../../packages/fast-mesh-wasm/dist/web.js");
});

const three = vi.hoisted(() => ({ current: null as null | { camera: unknown; gl: { domElement: HTMLElement }; raycaster: unknown } }));
vi.mock("@react-three/fiber", async (original) => ({
  ...(await original<object>()),
  useThree: () => three.current,
}));

import { type FittedPrimitiveReport, init, type MeshHandle, pointCloudEstimateNormals, Scene, type Vec3 } from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act } from "react";
import { createRoot } from "react-dom/client";
import * as THREE from "three";
import { AppState, type Item } from "../src/app/state/app-state";
import { SurfacePathLayer } from "../src/features/inspect/ui/SurfacePathLayer";
import { pathSegments, pickFromHit, startPath, traceLeg, extendPath } from "../src/features/inspect/ux/surface-paths";
import { exportContourSlicesDxf, sliceCurves } from "../src/features/inspect/ux/use-contour-slices";
import { executePlaceStampAtHit } from "../src/features/mesh-edit/ux/print-prep-ops";
import {
  buildContourSegmentsGeometry,
  buildPointCloudGeometry,
  buildPrimitiveGuideSegments,
  contourDrawOrigin,
  MetrologyOverlaysLayer,
  primitiveDrawOrigin,
} from "../src/features/points/ui/MetrologyOverlaysLayer";
import { computePointCloudBounds, parsePointCloudFileContent, sampleMeshToPointCloud } from "../src/features/points/ux/point-cloud";
import { runConvexHull } from "../src/engine/workbench-ops";
import { unpackPointCloudLocal } from "../src/engine/wave6-wasm-lazy";
import { blobbySphere, roundedBox } from "../src/io/demo";
import { bounds } from "../src/io/mesh-data";
import { drawOrigin } from "../src/shared/ux/draw-origin";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const cleanups: (() => void)[] = [];

beforeAll(async () => {
  const candidates = [
    resolve(process.cwd(), "../../packages/fast-mesh-wasm/dist/wasm/fast_mesh_wasm_bg.wasm"),
    resolve(process.cwd(), "packages/fast-mesh-wasm/dist/wasm/fast_mesh_wasm_bg.wasm"),
  ];
  const wasmPath = candidates.find((p) => existsSync(p));
  if (!wasmPath) {
    throw new Error(`Missing prebuilt Wasm binary in: ${candidates.join(", ")}`);
  }
  await init(readFileSync(wasmPath));
});

afterEach(() => {
  for (const fn of cleanups.splice(0).reverse()) {
    fn();
  }
});

const FAR: Vec3 = [148_000.3, -277_000.7, 91_234.1];
const ZERO: Vec3 = [0, 0, 0];
/** `f32` steps at the sizes here, and where an overlay draws (within 512 of its point). */
const TOL = 1e-6;
const DRAWN = 5e-5;

function setup() {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  cleanups.push(() => {
    store.dispose();
    scene.free();
  });
  return { scene, store, app };
}

function add(app: AppState, store: SceneStore, name: string, d: { positions: Float64Array; indices: Uint32Array }): Item {
  const item = app.addMesh(store, { name, ...d, ...bounds(d.positions), volumeId: null, preset: null });
  store.commitFrame();
  return item;
}

const ball = (at: Readonly<Vec3>) => blobbySphere([at[0], at[1], at[2]], 1, 32, 16, 0);

/** The largest distance along an axis of `got` from `want` moved by `shift`. */
function off(got: ArrayLike<number>, want: ArrayLike<number>, shift: Readonly<Vec3>): number {
  expect(got.length).toBe(want.length);
  let m = 0;
  for (let i = 0; i < got.length; i++) m = Math.max(m, Math.abs(got[i] - (want[i] + shift[i % 3])));
  return m;
}

function worldPositions(scene: Scene, mesh: MeshHandle): number[] {
  const p = scene.views(mesh).positions();
  const m = scene.matrix4(mesh);
  const out: number[] = [];
  for (let i = 0; i + 2 < p.length; i += 3) {
    for (let r = 0; r < 3; r++) out.push(m[r] * p[i] + m[4 + r] * p[i + 1] + m[8 + r] * p[i + 2] + m[12 + r]);
  }
  return out;
}

const drawnAt = (geo: THREE.BufferGeometry, around: Readonly<Vec3>) =>
  Float64Array.from(geo.getAttribute("position").array, (c, k) => c + around[k % 3]);

describe("what is drawn far away", () => {
  it("is drawn around a point near it, and near the world origin at its coordinates", () => {
    for (const p of [ZERO, [-500, 511, -0.1]] as Vec3[]) {
      expect(drawOrigin(p)).toEqual(ZERO);
      expect(drawOrigin(p).some((c) => Object.is(c, -0))).toBe(false);
    }
    const o = drawOrigin(FAR);
    for (let k = 0; k < 3; k++) {
      expect(Math.abs(o[k] - FAR[k])).toBeLessThanOrEqual(512);
      expect(Math.fround(o[k])).toBe(o[k]);
    }
  });

  it("contours, the curves kept of them and their export", () => {
    const { scene, store, app } = setup();
    const [near, far] = [add(app, store, "Near", ball(ZERO)), add(app, store, "Far", ball(FAR))];
    const slice = (item: Item, at: Readonly<Vec3>) => [
      scene.sliceContours(item.mesh, [at[0], at[1], at[2] + 0.1], [0, 0, 1]),
      scene.sliceContours(item.mesh, [at[0], at[1], at[2] + 0.1], [0.6, 0, 0.8]),
    ];
    const [a, b] = [slice(near, ZERO), slice(far, FAR)];
    const points = (slices: typeof a) => sliceCurves(slices).flatMap((c) => Array.from(c.points));
    expect(off(points(b), points(a), FAR)).toBeLessThan(TOL);
    // A slice without the f64 points (from an older save, say) still has curves.
    const plain = b.map((s) => ({ ...s, loops: s.loops.map(({ points3d64: _, ...l }) => l) }));
    expect(off(points(plain), points(a), FAR)).toBeGreaterThan(1e-3);
    // The DXF is in the plane: the same as near.
    expect(exportContourSlicesDxf(b.slice(0, 1))).toContain("LWPOLYLINE");
    const numbers = (dxf: string) => dxf.split("\n").map(Number).filter(Number.isFinite);
    const [na, nb] = [a, b].map((s) => numbers(exportContourSlicesDxf(s.slice(0, 1))));
    expect(off(nb, na, ZERO)).toBeLessThan(1e-5);

    const around = contourDrawOrigin(b);
    expect(off(drawnAt(buildContourSegmentsGeometry(b, around), around), drawnAt(buildContourSegmentsGeometry(a), ZERO), FAR)).toBeLessThan(DRAWN);
    expect(contourDrawOrigin(a)).toEqual(ZERO);
    expect(contourDrawOrigin(null)).toEqual(ZERO);
  });

  it("the guides of fitted primitives", () => {
    const report = (primitive: FittedPrimitiveReport["primitive"]): FittedPrimitiveReport => ({
      primitive,
      rmsError: 0,
      maxError: 0,
      inlierCount: 1,
      inlierRatio: 1,
      residuals: new Float32Array(0),
    });
    const all = (p: Vec3): FittedPrimitiveReport["primitive"][] => [
      { kind: "plane", origin: p, normal: [0, 0.6, 0.8] },
      { kind: "sphere", center: p, radius: 2 },
      { kind: "cylinder", axisOrigin: p, axisDir: [0, 1, 0], radius: 0.5, height: 2 },
      { kind: "cone", apex: p, axisDir: [0, 0, 1], halfAngleRad: 0.4, height: 2 },
    ];
    const near = all(ZERO);
    for (const [i, primitive] of all(FAR).entries()) {
      const far = report(primitive);
      const around = primitiveDrawOrigin(far);
      expect(around).toEqual(drawOrigin(FAR));
      const want = drawnAt(buildPrimitiveGuideSegments(report(near[i])), ZERO);
      expect(off(drawnAt(buildPrimitiveGuideSegments(far, around), around), want, FAR)).toBeLessThan(DRAWN);
    }
    expect(primitiveDrawOrigin(null)).toEqual(ZERO);
  });

  it("puts the overlays there", async () => {
    const { scene, store, app } = setup();
    const item = add(app, store, "Far", ball(FAR));
    const cloud = sampleMeshToPointCloud(scene, item.mesh);
    const slices = [scene.sliceContours(item.mesh, FAR, [0, 0, 1])];
    const fitted = scene.fitPrimitive(item.mesh, { kind: "sphere" });
    const host = document.createElement("div");
    const root = createRoot(host);
    const moved = vi.spyOn(THREE.Vector3.prototype, "set");
    cleanups.push(() => moved.mockRestore());
    await act(async () => root.render(<MetrologyOverlaysLayer app={app} store={store} />));
    cleanups.push(() => act(() => root.unmount()));
    const at = () => moved.mock.calls.map((c) => c.join());
    moved.mockClear();
    await act(async () => app.updateWave6((p) => ({ ...p, metrology: { ...p.metrology, pointCloud: cloud } })));
    expect(at()).toContain(cloud.origin!.join());
    moved.mockClear();
    await act(async () => app.updateWave6((p) => ({ ...p, metrology: { ...p.metrology, contourSlices: slices } })));
    expect(at()).toContain(drawOrigin(FAR).join());
    moved.mockClear();
    await act(async () => app.updateWave6((p) => ({ ...p, metrology: { ...p.metrology, fittedPrimitive: fitted } })));
    expect(at()).toContain(drawOrigin(FAR).join());
  });

  it("surface paths", async () => {
    const { scene, store, app } = setup();
    const trace = async (at: Vec3) => {
      const item = add(app, store, "Ball", ball(at));
      const hit = (triangle: number) => pickFromHit(scene, { mesh: item.mesh, triangle, barycentric: [1, 0, 0] });
      const [from, to] = [hit(40), hit(700)];
      const leg = await traceLeg(scene, item.mesh, from, to, "dijkstra");
      return { item, path: extendPath(startPath(from), to, leg) };
    };
    const [near, far] = [await trace(ZERO), await trace(FAR)];
    const [a, b] = [near.path.legs[0], far.path.legs[0]];
    expect(off(b.positions, a.positions, FAR)).toBeLessThan(TOL);
    expect(b.length).toBeCloseTo(a.length, 5);
    const around = drawOrigin(FAR);
    const drawn = Float64Array.from(pathSegments(far.path, 0.002, around), (c, k) => c + around[k % 3]);
    expect(off(drawn, pathSegments(near.path, 0.002), FAR)).toBeLessThan(DRAWN);

    three.current = {
      camera: new THREE.PerspectiveCamera(),
      gl: { domElement: document.createElement("canvas") },
      raycaster: new THREE.Raycaster(),
    };
    const added = vi.spyOn(THREE.Object3D.prototype, "add");
    const root = createRoot(document.createElement("div"));
    await act(async () =>
      root.render(
        <SceneProvider store={store}>
          <SurfacePathLayer app={app} store={store} />
        </SceneProvider>,
      ),
    );
    cleanups.push(() => act(() => root.unmount()));
    const group = added.mock.contexts[added.mock.calls.findIndex((c) => c.length === 6)] as THREE.Group;
    added.mockRestore();
    const at = () => group.children.slice(1, 4).map((l) => l.position.toArray());
    await act(async () => app.setSurfacePath({ mesh: far.item.mesh, path: far.path }));
    expect(at()).toEqual([around, around, around]);
    await act(async () => app.setSurfacePath({ mesh: near.item.mesh, path: near.path }));
    expect(at()).toEqual([ZERO, ZERO, ZERO]);
  });
});

describe("point clouds far away", () => {
  const text = (at: Vec3) => {
    const { positions } = ball(at);
    const rows: string[] = [];
    for (let i = 0; i < positions.length; i += 3) {
      const n = [0, 1, 2].map((k) => positions[i + k] - at[k]);
      rows.push([positions[i], positions[i + 1], positions[i + 2], ...n].join(" "));
    }
    return rows.join("\n");
  };

  it("keep their digits from a file to the mesh and the hull made of them", async () => {
    const { scene, store, app } = setup();
    const load = (at: Vec3, name = "scan.xyz") => parsePointCloudFileContent(name, new TextEncoder().encode(text(at)));
    const [near, far] = [await load(ZERO), await load(FAR)];
    const world = (c: typeof near) => {
      const { positions, origin } = unpackPointCloudLocal(c);
      return Float64Array.from(positions, (x, k) => x + origin[k % 3]);
    };
    expect(off(world(far), ball(FAR).positions, ZERO)).toBeLessThan(TOL);
    expect(off(world(near), ball(ZERO).positions, ZERO)).toBeLessThan(TOL);
    const [bn, bf] = [computePointCloudBounds(near)!, computePointCloudBounds(far)!];
    expect(off(bf.min, bn.min, FAR)).toBeLessThan(TOL);
    expect(off(bf.max, bn.max, FAR)).toBeLessThan(TOL);
    const around = far.origin as Vec3;
    expect(off(drawnAt(buildPointCloudGeometry(far), around), world(far), ZERO)).toBeLessThan(1e-9);

    // Both have their vertices at points of the cloud.
    const expectOnCloud = (mesh: MeshHandle) => {
      const [p, points] = [worldPositions(scene, mesh), ball(FAR).positions];
      expect(p.length).toBeGreaterThan(600);
      for (let i = 0; i < p.length; i += 3) {
        let nearest = Infinity;
        for (let j = 0; j < points.length; j += 3) {
          const d = [0, 1, 2].map((k) => Math.abs(p[i + k] - points[j + k]));
          nearest = Math.min(nearest, Math.max(...d));
        }
        expect(nearest).toBeLessThan(TOL);
      }
    };
    expectOnCloud(scene.reconstructPointCloud(far, { method: "ballPivoting" }));
    expect(pointCloudEstimateNormals(far, 8).origin).toEqual(far.origin);
    const hull = await runConvexHull({ store, app, name: "Hull", points: far });
    expectOnCloud(hull.item.mesh);

    const pts = await load(FAR, "scan.pts");
    expect(off(world(pts), ball(FAR).positions, ZERO)).toBeLessThan(TOL);
  });

  it("sampled from a mesh lie on it", () => {
    const { scene, store, app } = setup();
    const d = roundedBox(FAR, [1, 1, 1], 0.08, 2);
    const item = add(app, store, "Far", d);
    store.setPlacement(item.mesh, { rotation: [0, -1, 0, 1, 0, 0, 0, 0, 1], translation: [5, 0, -7], scale: [1, 1, 2] });
    store.commitFrame();
    const cloud = sampleMeshToPointCloud(scene, item.mesh);
    const vertices = worldPositions(scene, item.mesh);
    const { positions, origin } = unpackPointCloudLocal(cloud);
    const world = Float64Array.from(positions.subarray(0, vertices.length), (x, k) => x + origin[k % 3]);
    expect(off(world, vertices, ZERO)).toBeLessThan(TOL);
  });
});

describe("scene stamps far from their own zero", () => {
  it("stand on the hit point", async () => {
    const run = async (at: Vec3) => {
      const { scene, store, app } = setup();
      const target = add(app, store, "Target", roundedBox([0, 0, 0], [4, 4, 4], 0.2, 3));
      const stamp = add(app, store, "Stamp", roundedBox([at[0], at[1], at[2] + 0.5], [0.5, 0.5, 0.5], 0.05, 2));
      const res = await executePlaceStampAtHit({
        store,
        app,
        target: target.mesh,
        hitPoint: [1, 0.5, 4],
        hitNormal: [0, 0, 1],
        placeParams: {
          ...app.getSnapshot().wave6.printPrep.placeParams,
          stampSource: "sceneMesh",
          stampMeshId: stamp.id,
          mode: "keepSeparate",
          embedDepth: 0,
          twistDeg: 0,
          uniformScale: 1,
        },
      });
      return worldPositions(scene, res.added!);
    };
    const [near, far] = [await run(ZERO), await run(FAR)];
    expect(Math.min(...near.filter((_, i) => i % 3 === 2))).toBeCloseTo(4, 6);
    expect(off(far, near, ZERO)).toBeLessThan(TOL);
  });
});
