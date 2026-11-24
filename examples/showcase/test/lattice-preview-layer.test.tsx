// @vitest-environment jsdom
// The lattice preview in a headless viewport (#265): the real layer in an R3F
// root, on the real engine and a real mesh binding. The walls stand where
// their mesh does and follow it, the mesh turns see-through while they are up
// and back to its own opacity after, they never catch a pointer ray, and a
// mesh that is hidden or gone takes them along.
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeAll, expect, it, vi } from "vitest";

vi.mock("fast-mesh-wasm", async () => {
  return await import("../../../packages/fast-mesh-wasm/dist/web.js");
});

import { act, advance, createRoot } from "@react-three/fiber";
import { init, Scene } from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore } from "fast-mesh-wasm/store";
import { bindMesh } from "fast-mesh-wasm/three";
import * as THREE from "three";
import { AppState } from "../src/app/state/app-state";
import {
  LatticePreviewLayer,
  LatticeSeeThrough,
  SEE_THROUGH_OPACITY,
} from "../src/features/mesh-edit/ui/LatticePreviewLayer";
import { latticePreviewOf } from "../src/features/mesh-edit/ux/lattice-preview-state";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const cleanups: (() => void | Promise<void>)[] = [];

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

afterEach(async () => {
  for (const fn of cleanups.splice(0).reverse()) {
    await fn();
  }
});

/** A box two units across, stored around the origin. */
function box(): { positions: Float64Array; indices: Uint32Array } {
  const p: number[] = [];
  for (let i = 0; i < 8; i++) {
    p.push(i & 1 ? 1 : -1, i & 2 ? 1 : -1, i & 4 ? 1 : -1);
  }
  const t = [0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5];
  return { positions: Float64Array.from(p), indices: Uint32Array.from(t) };
}

async function setup() {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  const item = app.addMesh(store, { name: "part", ...box(), center: [0, 0, 0], radius: 2, volumeId: null, preset: null });
  store.commitFrame();
  const material = new THREE.MeshStandardMaterial();
  const bound = bindMesh(scene, item.mesh, { material });
  const canvas = document.createElement("canvas");
  const gl = {
    render: (s: THREE.Scene) => s.updateMatrixWorld(),
    setPixelRatio() {},
    setSize() {},
    setAnimationLoop() {},
    dispose() {},
    forceContextLoss() {},
    domElement: canvas,
    shadowMap: {},
    renderLists: { dispose() {} },
    xr: { addEventListener() {}, removeEventListener() {}, setAnimationLoop() {} },
  };
  const root = createRoot(canvas);
  await root.configure({
    gl: gl as never,
    frameloop: "never",
    dpr: 1,
    size: { width: 100, height: 100, top: 0, left: 0 },
  });
  let world!: THREE.Scene;
  await act(async () => {
    const three = root.render(
      <SceneProvider store={store}>
        <primitive object={bound.object} />
        <LatticePreviewLayer app={app} store={store} />
        <LatticeSeeThrough app={app} mesh={item.mesh} bound={bound} />
      </SceneProvider>,
    );
    world = three.getState().scene;
  });
  let time = 0;
  const frame = async () => {
    await act(async () => {
      time += 16;
      advance(time);
    });
  };
  cleanups.push(async () => {
    await act(async () => root.unmount());
    bound.dispose();
    store.dispose();
  });
  const walls = () => (world.getObjectByName("lattice-preview") ?? null) as THREE.Mesh | null;
  return { scene, store, app, item, material, frame, walls, world };
}

it("draws the walls in a see-through mesh, follows it, and puts it back", async () => {
  const { scene, store, app, item, material, frame, walls } = await setup();
  const preview = latticePreviewOf(app);
  expect(walls()).toBeNull();
  const built = scene.latticePreview(item.mesh, { cellSize: 1, wallThickness: 0.25, voxelSize: 0.125 });
  expect(built.indices.length).toBeGreaterThan(1000);
  await act(async () => preview.show({ mesh: item.mesh, walls: built }));
  await frame();
  expect(walls()?.visible).toBe(true);
  expect(material.opacity).toBe(SEE_THROUGH_OPACITY);
  expect(material.transparent).toBe(true);

  // Where the mesh is, through a move and a scale.
  scene.setPlacement(item.mesh, {
    rotation: [0, -1, 0, 1, 0, 0, 0, 0, 1],
    translation: [5, 0, 0],
    scale: [2, 2, 2],
  });
  store.commitFrame();
  await frame();
  const at = new THREE.Box3().setFromObject(walls()!);
  for (const [lo, hi, centre] of [
    [at.min.x, at.max.x, 5],
    [at.min.y, at.max.y, 0],
    [at.min.z, at.max.z, 0],
  ]) {
    expect(lo).toBeCloseTo(centre - 2, 1);
    expect(hi).toBeCloseTo(centre + 2, 1);
  }
  // Never under the pointer.
  const ray = new THREE.Raycaster(new THREE.Vector3(5, 0, 20), new THREE.Vector3(0, 0, -1));
  expect(ray.intersectObject(walls()!)).toHaveLength(0);

  // Hidden with its mesh.
  store.setDisplay(item.mesh, { visible: false });
  store.commitFrame();
  await frame();
  expect(walls()?.visible).toBe(false);
  store.setDisplay(item.mesh, { visible: true, opacity: 0.6 });
  store.commitFrame();
  await frame();
  expect(walls()?.visible).toBe(true);

  // A new preview replaces it; none puts the mesh back at its own opacity.
  const old = walls();
  const other = scene.latticePreview(item.mesh, { kind: "diamond", cellSize: 1, wallThickness: 0.25, voxelSize: 0.125 });
  await act(async () => preview.show({ mesh: item.mesh, walls: other }));
  await frame();
  expect(walls()).not.toBe(old);
  await act(async () => preview.show(null));
  await frame();
  expect(walls()).toBeNull();
  expect(material.opacity).toBe(0.6);
});

it("shows nothing for a lattice without walls, or a mesh that is gone", async () => {
  const { scene, store, app, item, material, frame, walls } = await setup();
  const preview = latticePreviewOf(app);
  const empty = scene.latticePreview(item.mesh, { cellSize: 1, wallThickness: 0.25, shellThickness: 1.5 });
  expect(empty.indices).toHaveLength(0);
  await act(async () => preview.show({ mesh: item.mesh, walls: empty }));
  await frame();
  expect(walls()).toBeNull();
  expect(material.opacity).toBe(1);

  const built = scene.latticePreview(item.mesh, { cellSize: 1, wallThickness: 0.25, voxelSize: 0.125 });
  await act(async () => preview.show({ mesh: item.mesh, walls: built }));
  await frame();
  expect(walls()?.visible).toBe(true);
  store.removeMesh(item.mesh);
  store.commitFrame();
  await frame();
  expect(walls()?.visible).toBe(false);
});
