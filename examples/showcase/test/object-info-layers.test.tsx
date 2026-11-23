// @vitest-environment jsdom
// The object overlays in a headless viewport (#257): the real layers in an
// R3F root, on the real engine and a real mesh binding, moved through the
// gizmo's own `attachTransformControls`. The bounding box (world and local)
// and the normals follow a drag while it is held, its release and its undo
// without being rebuilt; edits and their undo rebuild them once; they never
// catch a pointer ray; labels follow the unit; a cleanup followed by the
// effects again (a tree hidden and revealed) leaves them whole; and the angle
// stays where it was picked.
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("fast-mesh-wasm", async () => {
  return await import("../../../packages/fast-mesh-wasm/dist/web.js");
});

import { act, advance, createRoot } from "@react-three/fiber";
import { init, Scene } from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore } from "fast-mesh-wasm/store";
import { attachTransformControls, bindMesh } from "fast-mesh-wasm/three";
import { Activity } from "react";
import * as THREE from "three";
import { AppState } from "../src/app/state/app-state";
import { ObjectInfoLayers } from "../src/features/inspect/ui/ObjectInfoLayers";
import { MAX_NORMAL_LINES } from "../src/features/inspect/ux/overlay-geometry";
import { OBJECT_INFO_REFRESH_MS, objectInfoProbe } from "../src/features/inspect/ux/use-object-info";
import { formatLength } from "../src/shared/ux/units";
import { setDragging, view } from "../src/viewport/ux/view";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const cleanups: (() => void | Promise<void>)[] = [];
/** Every text drawn on a label, in order. */
let drawn: string[] = [];

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

beforeEach(() => {
  drawn = [];
  // jsdom has no 2D canvas: the labels draw on this one.
  const context = {
    measureText: (t: string) => ({ width: 20 * t.length }),
    beginPath() {},
    roundRect() {},
    fill() {},
    fillText: (t: string) => drawn.push(t),
  };
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation((() => context) as never);
});

afterEach(async () => {
  for (const fn of cleanups.splice(0).reverse()) {
    await fn();
  }
  vi.restoreAllMocks();
});

const SIZE: [number, number, number] = [4, 2, 1];

/** An axis-aligned box from the origin to `SIZE`, outward-wound. */
function boxMesh(): { positions: Float64Array; indices: Uint32Array } {
  const p: number[] = [];
  for (const z of [0, SIZE[2]]) {
    for (const y of [0, SIZE[1]]) {
      for (const x of [0, SIZE[0]]) {
        p.push(x, y, z);
      }
    }
  }
  const quads = [
    [0, 2, 3, 1],
    [4, 5, 7, 6],
    [0, 1, 5, 4],
    [2, 6, 7, 3],
    [0, 4, 6, 2],
    [1, 3, 7, 5],
  ];
  const t: number[] = [];
  for (const [a, b, c, d] of quads) {
    t.push(a, b, c, a, c, d);
  }
  return { positions: Float64Array.from(p), indices: Uint32Array.from(t) };
}

/** The app's overlays in a root whose renderer only updates the matrices, as a frame does. */
async function setup(patch: Parameters<AppState["setObjectInfo"]>[0]) {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  const data = boxMesh();
  const item = app.addMesh(store, {
    name: "part",
    positions: data.positions,
    indices: data.indices,
    center: [2, 1, 0.5],
    radius: Math.hypot(2, 1, 0.5),
    volumeId: null,
    preset: null,
  });
  store.select(item.mesh);
  store.commitFrame();
  const bound = bindMesh(scene, item.mesh);
  view.bindings = new Map([[item.mesh, bound]]);
  app.setObjectInfo(patch);

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
  let world: THREE.Scene | null = null;
  /** The viewport, or (`hidden`) its overlays with every effect cleaned up but their objects kept. */
  const render = async (mode: "visible" | "hidden") => {
    await act(async () => {
      const three = root.render(
        <SceneProvider store={store}>
          <primitive object={bound.object} />
          <Activity mode={mode}>
            <ObjectInfoLayers app={app} scene={scene} items={app.getSnapshot().items} />
          </Activity>
        </SceneProvider>,
      );
      world = three.getState().scene;
    });
  };
  await render("visible");
  let time = 0;
  const frame = async () => {
    await act(async () => {
      time += 16;
      advance(time);
    });
  };
  await frame();
  await frame();
  cleanups.push(async () => {
    await act(async () => root.unmount());
    view.bindings = new Map();
    setDragging(false);
    bound.dispose();
    store.dispose();
  });
  const find = (name: string): THREE.Object3D | null => world!.getObjectByName(name) ?? null;
  return { scene, store, app, item, bound, frame, find, render, world: world! as THREE.Scene };
}

const settle = async (ms = OBJECT_INFO_REFRESH_MS + 100) => {
  await act(async () => {
    await new Promise((r) => setTimeout(r, ms));
  });
};

/** The world box of an overlay's line segments, as they are drawn. */
function drawnBox(overlay: THREE.Object3D | null): { min: number[]; max: number[] } {
  let lines: THREE.LineSegments | null = null;
  overlay!.traverse((o) => {
    if (o instanceof THREE.LineSegments) {
      lines = o;
    }
  });
  const box = new THREE.Box3();
  const attr = lines!.geometry.getAttribute("position") as THREE.BufferAttribute;
  const p = new THREE.Vector3();
  for (let i = 0; i < attr.count; i++) {
    box.expandByPoint(p.fromBufferAttribute(attr, i).applyMatrix4(lines!.matrixWorld));
  }
  return { min: box.min.toArray(), max: box.max.toArray() };
}

/** The mesh's eight corners through the engine's placement: what a box around it must hold. */
function placedCorners(scene: Scene, mesh: Parameters<Scene["placement"]>[0]): number[][] {
  const { rotation: r, translation: t } = scene.placement(mesh);
  return [0, 1, 2, 3, 4, 5, 6, 7].map((i) => {
    const p = [i & 1 ? SIZE[0] : 0, i & 2 ? SIZE[1] : 0, i & 4 ? SIZE[2] : 0];
    return [0, 1, 2].map((k) => r[3 * k] * p[0] + r[3 * k + 1] * p[1] + r[3 * k + 2] * p[2] + t[k]);
  });
}

function expectAround(box: { min: number[]; max: number[] }, corners: number[][]): void {
  for (const k of [0, 1, 2]) {
    expect(box.min[k]).toBeCloseTo(Math.min(...corners.map((c) => c[k])), 4);
    expect(box.max[k]).toBeCloseTo(Math.max(...corners.map((c) => c[k])), 4);
  }
}

/** A gizmo on the mesh, through the same attachment as the app's: press, move, release. */
function gizmo(scene: Scene, bound: ReturnType<typeof bindMesh>) {
  const controls = Object.assign(new THREE.EventDispatcher<never>() as THREE.EventDispatcher<any>, {
    object: bound.object as THREE.Object3D,
    dragging: false,
  });
  // Mid-drag the object leads and the scene lags by up to a frame: sent at once, so both can be compared.
  const detach = attachTransformControls(controls, scene, bound, { schedule: (send) => send() });
  cleanups.push(detach);
  const setHeld = (held: boolean) => {
    controls.dragging = held;
    controls.dispatchEvent({ type: "dragging-changed", value: held });
    setDragging(held);
  };
  return {
    press: () => setHeld(true),
    move: (position: [number, number, number], zTurn = 0) => {
      bound.object.position.add(new THREE.Vector3(...position));
      bound.object.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), zTurn));
      controls.dispatchEvent({ type: "objectChange" });
    },
    release: () => setHeld(false),
  };
}

describe("object overlays in the viewport (#257)", () => {
  for (const space of ["local", "world"] as const) {
    it(`the ${space} bounding box follows a held gizmo drag, its release and its undo`, async () => {
      const { scene, store, item, bound, frame, find } = await setup({ showBox: true, space });
      const group = () => find("object-info-box");
      expect(group()).not.toBeNull();
      // The local box hangs off the mesh, the world box off the scene.
      expect(group()!.parent === bound.object).toBe(space === "local");
      expect(group()!.visible).toBe(true);
      expectAround(drawnBox(group()), placedCorners(scene, item.mesh));
      expect(drawnBox(group())).toEqual({ min: [0, 0, 0], max: SIZE });

      const g = gizmo(scene, bound);
      g.press();
      g.move([3, -1, 2]);
      store.commitFrame();
      await frame();
      // Held: the box is already there.
      expectAround(drawnBox(group()), placedCorners(scene, item.mesh));
      expect(drawnBox(group()).min).toEqual([3, -1, 2].map((x) => expect.closeTo(x, 4)));
      g.move([1, 0, 0], Math.PI / 6);
      store.commitFrame();
      await frame();
      const corners = placedCorners(scene, item.mesh);
      expectAround(drawnBox(group()), corners);
      // A turned mesh: the world box is wider than the mesh's own sides, and the local box's outline turned with it.
      expect(drawnBox(group()).max[1] - drawnBox(group()).min[1]).toBeGreaterThan(SIZE[1]);
      g.release();
      store.commitFrame();
      await frame();
      expectAround(drawnBox(group()), corners);

      store.undo();
      store.commitFrame();
      await frame();
      expect(scene.placement(item.mesh).translation).toEqual([0, 0, 0]);
      const undone = drawnBox(group());
      expectAround(undone, placedCorners(scene, item.mesh));
      expect(undone.min).toEqual([0, 0, 0].map((x) => expect.closeTo(x, 4)));
      expect(undone.max).toEqual(SIZE.map((x) => expect.closeTo(x, 4)));
    });
  }

  it("the normals follow a drag and its undo without a rebuild, and an edit and its undo rebuild them once each", async () => {
    const { scene, store, app, item, bound, frame, find } = await setup({ normals: "face", normalsScale: 0.1 });
    const lines = () => find("object-info-normals") as THREE.LineSegments;
    expect(lines().parent).toBe(bound.object);
    expect(objectInfoProbe.normalLines).toBe(12);
    const length = 0.1 * item.radius;
    const before = drawnBox(lines());
    expect(before.min).toEqual([-length, -length, -length].map((x) => expect.closeTo(x, 4)));
    expect(before.max).toEqual(SIZE.map((x) => expect.closeTo(x + length, 4)));

    const builds = objectInfoProbe.normalBuilds;
    const g = gizmo(scene, bound);
    g.press();
    g.move([5, 0, 0]);
    store.commitFrame();
    await frame();
    expect(drawnBox(lines()).min[0]).toBeCloseTo(5 - length, 4);
    g.release();
    store.undo();
    store.commitFrame();
    await frame();
    expect(drawnBox(lines()).min[0]).toBeCloseTo(-length, 4);
    await settle();
    expect(objectInfoProbe.normalBuilds).toBe(builds);

    await act(async () => {
      scene.edit(item.mesh, [{ op: "setPositions", vertices: [7], positions: Float64Array.from([8, 2, 1]) }]);
      store.commitFrame();
    });
    await settle();
    await frame();
    expect(objectInfoProbe.normalBuilds).toBe(builds + 1);
    expect(drawnBox(lines()).max[0]).toBeGreaterThan(SIZE[0] + length);
    await act(async () => {
      scene.undo();
      store.commitFrame();
    });
    await settle();
    await frame();
    expect(objectInfoProbe.normalBuilds).toBe(builds + 2);
    expect(drawnBox(lines()).max[0]).toBeCloseTo(SIZE[0] + length, 4);

    // Vertex mode draws one per vertex; off removes them.
    await act(async () => app.setObjectInfo({ normals: "vertex" }));
    await frame();
    expect(objectInfoProbe.normalLines).toBe(8);
    await act(async () => app.setObjectInfo({ normals: "off" }));
    await frame();
    expect(find("object-info-normals")).toBeNull();
    expect(objectInfoProbe.normalLines).toBe(0);
    expect(MAX_NORMAL_LINES).toBeLessThanOrEqual(20_000);
  });

  it("the box rebuilds after an edit and its undo, in both frames, and the probe says which is drawn", async () => {
    const { scene, store, app, item, frame, find } = await setup({ showBox: true, space: "local" });
    expect(objectInfoProbe.box).toEqual({ space: "local", box: { min: [0, 0, 0], max: SIZE } });
    const edit = async (x: number) => {
      await act(async () => {
        scene.edit(item.mesh, [{ op: "setPositions", vertices: [7], positions: Float64Array.from([x, 2, 1]) }]);
        store.commitFrame();
      });
      await settle();
      await frame();
    };
    await edit(8);
    expect(drawnBox(find("object-info-box")).max[0]).toBeCloseTo(8, 4);

    await act(async () => app.setObjectInfo({ space: "world" }));
    await frame();
    expect(objectInfoProbe.box?.space).toBe("world");
    expect(drawnBox(find("object-info-box")).max[0]).toBeCloseTo(8, 4);
    await edit(6);
    expect(drawnBox(find("object-info-box")).max[0]).toBeCloseTo(6, 4);

    await act(async () => {
      scene.undo();
      store.commitFrame();
    });
    await settle();
    await frame();
    expect(drawnBox(find("object-info-box")).max[0]).toBeCloseTo(8, 4);

    // Back to the local frame: its probe stands, the world box's reset does not wipe it.
    await act(async () => app.setObjectInfo({ space: "local" }));
    await frame();
    expect(objectInfoProbe.box?.space).toBe("local");
    await act(async () => app.setObjectInfo({ showBox: false }));
    await frame();
    expect(find("object-info-box")).toBeNull();
    expect(objectInfoProbe.box).toBeNull();
  });

  it("labels the box's dimensions in the unit setting", async () => {
    const { app, frame, find } = await setup({ showBox: true, space: "world" });
    expect(drawn).toEqual(expect.arrayContaining([`X ${formatLength(4, "mm")}`, `Y ${formatLength(2, "mm")}`, `Z ${formatLength(1, "mm")}`]));
    const count = drawn.length;
    // Frames without a change draw no text again.
    await frame();
    await frame();
    expect(drawn).toHaveLength(count);
    await act(async () => app.setUnits("in"));
    await frame();
    expect(drawn.slice(count)).toEqual([4, 2, 1].map((x, i) => `${"XYZ"[i]} ${formatLength(x, "in")}`));
  });

  it("never catches a pointer ray: the events raycast the mesh's children too", async () => {
    const { bound, world, frame } = await setup({ showBox: true, space: "local", normals: "face", normalsScale: 0.3 });
    await frame();
    const raycaster = new THREE.Raycaster();
    raycaster.camera = new THREE.PerspectiveCamera();
    // Along +x just above the top face: off the mesh, through its upward normals' reach and the labels.
    raycaster.set(new THREE.Vector3(-10, 1, SIZE[2] + 0.2), new THREE.Vector3(1, 0, 0));
    expect(raycaster.intersectObject(bound.object, true)).toEqual([]);
    expect(raycaster.intersectObject(world, true)).toEqual([]);
    // The mesh itself still does.
    raycaster.set(new THREE.Vector3(-10, 1, 0.5), new THREE.Vector3(1, 0, 0));
    const hits = raycaster.intersectObject(bound.object, true);
    expect(hits.length).toBeGreaterThan(0);
    expect(hits.every((h) => h.object === bound.object)).toBe(true);
  });

  for (const space of ["local", "world"] as const) {
    it(`comes back whole when its effects run again on the same objects (${space} box, normals, angle)`, async () => {
      const { scene, store, app, item, bound, frame, find, render } = await setup({
        showBox: true,
        space,
        normals: "face",
        angle: [[4, 0, 0], [0, 0, 0], [0, 2, 0]],
      });
      const whole = () => {
        // The lines and the three labels.
        expect(find("object-info-box")!.children).toHaveLength(4);
        expect(find("object-info-box")!.visible).toBe(true);
        expectAround(drawnBox(find("object-info-box")), placedCorners(scene, item.mesh));
        expect(objectInfoProbe.box?.space).toBe(space);
        expect(find("object-info-normals")!.parent).toBe(bound.object);
        expect(objectInfoProbe.normalLines).toBe(12);
        // The arms, the arc, three balls and the label.
        expect(find("object-info-angle")!.children).toHaveLength(6);
        expect(find("object-info-angle")!.visible).toBe(true);
      };
      whole();
      await render("hidden");
      await frame();
      expect(find("object-info-normals")).toBeNull();
      expect(objectInfoProbe.box).toBeNull();
      // Moved while hidden: revealed where it is now.
      store.setPlacement(item.mesh, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [2, 3, 4] });
      store.commitFrame();
      await render("visible");
      await frame();
      await frame();
      whole();
      // And where it was, when nothing moved meanwhile.
      await render("hidden");
      await frame();
      await render("visible");
      await frame();
      whole();
      expect(app.getSnapshot().objectInfo.showBox).toBe(true);
    });
  }

  it("hides the box with its mesh, and drops the overlays with the selection", async () => {
    const { app, store, bound, frame, find } = await setup({ showBox: true, space: "world", normals: "vertex" });
    bound.object.visible = false;
    await frame();
    expect(find("object-info-box")!.visible).toBe(false);
    bound.object.visible = true;
    await frame();
    expect(find("object-info-box")!.visible).toBe(true);
    await act(async () => {
      store.select(null);
      store.commitFrame();
    });
    await frame();
    expect(find("object-info-box")).toBeNull();
    expect(find("object-info-normals")).toBeNull();
    expect(app.getSnapshot().objectInfo.showBox).toBe(true);
  });

  it("draws the angle in the world, where it was picked, through moves and undo", async () => {
    const { scene, store, app, bound, frame, find } = await setup({});
    const group = () => find("object-info-angle")!;
    expect(group().visible).toBe(false);
    await act(async () => app.setObjectInfo({ angle: [[4, 0, 0], [0, 0, 0]] }));
    await frame();
    expect(group().visible).toBe(true);
    expect(drawn).toEqual([]);
    await act(async () => app.setObjectInfo({ angle: [[4, 0, 0], [0, 0, 0], [0, 2, 0]] }));
    await frame();
    expect(drawn).toEqual(["90.0°"]);
    expect(objectInfoProbe.angle).toEqual({ points: [[4, 0, 0], [0, 0, 0], [0, 2, 0]], degrees: 90 });
    const label = group().children.find((c) => c instanceof THREE.Sprite)!;
    expect(label.visible).toBe(true);
    const at = label.getWorldPosition(new THREE.Vector3()).toArray();
    const g = gizmo(scene, bound);
    g.press();
    g.move([9, 9, 9]);
    g.release();
    store.commitFrame();
    await frame();
    expect(label.getWorldPosition(new THREE.Vector3()).toArray()).toEqual(at);
    // A degenerate arm has no angle, and no label.
    await act(async () => app.setObjectInfo({ angle: [[1, 1, 1], [1, 1, 1], [0, 2, 0]] }));
    await frame();
    expect(label.visible).toBe(false);
    expect(objectInfoProbe.angle.degrees).toBeNull();
    await act(async () => app.setObjectInfo({ angle: [] }));
    await frame();
    expect(group().visible).toBe(false);
  });
});
