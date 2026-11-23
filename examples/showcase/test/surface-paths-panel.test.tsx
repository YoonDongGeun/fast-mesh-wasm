// @vitest-environment jsdom
// The geodesic map and surface path wiring (#252), on the real engine: the
// inspect panel's two sections (arming, the map with its readouts, iso-lines,
// units and how it shares the overlay with the surface field and the inspect
// maps; drawing, undoing and closing a path, its length, Trim and Split and
// their undo), what clears the picks and what leaves them (camera drags,
// colours, a cut that fails), the canvas taps, what the viewport layer draws,
// and the Esc key and hint.
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { Worker as NodeThread } from "node:worker_threads";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

vi.mock("fast-mesh-wasm", async () => {
  return await import("../../../packages/fast-mesh-wasm/dist/web.js");
});

/** What the canvas hook sees of three: a camera, a canvas and a raycaster. */
const three = vi.hoisted(() => ({ current: null as null | { camera: unknown; gl: { domElement: HTMLElement }; raycaster: unknown } }));
vi.mock("@react-three/fiber", async (original) => ({
  ...(await original<object>()),
  useThree: () => three.current,
}));

import { generatePrimitive, init, type MeshHandle, type RayHit, Scene } from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "fast-mesh-wasm/worker";
import { act, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import * as THREE from "three";
import type { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { AppState, type Item } from "../src/app/state/app-state";
import { useKeyboardShortcuts } from "../src/app/ux/use-keyboard-shortcuts";
import { viewportHint } from "../src/app/ux/viewport-hint";
import { InspectPanel } from "../src/features/inspect/ui/InspectPanel";
import { SurfacePathLayer } from "../src/features/inspect/ui/SurfacePathLayer";
import { autoIsoSpacing } from "../src/features/inspect/ux/geodesic-map";
import { inspectMapProbe } from "../src/features/inspect/ux/use-inspect-maps";
import {
  NO_CUT_MESSAGE,
  queueSurfaceTap,
  surfacePathIdle,
  surfaceStats,
} from "../src/features/inspect/ux/surface-path-actions";
import { pathLength, pathSegments } from "../src/features/inspect/ux/surface-paths";
import { geodesicMapProbe } from "../src/features/inspect/ux/use-geodesic-map";
import { useSurfacePickPointer } from "../src/features/inspect/ux/use-surface-pick";
import { blobbySphere } from "../src/io/demo";
import { formatLength } from "../src/shared/ux/units";
import { view } from "../src/viewport/ux/view";

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
  view.multiTouch = false;
  view.dragging = false;
});

type V3 = [number, number, number];

/** The triangles an index buffer holds, dead slots (`0, 0, 0`) left out. */
function liveTriangleCount(indices: ArrayLike<number>): number {
  let n = 0;
  for (let k = 0; k + 2 < indices.length; k += 3) {
    if (indices[k] !== 0 || indices[k + 1] !== 0 || indices[k + 2] !== 0) {
      n++;
    }
  }
  return n;
}

function addMesh(store: SceneStore, app: AppState, name: string, positions: Float64Array, indices: Uint32Array): Item {
  return app.addMesh(store, { name, positions, indices, center: [0, 0, 0], radius: 1, volumeId: null, preset: null });
}

/** A scene with the unit sphere "Ball" selected, and a second ball "Other" beside it. */
function setup() {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  const d = blobbySphere([0, 0, 0], 1, 32, 16, 0);
  const item = addMesh(store, app, "Ball", d.positions, d.indices);
  const e = blobbySphere([4, 0, 0], 1, 32, 16, 0);
  const other = addMesh(store, app, "Other", e.positions, e.indices);
  store.select(item.mesh);
  store.commitFrame();
  scene.clearHistory();
  cleanups.push(() => {
    store.dispose();
    scene.free();
  });
  return { scene, store, app, item, other };
}

function mount(app: AppState, store: SceneStore, scene: Scene): HTMLElement {
  const host = document.createElement("div");
  document.body.appendChild(host);
  const root: Root = createRoot(host);
  act(() => {
    root.render(
      <SceneProvider store={store}>
        <InspectPanel app={app} scene={scene} />
      </SceneProvider>,
    );
  });
  cleanups.push(() => {
    act(() => root.unmount());
    host.remove();
  });
  return host;
}

const q = <T extends Element = HTMLElement>(host: HTMLElement, selector: string): T => {
  const el = host.querySelector<T>(selector);
  if (el === null) {
    throw new Error(`missing ${selector}`);
  }
  return el;
};
const action = (host: HTMLElement, name: string) => q<HTMLButtonElement>(host, `[data-action="${name}"]`);

async function click(el: Element): Promise<void> {
  await act(async () => {
    (el as HTMLElement).click();
    await new Promise((r) => setTimeout(r, 0));
  });
}

/** Lets promises and effects run, then a frame's changes out. */
async function frame(store: SceneStore, ms = 5): Promise<void> {
  await act(async () => {
    await new Promise((r) => setTimeout(r, ms));
    store.commitFrame();
    await new Promise((r) => setTimeout(r, 0));
  });
}

async function type(input: HTMLInputElement, value: string): Promise<void> {
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
    setter?.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    await new Promise((r) => setTimeout(r, 0));
  });
}

/** The engine's hit on the ray from outside the unit sphere at `centre` toward it. */
function hitToward(scene: Scene, dir: V3, centre: V3 = [0, 0, 0]): RayHit {
  const n = Math.hypot(...dir);
  const u: V3 = [dir[0] / n, dir[1] / n, dir[2] / n];
  const hit = scene.raycast([centre[0] + 1.5 * u[0], centre[1] + 1.5 * u[1], centre[2] + 1.5 * u[2]], [-u[0], -u[1], -u[2]]);
  if (hit === null) {
    throw new Error("missed");
  }
  return hit;
}

/** Four directions around the +Y pole. */
const CAP: V3[] = [0, 1, 2, 3].map((k) => {
  const phi = (k * Math.PI) / 2 + 0.2;
  return [Math.sin(0.7) * Math.cos(phi), Math.cos(0.7), Math.sin(0.7) * Math.sin(phi)];
});

async function tap(app: AppState, scene: Scene, dirs: readonly V3[]): Promise<void> {
  await act(async () => {
    for (const d of dirs) {
      void queueSurfaceTap(app, scene, Promise.resolve(hitToward(scene, d)));
    }
    await surfacePathIdle(app);
    await new Promise((r) => setTimeout(r, 0));
  });
}

const readoutOf = (host: HTMLElement) => host.querySelector("[data-geodesic-readout]")?.textContent ?? null;
const pathOf = (host: HTMLElement) => host.querySelector("[data-path-readout]")?.textContent ?? null;
const tri = (scene: Scene, mesh: MeshHandle) => liveTriangleCount(scene.views(mesh).indices());

/** The SDK's Node worker-thread stand-in for the browser engine Worker. */
class NodeWorker implements WorkerLike {
  readonly worker = new NodeThread(
    [
      resolve(process.cwd(), "../../packages/fast-mesh-wasm/test/harness/node-worker.mjs"),
      resolve(process.cwd(), "packages/fast-mesh-wasm/test/harness/node-worker.mjs"),
    ].find((p) => existsSync(p)) ?? "",
    { workerData: { serve: true } },
  );
  readonly #message: ((e: { data: unknown }) => void)[] = [];
  readonly #error: ((e: unknown) => void)[] = [];

  constructor() {
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

  addEventListener(type: "message" | "error", cb: (e: never) => void): void {
    (type === "message" ? this.#message : this.#error).push(cb as (e: unknown) => void);
  }

  terminate(): void {
    void this.worker.terminate();
  }
}

describe("geodesic distance section (#252)", () => {
  it("arms on the selected mesh only, and disarms the other pointer tools", async () => {
    const { scene, store, app, item } = setup();
    const host = mount(app, store, scene);
    const pick = action(host, "geodesic-pick");
    expect(pick.disabled).toBe(false);
    act(() => {
      app.setInspectTool("ruler");
      app.setMaskPaint(true);
    });
    await click(pick);
    expect(app.getSnapshot().surfacePath).toMatchObject({ armed: "sources", mesh: item.mesh });
    expect(app.getSnapshot().inspectTool).toBeNull();
    expect(app.getSnapshot().maskPaint).toBe(false);
    expect(pick.getAttribute("aria-pressed")).toBe("true");
    // One pick at a time.
    await click(action(host, "path-pick"));
    expect(app.getSnapshot().surfacePath.armed).toBe("path");
    expect(pick.getAttribute("aria-pressed")).toBe("false");
    // Pressing the armed one disarms it; another tool arming stands it down and keeps the picks.
    await click(action(host, "path-pick"));
    expect(app.getSnapshot().surfacePath.armed).toBeNull();
    await click(pick);
    await tap(app, scene, [[0, 1, 0]]);
    await act(async () => app.setInspectTool("query"));
    expect(app.getSnapshot().surfacePath.armed).toBeNull();
    expect(app.getSnapshot().surfacePath.sources).toHaveLength(1);

    // Nothing selected: nothing to arm.
    await act(async () => {
      store.select(null);
      store.commitFrame();
    });
    expect(action(host, "geodesic-pick").disabled).toBe(true);
    expect(action(host, "path-pick").disabled).toBe(true);
  });

  it("maps the distance from every source, in the chosen method and unit", async () => {
    const { scene, store, app, item } = setup();
    const host = mount(app, store, scene);
    const runs = geodesicMapProbe.runs;
    await click(action(host, "geodesic-pick"));
    await tap(app, scene, [[0, 1, 0]]);
    await frame(store);
    const one = scene.inspect.geodesicDistances(item.mesh, [0]);
    let overlay = app.getSnapshot().surfaceFieldValues;
    expect(overlay?.mesh).toBe(item.mesh);
    expect(Array.from(overlay?.values ?? [])).toEqual(Array.from(one));
    const max = Math.max(...one);
    expect(overlay).toMatchObject({ map: "turbo", min: 0, max });
    expect(readoutOf(host)).toContain(`1 source · farthest ${formatLength(max, "mm")}`);
    expect(readoutOf(host)).toContain("edge graph");
    expect(host.querySelector("[data-inspect-geodesic] .legend-bar")).not.toBeNull();
    expect(geodesicMapProbe.shown).toMatchObject({ mesh: "Ball", sources: [0], method: "dijkstra" });
    expect(geodesicMapProbe.runs).toBe(runs + 1);

    // A second source brings the farthest vertex closer.
    await tap(app, scene, [[0, -1, 0]]);
    await frame(store);
    expect(readoutOf(host)).toContain("2 sources");
    const two = Math.max(...scene.inspect.geodesicDistances(item.mesh, [0, 481]));
    expect(two).toBeLessThan(max * 0.6);
    expect(readoutOf(host)).toContain(`farthest ${formatLength(two, "mm")}`);

    // The method picker runs it again; units follow the setting.
    await click(q(host, '[data-geodesic-method="heatMethod"]'));
    await frame(store);
    const heat = scene.inspect.geodesicDistances(item.mesh, [0, 481], { method: "heatMethod" });
    overlay = app.getSnapshot().surfaceFieldValues;
    expect(Array.from(overlay?.values ?? [])).toEqual(Array.from(heat));
    expect(readoutOf(host)).toContain("heat method");
    await act(async () => app.setUnits("in"));
    expect(readoutOf(host)).toContain(`farthest ${formatLength(Math.max(...heat), "in")}`);

    // Undo source, then Clear.
    await click(action(host, "geodesic-undo"));
    await frame(store);
    expect(readoutOf(host)).toContain("1 source");
    await click(action(host, "geodesic-clear"));
    await frame(store);
    expect(app.getSnapshot().surfaceFieldValues).toBeNull();
    expect(readoutOf(host)).toBeNull();
    expect(geodesicMapProbe.shown).toBeNull();
  });

  it("while a query runs, only the latest sources are asked for next, and a cleared map does not come back", async () => {
    const { scene, store, app } = setup();
    const host = mount(app, store, scene);
    const real = scene.inspect.geodesicDistances;
    const asks: { sources: readonly number[]; answer: () => void; fail: (e: Error) => void }[] = [];
    (scene.inspect as { geodesicDistances: unknown }).geodesicDistances = (m: MeshHandle, src: readonly number[], o: never) =>
      new Promise((res, rej) =>
        asks.push({ sources: src, answer: () => res(real(m, src, o)), fail: (e) => rej(e) }),
      );
    const runs = geodesicMapProbe.runs;
    const answer = async (k: number) => {
      await act(async () => {
        asks[k].answer();
        await new Promise((r) => setTimeout(r, 0));
      });
    };
    await click(action(host, "geodesic-pick"));
    await tap(app, scene, [[0, 1, 0]]);
    expect(asks.map((a) => a.sources.length)).toEqual([1]);
    expect(host.textContent).toContain("Computing surface distances");
    // Two more sources while it runs: the first answer is stale, and the next ask is for all three.
    await tap(app, scene, [[1, 0, 0]]);
    await tap(app, scene, [[0, 0, 1]]);
    expect(asks).toHaveLength(1);
    await answer(0);
    expect(asks.map((a) => a.sources.length)).toEqual([1, 3]);
    expect(app.getSnapshot().surfaceFieldValues).toBeNull();
    expect(geodesicMapProbe.runs).toBe(runs);
    await answer(1);
    expect(geodesicMapProbe.runs).toBe(runs + 1);
    expect(geodesicMapProbe.shown?.sources).toHaveLength(3);
    expect(host.textContent).not.toContain("Computing surface distances");

    // Cleared while a query runs: nothing shows when it answers.
    await tap(app, scene, [[0, -1, 0]]);
    await click(action(host, "geodesic-clear"));
    await answer(2);
    expect(app.getSnapshot().surfaceFieldValues).toBeNull();
    expect(readoutOf(host)).toBeNull();

    // Cleared in the same tick as the answer, before anything re-renders: it is not shown for a moment either.
    const landed = geodesicMapProbe.runs;
    await tap(app, scene, [[0, 1, 0]]);
    await act(async () => {
      app.setSurfacePath({ sources: [] });
      asks[3].answer();
      await new Promise((r) => setTimeout(r, 0));
    });
    expect(geodesicMapProbe.runs).toBe(landed);
    expect(app.getSnapshot().surfaceFieldValues).toBeNull();

    // A failing query says why, and asks nothing more.
    await tap(app, scene, [[0, 1, 0]]);
    await act(async () => {
      asks[4].fail(new Error("out of memory"));
      await new Promise((r) => setTimeout(r, 0));
    });
    expect(host.querySelector("[data-inspect-geodesic] .report.error")?.textContent).toBe("out of memory");
    expect(asks).toHaveLength(5);
  });

  it("the surface taking another colouring while a query runs is not overridden", async () => {
    const { scene, store, app } = setup();
    const host = mount(app, store, scene);
    const real = scene.inspect.geodesicDistances;
    let answer = () => {};
    (scene.inspect as { geodesicDistances: unknown }).geodesicDistances = (m: MeshHandle, src: readonly number[], o: never) =>
      new Promise((res) => {
        answer = () => res(real(m, src, o));
      });
    await click(action(host, "geodesic-pick"));
    await tap(app, scene, [[0, 1, 0]]);
    // The user shows an inspect map meanwhile: the later action wins.
    await click(q(host, '[data-inspect-map-kind="quality"]'));
    await click(action(host, "inspect-map-show"));
    await frame(store);
    await act(async () => {
      answer();
      await new Promise((r) => setTimeout(r, 0));
    });
    expect(inspectMapProbe.shown?.kind).toBe("quality");
    expect(geodesicMapProbe.shown).toBeNull();
  });

  it("draws iso-lines at a spacing in the display unit", async () => {
    const { scene, store, app, item } = setup();
    const host = mount(app, store, scene);
    await click(action(host, "geodesic-pick"));
    await tap(app, scene, [[0, 1, 0]]);
    await frame(store);
    expect(app.getSnapshot().surfacePath.isoLines).toBeNull();
    const spacingInput = () => q<HTMLInputElement>(host, "[data-geodesic-iso-spacing]");
    expect(spacingInput().disabled).toBe(true);
    await click(q(host, "[data-geodesic-iso]"));
    const auto = autoIsoSpacing(Math.max(...scene.inspect.geodesicDistances(item.mesh, [0])));
    expect(Number(spacingInput().value)).toBeCloseTo(auto, 6);
    const isoAuto = app.getSnapshot().surfacePath.isoLines!.length;
    expect(isoAuto).toBeGreaterThan(0);
    expect(geodesicMapProbe.shown?.isoSpacing).toBeCloseTo(auto, 9);

    // Half the spacing draws more; an empty field goes back to the automatic spacing.
    await type(spacingInput(), String(auto / 2));
    expect(app.getSnapshot().surfacePath.isoLines!.length).toBeGreaterThan(isoAuto * 1.5);
    await type(spacingInput(), "0");
    expect(app.getSnapshot().surfacePath.isoLines!.length).toBe(isoAuto);
    // In inches the same spacing shows as its inch value, and typing takes inches.
    await act(async () => app.setUnits("in"));
    expect(Number(spacingInput().value)).toBeCloseTo(auto / 25.4, 5);
    await type(spacingInput(), String(auto / 25.4 / 2));
    expect(geodesicMapProbe.shown?.isoSpacing).toBeCloseTo(auto / 2, 5);

    await click(q(host, "[data-geodesic-iso]"));
    expect(app.getSnapshot().surfacePath.isoLines).toBeNull();
  });

  it("shares the overlay: the surface field and the inspect maps take it, and it takes it back", async () => {
    const { scene, store, app } = setup();
    const host = mount(app, store, scene);
    await click(action(host, "geodesic-pick"));
    await tap(app, scene, [[0, 1, 0]]);
    await frame(store);
    expect(geodesicMapProbe.shown).not.toBeNull();

    // An inspect map shows over it.
    await click(q(host, '[data-inspect-map-kind="quality"]'));
    const runs = inspectMapProbe.runs;
    await click(action(host, "inspect-map-show"));
    await frame(store);
    expect(inspectMapProbe.runs).toBe(runs + 1);
    expect(inspectMapProbe.shown?.kind).toBe("quality");
    expect(geodesicMapProbe.shown).toBeNull();
    expect(readoutOf(host)).toBeNull();
    // The sources stay: Show map brings the map back over it.
    expect(app.getSnapshot().surfacePath.sources).toHaveLength(1);
    await click(action(host, "geodesic-show"));
    await frame(store);
    expect(geodesicMapProbe.shown).not.toBeNull();
    expect(inspectMapProbe.shown).toBeNull();

    // A surface field mode takes it, and adding a source turns the field off.
    await click(q(host, '[data-inspect-field-mode="meanCurvature"]'));
    await frame(store);
    expect(app.getSnapshot().surfaceField.mode).toBe("meanCurvature");
    expect(geodesicMapProbe.shown).toBeNull();
    await tap(app, scene, [[1, 0, 0]]);
    await frame(store);
    expect(app.getSnapshot().surfaceField.mode).toBe("none");
    expect(geodesicMapProbe.shown?.sources).toHaveLength(2);
  });
});

describe("surface path section (#252)", () => {
  it("draws, undoes and closes a path with its length", async () => {
    const { scene, store, app } = setup();
    const host = mount(app, store, scene);
    expect(action(host, "path-close").disabled).toBe(true);
    await click(action(host, "path-pick"));
    await tap(app, scene, [CAP[0]]);
    expect(pathOf(host)).toBe("1 point · click the next point");
    await tap(app, scene, CAP.slice(1, 3));
    const path = () => app.getSnapshot().surfacePath.path!;
    expect(pathOf(host)).toBe(`3 points · ${formatLength(pathLength(path()), "mm")}`);
    expect(action(host, "path-close").disabled).toBe(false);
    for (const cut of ["path-trim-inside", "path-trim-outside", "path-split"]) {
      expect(action(host, cut).disabled).toBe(true);
    }

    await click(action(host, "path-undo"));
    expect(path().points).toHaveLength(2);
    expect(path().legs).toHaveLength(1);
    expect(action(host, "path-close").disabled).toBe(true);
    await tap(app, scene, CAP.slice(2));
    await click(action(host, "path-close"));
    await act(async () => {
      await surfacePathIdle(app);
    });
    expect(path()).toMatchObject({ closed: true });
    expect(path().legs).toHaveLength(4);
    expect(pathOf(host)).toBe(`4 points · ${formatLength(pathLength(path()), "mm")} · closed loop`);
    await act(async () => app.setUnits("in"));
    expect(pathOf(host)).toBe(`4 points · ${formatLength(pathLength(path()), "in")} · closed loop`);
    for (const cut of ["path-trim-inside", "path-trim-outside", "path-split"]) {
      expect(action(host, cut).disabled).toBe(false);
    }

    // Undo reopens the loop; Clear starts over.
    await click(action(host, "path-undo"));
    expect(path()).toMatchObject({ closed: false });
    expect(action(host, "path-split").disabled).toBe(true);
    await click(action(host, "path-clear"));
    expect(app.getSnapshot().surfacePath.path).toBeNull();
    expect(pathOf(host)).toBeNull();
  });

  it("trims the selected mesh along the loop, one undo step, and clears the picks", async () => {
    const { scene, store, app, item } = setup();
    const host = mount(app, store, scene);
    const whole = tri(scene, item.mesh);
    for (const [button, side] of [
      ["path-trim-inside", "inside"],
      ["path-trim-outside", "outside"],
    ] as const) {
      await click(action(host, "path-pick"));
      await tap(app, scene, [...CAP, CAP[0]]);
      await click(action(host, button));
      // At once, before the frame that would deliver the change: the loop cannot be cut along twice.
      expect(app.getSnapshot().surfacePath.path).toBeNull();
      expect(action(host, "path-split").disabled).toBe(true);
      await frame(store);
      expect(tri(scene, item.mesh)).not.toBe(whole);
      expect(host.querySelector("[data-path-status]")?.textContent).toContain(`kept the ${side} (${tri(scene, item.mesh)} triangles)`);
      // The picks belonged to the shape that was cut.
      expect(app.getSnapshot().surfacePath).toMatchObject({ path: null, armed: null });
      await act(async () => {
        store.undo();
        store.commitFrame();
      });
      expect(scene.canUndo()).toBe(false);
      expect(tri(scene, item.mesh)).toBe(whole);
    }
  });

  it("splits the mesh into two, one undo step, and says what went where", async () => {
    const { scene, store, app, item } = setup();
    const host = mount(app, store, scene);
    const whole = tri(scene, item.mesh);
    await click(action(host, "path-pick"));
    await tap(app, scene, [...CAP, CAP[0]]);
    await click(action(host, "path-split"));
    await frame(store);
    expect(app.getSnapshot().items.map((i) => i.name)).toEqual(["Ball", "Other", "Ball inside"]);
    const inside = app.getSnapshot().items[2];
    const status = host.querySelector("[data-path-status]")?.textContent ?? "";
    expect(status).toContain(`"Ball" keeps the outside (${tri(scene, item.mesh)} triangles)`);
    expect(status).toContain(`"Ball inside" holds the inside (${tri(scene, inside.mesh)})`);
    expect(store.getSnapshot().selection).toBe(item.mesh);
    await act(async () => {
      store.undo();
      store.commitFrame();
    });
    expect(scene.canUndo()).toBe(false);
    expect(tri(scene, item.mesh)).toBe(whole);
    expect(tri(scene, inside.mesh)).toBe(0);
  });

  it("a loop that separates nothing says so and keeps the path and the map", async () => {
    const scene = new Scene();
    const store = createSceneStore(scene);
    const app = new AppState();
    cleanups.push(() => {
      store.dispose();
      scene.free();
    });
    const prim = generatePrimitive({ kind: "torus", majorRadius: 1, minorRadius: 0.35 });
    const ring = addMesh(store, app, "Ring", Float64Array.from(prim.positions), prim.indices);
    store.select(ring.mesh);
    store.commitFrame();
    scene.clearHistory();
    const host = mount(app, store, scene);
    // Once around the tube, off the mesh lines.
    const dirs = [0, 0.3, 0, -0.3, 0.2].map((a, k): { at: V3; toward: V3 } => {
      const t = (k * 2 * Math.PI) / 5;
      const centre: V3 = [Math.cos(a), 0, Math.sin(a)];
      const n: V3 = [Math.cos(t) * Math.cos(a), Math.sin(t), Math.cos(t) * Math.sin(a)];
      return { at: [centre[0] + 0.6 * n[0], 0.6 * n[1], centre[2] + 0.6 * n[2]], toward: [-n[0], -n[1], -n[2]] };
    });
    await click(action(host, "geodesic-pick"));
    const shoot = (d: { at: V3; toward: V3 }) => scene.raycast(d.at, d.toward)!;
    await act(async () => {
      void queueSurfaceTap(app, scene, Promise.resolve(shoot(dirs[0])));
      await surfacePathIdle(app);
    });
    await frame(store);
    expect(geodesicMapProbe.shown).not.toBeNull();
    // A path through the faces (the heat method's legs leave the mesh lines).
    await click(q(host, '[data-geodesic-method="heatMethod"]'));
    await click(action(host, "path-pick"));
    await act(async () => {
      for (const d of [...dirs, dirs[0]]) {
        void queueSurfaceTap(app, scene, Promise.resolve(shoot(d)));
      }
      await surfacePathIdle(app);
    });
    await frame(store);
    const before = surfaceStats(scene.views(ring.mesh).positions(), scene.views(ring.mesh).indices());
    const path = app.getSnapshot().surfacePath.path;
    expect(path?.closed).toBe(true);
    const shown = geodesicMapProbe.shown;
    for (const cut of ["path-trim-inside", "path-trim-outside", "path-split"]) {
      await click(action(host, cut));
      await frame(store);
      await frame(store);
      expect(host.querySelector("[data-path-error]")?.textContent).toBe(NO_CUT_MESSAGE);
      // Nothing changed, and nothing is lost: the same path, the sources and the map.
      expect(surfaceStats(scene.views(ring.mesh).positions(), scene.views(ring.mesh).indices())).toEqual(before);
      expect(app.getSnapshot().surfacePath.path).toBe(path);
      expect(app.getSnapshot().surfacePath.sources).toHaveLength(1);
      expect(geodesicMapProbe.shown).toEqual(shown);
      expect(app.getSnapshot().items.map((i) => i.name)).toEqual(["Ring"]);
      expect(scene.canUndo()).toBe(false);
    }
    // After that the watcher is back: an edit clears the picks.
    await act(async () => {
      scene.edit(ring.mesh, [{ op: "setPositions", vertices: [0], positions: Float64Array.from([2, 2, 2]) }]);
      store.commitFrame();
    });
    expect(app.getSnapshot().surfacePath.path).toBeNull();
  });

  it("a failed cut on a Worker scene, whose taken-back changes arrive later, keeps the picks too", async () => {
    const scene: WorkerScene = await createWorkerScene({ createWorker: () => new NodeWorker() });
    const store = createSceneStore(scene);
    const app = new AppState();
    cleanups.push(() => {
      store.dispose();
      scene.terminate();
    });
    const prim = generatePrimitive({ kind: "torus", majorRadius: 1, minorRadius: 0.35 });
    const ring = addMesh(store, app, "Ring", Float64Array.from(prim.positions), prim.indices);
    const settle = async () => {
      for (let k = 0; k < 500; k++) {
        await frame(store, 2);
        if (store.getSnapshot().status !== "computing") {
          return;
        }
      }
      throw new Error("the store did not settle");
    };
    await settle();
    act(() => store.select(ring.mesh));
    const host = document.createElement("div");
    document.body.appendChild(host);
    const root = createRoot(host);
    act(() => {
      root.render(
        <SceneProvider store={store}>
          <InspectPanel app={app} scene={scene} />
        </SceneProvider>,
      );
    });
    cleanups.push(() => {
      act(() => root.unmount());
      host.remove();
    });
    await settle();
    await click(q(host, '[data-geodesic-method="heatMethod"]'));
    await click(action(host, "path-pick"));
    const wobble = [0, 0.3, 0, -0.3, 0.2];
    const shots = [0, 1, 2, 3, 4, 0].map((k) => {
      const a = wobble[k];
      const t = (k * 2 * Math.PI) / 5;
      const n: V3 = [Math.cos(t) * Math.cos(a), Math.sin(t), Math.cos(t) * Math.sin(a)];
      return scene.raycast([Math.cos(a) + 0.6 * n[0], 0.6 * n[1], Math.sin(a) + 0.6 * n[2]], [-n[0], -n[1], -n[2]]);
    });
    // The canvas keeps a frame loop running meanwhile, as the app does: the taken-back changes must not slip through it.
    let looping = true;
    const loop = (async () => {
      while (looping) {
        await new Promise((r) => setTimeout(r, 1));
        store.commitFrame();
      }
    })();
    await act(async () => {
      for (const shot of shots) {
        void queueSurfaceTap(app, scene, shot as Promise<RayHit | null>);
      }
      await surfacePathIdle(app);
    });
    const path = app.getSnapshot().surfacePath.path;
    expect(path?.closed).toBe(true);
    const before = surfaceStats(scene.views(ring.mesh).positions(), scene.views(ring.mesh).indices());
    for (const cut of ["path-trim-inside", "path-split"]) {
      await click(action(host, cut));
      for (let k = 0; k < 200 && host.querySelector("[data-path-error]") === null; k++) {
        await act(async () => {
          await new Promise((r) => setTimeout(r, 5));
        });
      }
      expect(host.querySelector("[data-path-error]")?.textContent).toBe(NO_CUT_MESSAGE);
      for (let k = 0; k < 5; k++) {
        await frame(store, 5);
      }
      expect(app.getSnapshot().surfacePath.path).toBe(path);
      expect(surfaceStats(scene.views(ring.mesh).positions(), scene.views(ring.mesh).indices())).toEqual(before);
      expect(store.getSnapshot().history.canUndo).toBe(false);
      expect(app.getSnapshot().items.map((i) => i.name)).toEqual(["Ring"]);
    }
    looping = false;
    await loop;
  }, 60_000);

  it("says when the accurate method is slow on the selected mesh", async () => {
    const { scene, store, app, item } = setup();
    const host = mount(app, store, scene);
    expect(host.querySelector("[data-geodesic-slow]")).toBeNull();
    await click(q(host, '[data-geodesic-method="heatMethod"]'));
    expect(host.querySelector("[data-geodesic-slow]")).toBeNull();
    // The list's triangle count is the one it was loaded with.
    (item as { triangles: number }).triangles = 1_046_528;
    await act(async () => app.setUnits("in"));
    expect(host.querySelector("[data-geodesic-slow]")?.textContent).toContain("1046528");
    await click(q(host, '[data-geodesic-method="dijkstra"]'));
    expect(host.querySelector("[data-geodesic-slow]")).toBeNull();
  });

  it("a locked mesh can be measured but not cut", async () => {
    const { scene, store, app, item } = setup();
    const host = mount(app, store, scene);
    await click(action(host, "path-pick"));
    await tap(app, scene, [...CAP, CAP[0]]);
    expect(action(host, "path-split").disabled).toBe(false);
    await act(async () => app.setMeshLocked(item.mesh, true));
    expect(action(host, "path-split").disabled).toBe(true);
    expect(host.textContent).toContain("The mesh is locked");
  });
});

describe("the picks belong to their mesh", () => {
  /** A closed path and a source on the selected mesh, with the panel mounted. */
  async function picked(earlier?: (ctx: ReturnType<typeof setup>) => void) {
    const ctx = setup();
    earlier?.(ctx);
    ctx.store.commitFrame();
    const host = mount(ctx.app, ctx.store, ctx.scene);
    await click(action(host, "geodesic-pick"));
    await tap(ctx.app, ctx.scene, [[0, 1, 0]]);
    await frame(ctx.store);
    await click(action(host, "path-pick"));
    await tap(ctx.app, ctx.scene, CAP.slice(0, 3));
    expect(ctx.app.getSnapshot().surfacePath.path?.points).toHaveLength(3);
    expect(ctx.app.getSnapshot().surfacePath.sources).toHaveLength(1);
    return { ...ctx, host };
  }
  const kept = (app: AppState) => {
    const s = app.getSnapshot().surfacePath;
    expect(s.path?.points).toHaveLength(3);
    expect(s.sources).toHaveLength(1);
  };
  const gone = (app: AppState) => {
    const s = app.getSnapshot().surfacePath;
    expect(s.path).toBeNull();
    expect(s.sources).toHaveLength(0);
    expect(s.isoLines).toBeNull();
  };

  it("an edit, a move, its undo and a removal clear them, and the map with them", async () => {
    for (const change of ["edit", "move", "undo", "remove"] as const) {
      const { scene, store, app, item } = await picked(({ scene: s, item: i }) => {
        if (change === "undo") {
          s.edit(i.mesh, [{ op: "setPositions", vertices: [0], positions: Float64Array.from([0, 1.1, 0]) }]);
        }
      });
      await act(async () => {
        if (change === "edit") {
          scene.edit(item.mesh, [{ op: "setPositions", vertices: [0], positions: Float64Array.from([0, 1.1, 0]) }]);
        } else if (change === "move") {
          store.setPlacement(item.mesh, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [1, 0, 0] });
        } else if (change === "undo") {
          expect(scene.canUndo()).toBe(true);
          store.undo();
        } else {
          app.removeMesh(store, item);
        }
        store.commitFrame();
      });
      await frame(store);
      gone(app);
      expect(app.getSnapshot().surfaceFieldValues).toBeNull();
      expect(geodesicMapProbe.shown).toBeNull();
      for (const fn of cleanups.splice(0).reverse()) {
        fn();
      }
    }
  });

  it("selecting another mesh clears them and disarms the pick", async () => {
    const { store, app, other } = await picked();
    await act(async () => {
      store.select(other.mesh);
      store.commitFrame();
    });
    gone(app);
    expect(app.getSnapshot().surfacePath).toMatchObject({ mesh: null, armed: null });
  });

  it("camera drags, colours and frames without a change keep them", async () => {
    const { store, app, item } = await picked();
    // Camera drags: the gesture is the orbit's, and nothing of the scene changes.
    view.dragging = true;
    for (const l of view.dragListeners) {
      l(true);
    }
    view.dragging = false;
    for (const l of view.dragListeners) {
      l(false);
    }
    await frame(store);
    await act(async () => {
      store.setDisplay(item.mesh, { coloring: null, visible: true });
      store.commitFrame();
    });
    await frame(store);
    kept(app);
    expect(app.getSnapshot().surfacePath.armed).toBe("path");
  });

  it("another mesh's changes keep them", async () => {
    const { scene, store, app, other } = await picked();
    await act(async () => {
      scene.edit(other.mesh, [{ op: "setPositions", vertices: [0], positions: Float64Array.from([4, 1.2, 0]) }]);
      store.setPlacement(other.mesh, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0, 1, 0] });
      store.commitFrame();
    });
    await frame(store);
    kept(app);
  });
});

describe("canvas taps (#252)", () => {
  /** A canvas of 200 x 200 CSS px and a camera on +z looking at the origin. */
  function canvas(camera: THREE.Camera) {
    const el = document.createElement("canvas");
    el.getBoundingClientRect = () => ({ left: 0, top: 0, width: 200, height: 200, right: 200, bottom: 200, x: 0, y: 0, toJSON: () => ({}) });
    document.body.appendChild(el);
    three.current = { camera, gl: { domElement: el }, raycaster: new THREE.Raycaster() };
    cleanups.push(() => el.remove());
    return el;
  }
  function camera(): THREE.PerspectiveCamera {
    const c = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
    c.position.set(0, 0, 4);
    c.lookAt(0, 0, 0);
    c.updateMatrixWorld();
    return c;
  }
  function fire(el: HTMLElement, type: string, x: number, y: number, extra: Record<string, unknown> = {}) {
    const e = new MouseEvent(type, { clientX: x, clientY: y, button: (extra.button as number) ?? 0, bubbles: true });
    for (const [k, v] of Object.entries({ pointerId: 1, pointerType: "mouse", ...extra })) {
      if (k !== "button") {
        Object.defineProperty(e, k, { value: v });
      }
    }
    el.dispatchEvent(e);
  }
  function Pointer(props: { app: AppState; store: SceneStore }) {
    useSurfacePickPointer(props.app, props.store);
    return null;
  }
  async function pointer(app: AppState, store: SceneStore) {
    const host = document.createElement("div");
    const root = createRoot(host);
    await act(async () => {
      root.render(<Pointer app={app} store={store} />);
    });
    cleanups.push(() => act(() => root.unmount()));
  }
  async function settleTaps(app: AppState) {
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
      await surfacePathIdle(app);
    });
  }

  it("a tap on the picked mesh lands; a drag, another button, a pinch and a miss do not", async () => {
    const { store, app, item } = setup();
    const el = canvas(camera());
    await pointer(app, store);
    // Nothing armed: no listener takes the tap.
    fire(el, "pointerdown", 100, 100);
    fire(el, "pointerup", 100, 100);
    await settleTaps(app);
    expect(app.getSnapshot().surfacePath.sources).toHaveLength(0);

    await act(async () => app.setSurfacePath({ mesh: item.mesh, armed: "sources" }));
    const tapAt = async (x: number, y: number, up: [number, number] = [x, y], extra: Record<string, unknown> = {}) => {
      fire(el, "pointerdown", x, y, extra);
      fire(el, "pointerup", up[0], up[1], extra);
      await settleTaps(app);
    };
    await tapAt(100, 100);
    expect(app.getSnapshot().surfacePath.sources).toHaveLength(1);
    // The camera orbit's drag, the other buttons, a two-finger gesture and empty space.
    await tapAt(100, 100, [140, 100]);
    await tapAt(110, 110, [110, 110], { button: 2 });
    view.multiTouch = true;
    await tapAt(120, 100);
    view.multiTouch = false;
    await tapAt(2, 2);
    expect(app.getSnapshot().surfacePath.sources).toHaveLength(1);
    await tapAt(120, 100);
    expect(app.getSnapshot().surfacePath.sources).toHaveLength(2);
  });

  it("looks past a mesh in front of the picked one, and stays put while the camera moves", async () => {
    const { scene, store, app, item, other } = setup();
    // "Other" between the camera and the ball.
    await act(async () => {
      store.setPlacement(other.mesh, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [-4, 0, 2] });
      store.commitFrame();
    });
    const cam = camera();
    const el = canvas(cam);
    await pointer(app, store);
    await act(async () => app.setSurfacePath({ mesh: item.mesh, armed: "path" }));
    expect(scene.raycast([0, 0, 4], [0, 0, -1])?.mesh).toBe(other.mesh);
    fire(el, "pointerdown", 100, 100);
    fire(el, "pointerup", 100, 100);
    await settleTaps(app);
    let path = app.getSnapshot().surfacePath.path;
    expect(path?.points).toHaveLength(1);
    const first = path!.points[0].point;
    expect(first[2]).toBeCloseTo(1, 1);
    // Orbit to the side (a drag) and tap again: the first point has not moved, the second is on the new side.
    fire(el, "pointerdown", 100, 100);
    fire(el, "pointermove", 160, 100);
    fire(el, "pointerup", 160, 100);
    cam.position.set(4, 0, 0);
    cam.lookAt(0, 0, 0);
    cam.updateMatrixWorld();
    fire(el, "pointerdown", 100, 100);
    fire(el, "pointerup", 100, 100);
    await settleTaps(app);
    path = app.getSnapshot().surfacePath.path;
    expect(path?.points).toHaveLength(2);
    expect(path!.points[0].point).toEqual(first);
    expect(path!.points[1].point[0]).toBeCloseTo(1, 1);
  });
});

describe("what the viewport draws (#252)", () => {
  it("markers, the path over the surface and through it, its label, and the iso-lines", async () => {
    const { scene, store, app, item } = setup();
    const el = document.createElement("canvas");
    three.current = { camera: new THREE.PerspectiveCamera(), gl: { domElement: el }, raycaster: new THREE.Raycaster() };
    const added = vi.spyOn(THREE.Object3D.prototype, "add");
    const host = document.createElement("div");
    const root = createRoot(host);
    const layer = () => (
      <SceneProvider store={store}>
        <SurfacePathLayer app={app} store={store} />
      </SceneProvider>
    );
    await act(async () => root.render(layer()));
    cleanups.push(() => act(() => root.unmount()));
    // The group's own add call (the layer's six parts) is on `added`.
    const group = added.mock.contexts[added.mock.calls.findIndex((c) => c.length === 6)] as THREE.Group;
    added.mockRestore();
    const [iso, edge, line, through, markers, label] = group.children as [
      THREE.LineSegments,
      LineSegments2,
      LineSegments2,
      LineSegments2,
      THREE.Group,
      THREE.Sprite,
    ];
    // A line's segments, and none while it is hidden.
    const segmentsOf = (l: LineSegments2) => (l.visible ? l.geometry.instanceCount : 0);
    const isoPoints = () => (iso.visible ? iso.geometry.getAttribute("position").count : 0);
    const drawn = () => [isoPoints(), segmentsOf(edge), segmentsOf(line), segmentsOf(through), markers.children.length];
    expect(drawn()).toEqual([0, 0, 0, 0, 0]);
    expect(label.visible).toBe(false);

    await act(async () => app.setSurfacePath({ mesh: item.mesh, armed: "path" }));
    await tap(app, scene, CAP.slice(0, 3));
    const path = app.getSnapshot().surfacePath.path!;
    const segments = path.legs.reduce((n, l) => n + l.positions.length / 3 - 1, 0);
    // Drawn just off the surface on both sides, with a wider dark edge under it, and once, faintly, through the mesh.
    expect(segmentsOf(line)).toBe(2 * segments);
    expect(segmentsOf(edge)).toBe(2 * segments);
    expect(segmentsOf(through)).toBe(segments);
    expect(through.material.depthTest).toBe(false);
    expect(line.material.depthTest).toBe(true);
    expect(edge.material.linewidth).toBeGreaterThan(line.material.linewidth);
    expect(edge.renderOrder).toBeLessThan(line.renderOrder);
    const starts = line.geometry.getAttribute("instanceStart") as THREE.InterleavedBufferAttribute;
    expect(Array.from((starts.data.array as Float32Array).subarray(0, 6 * segments))).toEqual(Array.from(pathSegments(path, 0.002)));
    // Two balls per point: the dark edge behind and the point.
    expect(markers.children).toHaveLength(2 * 3);
    expect(markers.children[0].position.toArray()).toEqual(path.points[0].point);
    expect(label.visible).toBe(true);

    // Iso-lines and sources draw too, and clearing empties it all.
    await act(async () =>
      app.setSurfacePath({ isoLines: Float32Array.from([0, 0, 0, 1, 0, 0]), sources: [{ slot: 0, point: [0, 1, 0] }] }),
    );
    expect(isoPoints()).toBe(2);
    expect(markers.children).toHaveLength(2 * 4);
    await act(async () => app.setSurfacePath({ sources: [], path: null, isoLines: null }));
    expect(drawn()).toEqual([0, 0, 0, 0, 0]);
    expect(label.visible).toBe(false);

    // Replacing lines frees the old buffers instead of leaving them to the GPU.
    const freed = vi.fn();
    iso.geometry.addEventListener("dispose", freed);
    line.geometry.addEventListener("dispose", freed);
    await act(async () => app.setSurfacePath({ isoLines: Float32Array.from([0, 0, 0, 1, 0, 0]) }));
    await tap(app, scene, CAP.slice(0, 2));
    expect(freed.mock.calls.length).toBeGreaterThanOrEqual(3);
  });
});

describe("the Esc key and the hint (#252)", () => {
  it("Esc stops the pick before it deselects", async () => {
    const { store, app, item } = setup();
    function Keys() {
      const [, setMode] = useState<"translate" | "rotate" | "scale">("translate");
      useKeyboardShortcuts({
        app,
        select: (m) => store.select(m),
        selection: item.mesh,
        undo: () => {},
        redo: () => {},
        focusOn: () => {},
        setMode,
        helpOpen: false,
        setHelpOpen: () => {},
        tourStep: null,
        endTour: () => {},
      });
      return null;
    }
    const host = document.createElement("div");
    const root = createRoot(host);
    await act(async () => root.render(<Keys />));
    cleanups.push(() => act(() => root.unmount()));
    await act(async () => app.setSurfacePath({ mesh: item.mesh, armed: "path" }));
    const esc = () => act(async () => window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" })));
    await esc();
    expect(app.getSnapshot().surfacePath.armed).toBeNull();
    expect(store.getSnapshot().selection).toBe(item.mesh);
    await esc();
    expect(store.getSnapshot().selection).toBeNull();
  });

  it("the hint says what a click does", () => {
    const state = {
      sculpting: false,
      maskPaint: false,
      selectTool: null,
      deformTool: null,
      printTool: null,
      picking: false,
      selected: true,
      coarse: false,
    };
    expect(viewportHint({ ...state, surfacePick: "sources" })).toContain("distance source");
    expect(viewportHint({ ...state, surfacePick: "path" })).toContain("draw a path");
    expect(viewportHint({ ...state, surfacePick: null })).toContain("gizmo");
  });
});
