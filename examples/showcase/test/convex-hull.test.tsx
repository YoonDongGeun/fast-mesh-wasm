// @vitest-environment jsdom
// The Convex Hull actions: which meshes the Modeling panel's hull is built
// from, its name and report line; and the wiring of the Mesh Edit and Points
// buttons on a real Wasm Scene, SceneStore and AppState (a new listed and
// selected mesh, one undo step, the sources untouched).
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { MeshHandle, SceneLike } from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act, createElement, type ReactElement } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { init, packPointCloud, Scene } from "../../../packages/fast-mesh-wasm/dist/web.js";
import { AppState, type Item } from "../src/app/state/app-state";
import { ModelingPanel } from "../src/features/mesh-edit/ui/ModelingPanel";
import { describeHull, hullMeshName, hullSources } from "../src/features/mesh-edit/ux/use-convex-hull";
import { PointsPanel } from "../src/features/points/ui/PointsPanel";
import { countLiveTriangles } from "../src/features/select/ux/face-sets";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeAll(() =>
  init(readFileSync(resolve(__dirname, "../../../packages/fast-mesh-wasm/dist/wasm/fast_mesh_wasm_bg.wasm"))),
);

const cleanups: (() => void)[] = [];

afterEach(() => {
  while (cleanups.length > 0) {
    cleanups.pop()?.();
  }
});

const handle = (n: number) => ({ __mesh: n }) as unknown as MeshHandle;
const fakeItem = (n: number, name = `M${n}`) => ({ id: n, mesh: handle(n), name }) as unknown as Item;

describe("convex hull sources, name and report", () => {
  it("takes the multi-selection while it starts at the active mesh, else the active mesh", () => {
    const items = [fakeItem(1), fakeItem(2), fakeItem(3)];
    const [a, b, c] = items;
    expect(hullSources(items, [], a)).toEqual([a]);
    expect(hullSources(items, [a.mesh], a)).toEqual([a]);
    expect(hullSources(items, [a.mesh, c.mesh], a)).toEqual([a, c]);
    // In click order, without meshes that are gone.
    expect(hullSources(items, [b.mesh, handle(9), a.mesh], b)).toEqual([b, a]);
    // A list that no longer starts at the active mesh does not count.
    expect(hullSources(items, [b.mesh, c.mesh], a)).toEqual([a]);
    expect(hullSources(items, [a.mesh, b.mesh], null)).toEqual([]);
    expect(hullSources([], [], null)).toEqual([]);
  });

  it("names the hull after its source and reports its counts", () => {
    expect(hullMeshName([fakeItem(1, "Bracket")])).toBe("Bracket hull");
    expect(hullMeshName([fakeItem(1), fakeItem(2), fakeItem(3)])).toBe("Hull of 3 meshes");
    expect(describeHull({ name: "H", inputPoints: 1_000_000, vertices: 1234, triangles: 2464 })).toBe(
      `H: ${(1234).toLocaleString()} vertices, ${(2464).toLocaleString()} triangles from ${(1_000_000).toLocaleString()} points`,
    );
  });
});

function makeBox(c: [number, number, number], h: number): { positions: Float64Array; indices: Uint32Array } {
  const positions: number[] = [];
  for (let i = 0; i < 8; i++) {
    positions.push(c[0] + (i & 1 ? h : -h), c[1] + (i & 2 ? h : -h), c[2] + (i & 4 ? h : -h));
  }
  const indices = Uint32Array.from([
    0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5,
    3, 7, 5,
  ]);
  return { positions: Float64Array.from(positions), indices };
}

function harness(): { scene: Scene; store: SceneStore; app: AppState } {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  cleanups.push(() => {
    store.dispose();
    scene.free();
  });
  return { scene, store, app };
}

function addBox(store: SceneStore, app: AppState, name: string, c: [number, number, number]): Item {
  const { positions, indices } = makeBox(c, 1);
  return app.addMesh(store, { name, positions, indices, center: c, radius: 2, volumeId: null, preset: null });
}

function mount(store: SceneStore, element: ReactElement): HTMLDivElement {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(createElement(SceneProvider, { store }, element));
  });
  cleanups.push(() => {
    act(() => root.unmount());
    container.remove();
  });
  return container;
}

async function click(el: Element | null): Promise<void> {
  expect(el).not.toBeNull();
  await act(async () => {
    el!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    for (let i = 0; i < 4; i++) await Promise.resolve();
  });
}

const live = (scene: SceneLike, mesh: MeshHandle) => countLiveTriangles(scene.views(mesh).indices());

describe("convex hull actions", () => {
  it("Mesh Edit adds the hull of the selected meshes as a new mesh, one undo step", async () => {
    const { scene, store, app } = harness();
    const a = addBox(store, app, "A", [0, 0, 0]);
    const b = addBox(store, app, "B", [4, 0, 0]);
    store.select(a.mesh);
    app.setMulti([a.mesh, b.mesh]);
    store.commitFrame();

    const panel = mount(store, createElement(ModelingPanel, { app }));
    expect(panel.querySelector("[data-modeling-hull-sources]")?.textContent).toContain("2 selected meshes");
    await click(panel.querySelector("[data-modeling-hull-run]"));

    const items = app.getSnapshot().items;
    expect(items.length).toBe(3);
    const hull = items[2];
    expect(hull.name).toBe("Hull of 2 meshes");
    expect(store.getSnapshot().selection).toBe(hull.mesh);
    expect(live(scene, hull.mesh)).toBe(12);
    expect(panel.querySelector("[data-modeling-report]")?.textContent).toContain(
      "Hull of 2 meshes: 8 vertices, 12 triangles from 16 points",
    );

    // One undo step empties the hull; the sources stay; redo fills it again.
    store.undo();
    store.commitFrame();
    expect(live(scene, hull.mesh)).toBe(0);
    expect(live(scene, a.mesh)).toBe(12);
    expect(live(scene, b.mesh)).toBe(12);
    store.redo();
    store.commitFrame();
    expect(live(scene, hull.mesh)).toBe(12);
  });

  it("Mesh Edit reports a flat selection as an error and adds nothing", async () => {
    const { store, app } = harness();
    const flat = app.addMesh(store, {
      name: "Sheet",
      positions: Float64Array.from([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]),
      indices: Uint32Array.from([0, 1, 2, 0, 2, 3]),
      center: [0.5, 0.5, 0],
      radius: 1,
      volumeId: null,
      preset: null,
    });
    store.select(flat.mesh);
    store.commitFrame();
    const panel = mount(store, createElement(ModelingPanel, { app }));
    await click(panel.querySelector("[data-modeling-hull-run]"));
    const report = panel.querySelector("[data-modeling-report]");
    expect(report?.className).toContain("error");
    expect(report?.textContent).toContain("coplanar");
    expect(app.getSnapshot().items.length).toBe(1);
  });

  it("Points adds the hull of the active point cloud as a new mesh", async () => {
    const { scene, store, app } = harness();
    const { positions } = makeBox([10, 0, 0], 2);
    const cloud = packPointCloud(Float32Array.from([...positions, 10, 0, 0, 10.5, 0.5, 0.5]));
    app.updateWave6((prev) => ({
      ...prev,
      metrology: { ...prev.metrology, pointCloud: cloud, pointCloudReport: null },
    }));
    const panel = mount(store, createElement(PointsPanel, { app, store }));
    await click(panel.querySelector('[data-action="points-convex-hull"]'));

    const items = app.getSnapshot().items;
    expect(items.map((i) => i.name)).toEqual(["Point cloud hull"]);
    expect(live(scene, items[0].mesh)).toBe(12);
    expect(scene.bounds(items[0].mesh)).not.toBeNull();
    expect(app.getSnapshot().wave6.metrology.pointCloudReport).toContain("8 vertices, 12 triangles from 10 points");
    // The cloud stays.
    expect(app.getSnapshot().wave6.metrology.pointCloud?.count).toBe(10);
    store.undo();
    store.commitFrame();
    expect(live(scene, items[0].mesh)).toBe(0);
  });
});
