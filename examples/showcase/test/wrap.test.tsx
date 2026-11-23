// @vitest-environment jsdom
// The Wrap action (#263): its settings and options, the default gap, the plan
// and report lines, and the wiring of the repair panel's section on a real
// Wasm Scene, SceneStore and AppState (the planned grid, a new listed and
// selected mesh, one undo step, the sources untouched, errors reported).
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { MeshHandle, SceneLike, WrapPlan } from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act, createElement, type ReactElement } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { init, Scene } from "../../../packages/fast-mesh-wasm/dist/web.js";
import { AppState, type Item } from "../src/app/state/app-state";
import { WrapSection } from "../src/features/mesh-repair/ui/WrapSection";
import {
  DEFAULT_WRAP_SETTINGS,
  defaultWrapGap,
  describeWrap,
  describeWrapPlan,
  detailVoxelSize,
  formatBytes,
  worldExtent,
  wrapDetailInput,
  wrapMeshName,
  wrapOptions,
} from "../src/features/mesh-repair/ux/wrap";
import { countLiveTriangles } from "../src/features/select/ux/face-sets";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeAll(() =>
  init(readFileSync(resolve(__dirname, "../../../packages/fast-mesh-wasm/dist/wasm/fast_mesh_wasm_bg.wasm"))),
);

const cleanups: (() => void)[] = [];

afterEach(() => {
  vi.useRealTimers();
  while (cleanups.length > 0) {
    cleanups.pop()?.();
  }
});

const PLAN: WrapPlan = {
  dims: [143, 113, 113],
  voxelSize: 0.5,
  radius: 1,
  voxels: 1_825_967,
  bytes: 31_041_439,
  maxVoxels: 16_777_216,
  fits: true,
};

describe("wrap settings, options and lines", () => {
  it("turns settings into engine options, with the default gap until one is typed", () => {
    expect(wrapOptions(DEFAULT_WRAP_SETTINGS, 0.4)).toEqual({ gap: 0.4, project: true, detail: 128 });
    expect(wrapOptions({ ...DEFAULT_WRAP_SETTINGS, gap: 0, project: false, detail: 64 }, 0.4)).toEqual({
      gap: 0,
      project: false,
      detail: 64,
    });
    expect(wrapOptions({ ...DEFAULT_WRAP_SETTINGS, sizing: "voxel", voxelSize: 0.25 }, 1)).toEqual({
      gap: 1,
      project: true,
      voxelSize: 0.25,
    });
    // No usable voxel size yet: the detail.
    expect(wrapOptions({ ...DEFAULT_WRAP_SETTINGS, sizing: "voxel", voxelSize: null }, 1)).toEqual({
      gap: 1,
      project: true,
      detail: 128,
    });
  });

  it("starts the gap at two percent of the size, to two digits", () => {
    expect(defaultWrapGap(100)).toBe(2);
    expect(defaultWrapGap(1234)).toBe(25);
    expect(defaultWrapGap(0.333)).toBe(0.0067);
    expect(defaultWrapGap(0)).toBe(0);
    expect(detailVoxelSize(256, 128)).toBe(2);
  });

  it("reads typed details in the engine's range", () => {
    expect(wrapDetailInput("200")).toBe(200);
    expect(wrapDetailInput("63.6")).toBe(64);
    expect(wrapDetailInput("1")).toBe(8);
    expect(wrapDetailInput("99999")).toBe(4096);
    expect(wrapDetailInput("x")).toBe(128);
  });

  it("describes the planned grid in the display unit, and the limit it is over", () => {
    expect(describeWrapPlan(PLAN, "mm")).toBe(
      "Grid 143 × 113 × 113: 1.8M voxels of 0.5 mm, about 30 MB. Closes gaps under 2 mm.",
    );
    const inches = describeWrapPlan({ ...PLAN, voxelSize: 25.4, radius: 25.4 }, "in");
    expect(inches).toContain("voxels of 1 in");
    expect(inches).toContain("gaps under 2 in");
    const over = describeWrapPlan(
      { ...PLAN, dims: [2407, 2407, 2407], voxels: 13_945_313_143, bytes: 237_070_323_431, fits: false },
      "mm",
    );
    expect(over).toContain("2407 × 2407 × 2407: 13.9G voxels");
    expect(over).toContain("220.8 GB");
    expect(over).toContain("over the limit of 16.8M voxels");
    expect(formatBytes(1000)).toBe("1 MB");
  });

  it("names the wrap after its sources and reports its counts", () => {
    const item = (name: string) => ({ name }) as unknown as Item;
    expect(wrapMeshName([item("Bracket")])).toBe("Bracket wrap");
    expect(wrapMeshName([item("A"), item("B")])).toBe("Wrap of 2 meshes");
    const r = { name: "W", vertices: 49_662, triangles: 99_320, projected: 48_286, plan: PLAN };
    expect(describeWrap(r)).toBe(
      "W: 49,662 vertices, 99,320 triangles, 48,286 vertices on the input (grid 143 × 113 × 113, 30 MB)",
    );
    expect(describeWrap({ ...r, projected: 0 })).not.toContain("on the input");
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
  app.follow(store);
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

function mount(store: SceneStore, element: () => ReactElement) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  const render = () =>
    act(() => {
      root.render(createElement(SceneProvider, { store }, element()));
    });
  render();
  cleanups.push(() => {
    act(() => root.unmount());
    container.remove();
  });
  return { container, render };
}

/** Lets the debounced plan run and its promise land. */
async function rest(ms = 400): Promise<void> {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

async function click(el: Element | null): Promise<void> {
  expect(el).not.toBeNull();
  await act(async () => {
    el!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await vi.advanceTimersByTimeAsync(400);
  });
}

async function type(el: HTMLInputElement | null, value: string): Promise<void> {
  expect(el).not.toBeNull();
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(el, value);
    el!.dispatchEvent(new Event("input", { bubbles: true }));
    await vi.advanceTimersByTimeAsync(400);
  });
}

const live = (scene: SceneLike, mesh: MeshHandle) => countLiveTriangles(scene.views(mesh).indices());

/** The section on the store's selected mesh. */
function section(store: SceneStore, app: AppState) {
  return () => {
    const items = app.getSnapshot().items;
    const activeItem = items.find((i) => i.mesh === store.getSnapshot().selection) ?? null;
    return createElement(WrapSection, { app, store, items, activeItem, units: "mm", disabled: false });
  };
}

describe("wrap section", () => {
  it("plans the grid, and wraps the selected meshes as a new mesh, one undo step", async () => {
    vi.useFakeTimers();
    const { scene, store, app } = harness();
    const a = addBox(store, app, "A", [0, 0, 0]);
    const b = addBox(store, app, "B", [2.02, 0, 0]);
    store.select(a.mesh);
    app.setMulti([a.mesh, b.mesh]);
    store.commitFrame();
    expect(worldExtent(scene, [a, b])).toBeCloseTo(4.02, 5);

    const { container, render } = mount(store, section(store, app));
    await rest();
    const $ = <T extends Element>(s: string) => container.querySelector<T>(s);
    expect($("[data-wrap-sources]")?.textContent).toContain("2 selected meshes");
    // Two percent of 4.02 wide: the 0.02 gap between the boxes closes.
    expect($<HTMLInputElement>("[data-wrap-gap]")?.value).toBe("0.08");
    expect($("[data-wrap-plan]")?.textContent).toMatch(/^Grid 138 × \d+ × \d+: .* voxels of .* mm, about \d+ MB\. Closes/);

    await type($<HTMLInputElement>("[data-wrap-detail]"), "48");
    expect($("[data-wrap-plan]")?.textContent).toContain("Grid 57 × ");
    await click($("[data-wrap-run]"));

    const items = app.getSnapshot().items;
    expect(items.map((i) => i.name)).toEqual(["A", "B", "Wrap of 2 meshes"]);
    const wrap = items[2];
    expect(store.getSnapshot().selection).toBe(wrap.mesh);
    const triangles = live(scene, wrap.mesh);
    expect(triangles).toBeGreaterThan(1000);
    expect($("[data-wrap-report]")?.textContent).toContain(`${triangles.toLocaleString("en")} triangles`);
    expect($("[data-wrap-report]")?.textContent).toContain("vertices on the input");
    expect(scene.selfIntersectingTriangles(wrap.mesh)).toHaveLength(0);
    expect(scene.analyze(wrap.mesh).loops).toHaveLength(0);
    // One body: both boxes and the gap between them.
    expect(scene.measure(wrap.mesh).volume).toBeGreaterThan(15.5);
    render();
    await rest();
    expect($("[data-wrap-sources]")?.textContent).toContain("Wrap of 2 meshes");

    // One undo step removes the wrap and its row (#271); the sources stay;
    // redo brings it back, selected.
    store.undo();
    store.commitFrame();
    expect(app.getSnapshot().items).toEqual([a, b]);
    expect(store.getSnapshot().selection).toBeNull();
    expect(live(scene, a.mesh)).toBe(12);
    expect(live(scene, b.mesh)).toBe(12);
    render();
    await rest();
    store.redo();
    store.commitFrame();
    expect(live(scene, wrap.mesh)).toBe(triangles);
    expect(app.getSnapshot().items).toEqual([a, b, wrap]);
    expect(store.getSnapshot().selection).toBe(wrap.mesh);
    render();
    await rest();
  });

  it("blocks a grid over the limit, and follows the meshes it reads", async () => {
    vi.useFakeTimers();
    const { scene, store, app } = harness();
    const a = addBox(store, app, "A", [0, 0, 0]);
    store.select(a.mesh);
    store.commitFrame();
    const { container } = mount(store, section(store, app));
    await rest();
    const $ = <T extends Element>(s: string) => container.querySelector<T>(s);
    expect($<HTMLButtonElement>("[data-wrap-run]")?.disabled).toBe(false);

    // A voxel size of its own starts at the detail's, then is typed.
    await act(async () => {
      const select = $<HTMLSelectElement>("[data-wrap-sizing]")!;
      Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value")!.set!.call(select, "voxel");
      select.dispatchEvent(new Event("change", { bubbles: true }));
      await vi.advanceTimersByTimeAsync(400);
    });
    expect(Number($<HTMLInputElement>("[data-wrap-voxel]")?.value)).toBeCloseTo(2 / 128, 4);
    await type($<HTMLInputElement>("[data-wrap-voxel]"), "0.002");
    expect($("[data-wrap-plan]")?.textContent).toContain("over the limit");
    expect($("[data-wrap-plan]")?.className).toContain("error");
    expect($<HTMLButtonElement>("[data-wrap-run]")?.disabled).toBe(true);

    await type($<HTMLInputElement>("[data-wrap-voxel]"), "0.1");
    const before = $("[data-wrap-plan]")?.textContent;
    expect($<HTMLButtonElement>("[data-wrap-run]")?.disabled).toBe(false);
    // Scaling the mesh changes the grid once the change rests.
    await act(async () => {
      scene.setPlacement(a.mesh, {
        rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
        translation: [0, 0, 0],
        scale: [3, 1, 1],
      });
      store.commitFrame();
    });
    await rest();
    await rest();
    expect(before).toContain("Grid 29 × 29 × 29");
    expect($("[data-wrap-plan]")?.textContent).toContain("Grid 69 × 29 × 29");
  });

  it("reports a selection that encloses nothing as an error and adds nothing", async () => {
    vi.useFakeTimers();
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
    const { container } = mount(store, section(store, app));
    await rest();
    await type(container.querySelector<HTMLInputElement>("[data-wrap-detail]"), "16");
    await click(container.querySelector("[data-wrap-run]"));
    const report = container.querySelector("[data-wrap-report]");
    expect(report?.className).toContain("error");
    expect(report?.textContent).toContain("encloses no volume");
    expect(app.getSnapshot().items.length).toBe(1);
  });
});
