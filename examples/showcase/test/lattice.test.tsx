// @vitest-environment jsdom
// Lattice infill (#265): the settings, default lengths, options and lines, and
// the wiring of the Modeling panel's section on a real Wasm Scene, SceneStore
// and AppState (the planned grid, the preview that follows the settings and
// goes with the panel, the fill as one undo step, cavities, errors).
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { LatticePlan, MeshHandle, SceneLike } from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act, createElement, type ReactElement } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { init, Scene } from "../../../packages/fast-mesh-wasm/dist/web.js";
import { AppState, type Item } from "../src/app/state/app-state";
import { LatticeSection } from "../src/features/mesh-edit/ui/LatticeSection";
import { ModelingPanel } from "../src/features/mesh-edit/ui/ModelingPanel";
import {
  DEFAULT_LATTICE_SETTINGS,
  describeLattice,
  describeLatticePlan,
  describeLatticePreview,
  latticeLengths,
  latticeOptions,
  latticeProblem,
} from "../src/features/mesh-edit/ux/lattice";
import { latticePreviewOf } from "../src/features/mesh-edit/ux/lattice-preview-state";
import { countLiveTriangles } from "../src/features/select/ux/face-sets";
import { buildCommandPaletteItems } from "../src/shell/ux/command-registry";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeAll(() =>
  init(readFileSync(resolve(__dirname, "../../../packages/fast-mesh-wasm/dist/wasm/fast_mesh_wasm_bg.wasm"))),
);

const cleanups: (() => void)[] = [];

afterEach(() => {
  while (cleanups.length > 0) {
    cleanups.pop()?.();
  }
  vi.useRealTimers();
});

const PLAN: LatticePlan = {
  region: "inside",
  dims: [138, 138, 101],
  voxelSize: 0.5,
  wallThickness: 1.25,
  voxels: 1_923_444,
  triangles: 812_000,
  bytes: 61_585_220,
  maxVoxels: 16_777_216,
  fits: true,
  solidFraction: 0.449,
  regionVolume: 125_000,
  cropped: false,
};

describe("lattice settings, options and lines", () => {
  it("starts with a sixth of the part, 15 % walls and a shell as thick as them", () => {
    expect(latticeLengths(DEFAULT_LATTICE_SETTINGS, 60)).toEqual({ cell: 10, wall: 1.5, shell: 1.5 });
    expect(latticeLengths(DEFAULT_LATTICE_SETTINGS, 100)).toEqual({ cell: 17, wall: 2.5, shell: 2.5 });
    const typed = { ...DEFAULT_LATTICE_SETTINGS, cell: 8, wall: 1, shellThickness: 2 };
    expect(latticeLengths(typed, 100)).toEqual({ cell: 8, wall: 1, shell: 2 });
    expect(latticeLengths({ ...typed, shell: false }, 100).shell).toBe(0);
    // The wall follows a typed cell, and the shell a typed wall.
    expect(latticeLengths({ ...DEFAULT_LATTICE_SETTINGS, cell: 4 }, 100)).toEqual({ cell: 4, wall: 0.6, shell: 0.6 });
  });

  it("tells lengths that make no lattice", () => {
    expect(latticeProblem({ cell: 10, wall: 1.5, shell: 0 })).toBeNull();
    expect(latticeProblem({ cell: 9, wall: 3, shell: 0 })).toBeNull();
    expect(latticeProblem({ cell: 0, wall: 0, shell: 0 })).toContain("cell size");
    expect(latticeProblem({ cell: 10, wall: 0, shell: 0 })).toContain("wall thickness");
    expect(latticeProblem({ cell: 10, wall: 3.4, shell: 0 })).toContain("a third");
  });

  it("turns settings into engine options", () => {
    const lengths = { cell: 10, wall: 1.5, shell: 2 };
    expect(latticeOptions(DEFAULT_LATTICE_SETTINGS, lengths)).toEqual({
      kind: "gyroid",
      cellSize: 10,
      wallThickness: 1.5,
      shellThickness: 2,
      region: "auto",
    });
    const typed = { ...DEFAULT_LATTICE_SETTINGS, kind: "diamond", region: "cavity", voxel: 0.25 } as const;
    expect(latticeOptions(typed, { ...lengths, shell: 0 })).toEqual({
      kind: "diamond",
      cellSize: 10,
      wallThickness: 1.5,
      shellThickness: 0,
      region: "cavity",
      voxelSize: 0.25,
    });
    expect(latticeOptions({ ...typed, voxel: 0 }, lengths)).not.toHaveProperty("voxelSize");
  });

  it("describes the planned grid in the display unit, and the limit it is over", () => {
    expect(describeLatticePlan(PLAN, "mm")).toBe(
      "Grid 138 × 138 × 101: 1.9M voxels of 0.5 mm, about 812k triangles and 59 MB. " +
        "Fills the inside (125000 mm³); the walls take 45%.",
    );
    const inches = describeLatticePlan({ ...PLAN, voxelSize: 25.4, region: "cavity" }, "in");
    expect(inches).toContain("voxels of 1 in");
    expect(inches).toContain("Fills the cavity (7.6");
    const over = describeLatticePlan({ ...PLAN, voxels: 13.9e9, triangles: 2.5e9, bytes: 2e11, fits: false }, "mm");
    expect(over).toContain("13.9G voxels");
    expect(over).toContain("over the limit of 16.8M voxels");
  });

  it("says how a preview differs from the lattice", () => {
    expect(describeLatticePreview(PLAN, 12_345, 1.25, "mm")).toBe("Preview: 12k triangles on 1.9M voxels.");
    expect(describeLatticePreview({ ...PLAN, wallThickness: 2, cropped: true }, 9000, 1.25, "mm")).toBe(
      "Preview: 9,000 triangles on 1.9M voxels, walls drawn 2 mm thick, the middle of the part only.",
    );
    expect(describeLatticePreview(PLAN, 0, 1.25, "mm")).toContain("no walls");
  });

  it("reports a finished fill", () => {
    const r = { vertices: 10, triangles: 123_456, fragments: 0, solidVolume: 56_000, uvsDropped: false, plan: PLAN };
    expect(describeLattice("Bracket", "schwarzP", r, "mm")).toBe(
      "Bracket: Schwarz P lattice, 123,456 triangles; 45% of the volume is solid (56000 mm³).",
    );
    const cavity = { ...r, fragments: 1, plan: { ...PLAN, region: "cavity" as const } };
    expect(describeLattice("B", "gyroid", cavity, "mm")).toContain("45% of the cavity is solid");
    expect(describeLattice("B", "gyroid", cavity, "mm")).toContain("1 loose piece left out.");
  });

  it("is in the command palette", () => {
    const app = new AppState();
    const commands = buildCommandPaletteItems({
      app,
      items: [],
      selection: null,
      canUndo: false,
      canRedo: false,
      ortho: false,
      units: "mm",
      onSelectItem: () => {},
      onSetMode: () => {},
      onResetView: () => {},
      onUndo: () => {},
      onRedo: () => {},
      onOpenHelp: () => {},
    });
    const entry = commands.find((c) => c.id === "tool:print-lattice");
    expect(entry?.keywords).toContain("gyroid");
    entry?.run();
    expect(app.getSnapshot().activeTab).toBe("edit");
  });
});

/** A closed box of half size `h` around the origin, wound inward when `inward`. */
function makeBox(h: number, inward = false): { positions: number[]; indices: number[] } {
  const positions: number[] = [];
  for (let i = 0; i < 8; i++) {
    positions.push(i & 1 ? h : -h, i & 2 ? h : -h, i & 4 ? h : -h);
  }
  const indices = [
    0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4, 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5,
  ];
  if (inward) {
    for (let k = 0; k < indices.length; k += 3) {
      [indices[k + 1], indices[k + 2]] = [indices[k + 2], indices[k + 1]];
    }
  }
  return { positions, indices };
}

function harness(): { scene: Scene; store: SceneStore; app: AppState } {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  app.setActiveTab("edit");
  cleanups.push(() => {
    store.dispose();
    scene.free();
  });
  return { scene, store, app };
}

function add(store: SceneStore, app: AppState, name: string, part: { positions: number[]; indices: number[] }): Item {
  const item = app.addMesh(store, {
    name,
    positions: Float64Array.from(part.positions),
    indices: Uint32Array.from(part.indices),
    center: [0, 0, 0],
    radius: 2,
    volumeId: null,
    preset: null,
  });
  store.select(item.mesh);
  store.commitFrame();
  return item;
}

function mount(store: SceneStore, element: () => ReactElement) {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  let mounted = true;
  const render = () =>
    act(() => {
      root.render(createElement(SceneProvider, { store }, element()));
    });
  const unmount = () => {
    if (mounted) {
      mounted = false;
      act(() => root.unmount());
      container.remove();
    }
  };
  render();
  cleanups.push(unmount);
  return { container, render, unmount };
}

/** Lets the debounced plan and preview run and their promises land. */
async function rest(ms = 500): Promise<void> {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

async function click(el: Element | null): Promise<void> {
  expect(el).not.toBeNull();
  await act(async () => {
    el!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await vi.advanceTimersByTimeAsync(500);
  });
}

async function type(el: HTMLInputElement | null, value: string): Promise<void> {
  expect(el).not.toBeNull();
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(el, value);
    el!.dispatchEvent(new Event("input", { bubbles: true }));
    await vi.advanceTimersByTimeAsync(500);
  });
}

const live = (scene: SceneLike, mesh: MeshHandle) => countLiveTriangles(scene.views(mesh).indices());

/** The section on the store's selected mesh. */
function section(store: SceneStore, app: AppState) {
  return () => {
    const items = app.getSnapshot().items;
    const activeItem = items.find((i) => i.mesh === store.getSnapshot().selection) ?? null;
    return createElement(LatticeSection, { app, store, activeItem, units: "mm", disabled: false });
  };
}

/** Closed, manifold and free of self-intersections. */
function expectClean(scene: Scene, mesh: MeshHandle): void {
  const a = scene.analyze(mesh);
  expect(a.loops).toHaveLength(0);
  expect(a.nonManifoldEdges).toBe(0);
  expect(scene.selfIntersectingTriangles(mesh)).toHaveLength(0);
}

describe("lattice section", () => {
  it("plans, previews and fills the active mesh in one undo step", async () => {
    vi.useFakeTimers();
    const { scene, store, app } = harness();
    const item = add(store, app, "Block", makeBox(1));
    const { container } = mount(store, section(store, app));
    await rest();
    const $ = <T extends Element>(s: string) => container.querySelector<T>(s);
    // Two across: 0.33 cells and 0.05 walls by default.
    expect($<HTMLInputElement>("[data-lattice-cell]")?.value).toBe("0.33");
    expect($("[data-lattice-plan]")?.textContent).toMatch(/^Grid 105 × 105 × 105: .* Fills the inside \(8 mm³\)/);

    await type($<HTMLInputElement>("[data-lattice-cell]"), "1");
    await type($<HTMLInputElement>("[data-lattice-wall]"), "0.2");
    expect($<HTMLInputElement>("[data-lattice-shell-thickness]")?.value).toBe("0.2");
    await type($<HTMLInputElement>("[data-lattice-voxel]"), "0.1");
    expect($("[data-lattice-plan]")?.textContent).toContain("Grid 25 × 25 × 25");

    // The preview: the walls at the shell's inner face, and no change.
    await click($("[data-lattice-preview]"));
    const preview = latticePreviewOf(app);
    const first = preview.getSnapshot();
    expect(first?.mesh).toBe(item.mesh);
    expect(first?.walls.indices.length).toBeGreaterThan(1000);
    expect(first?.walls.positions.every((c) => Math.abs(c) < 0.801)).toBe(true);
    expect($("[data-lattice-preview-line]")?.textContent).toMatch(/^Preview: .* triangles on 16k voxels\.$/);
    expect(live(scene, item.mesh)).toBe(12);
    // It follows the settings.
    await click($('[data-lattice-kind="diamond"]'));
    const second = preview.getSnapshot();
    expect(second).not.toBe(first);
    expect(second?.walls.indices.length).not.toBe(first?.walls.indices.length);
    // And goes when it is turned off.
    await click($("[data-lattice-preview]"));
    expect(preview.getSnapshot()).toBeNull();
    expect($("[data-lattice-preview-line]")).toBeNull();

    await click($("[data-lattice-preview]"));
    expect(preview.getSnapshot()).not.toBeNull();
    await click($("[data-lattice-run]"));
    expect($("[data-lattice-report]")?.className).not.toContain("error");
    expect(preview.getSnapshot()).toBeNull();
    const triangles = live(scene, item.mesh) - 12;
    expect(triangles).toBeGreaterThan(1000);
    expect($("[data-lattice-report]")?.textContent).toContain(
      `Block: Diamond lattice, ${triangles.toLocaleString("en")} triangles`,
    );
    expectClean(scene, item.mesh);
    expect(app.getSnapshot().items.length).toBe(1);
    store.undo();
    store.commitFrame();
    expect(live(scene, item.mesh)).toBe(12);
    await rest();

    // Without a shell the mesh becomes the lattice.
    await click($("[data-lattice-shell]"));
    expect($("[data-lattice-shell-thickness]")).toBeNull();
    await click($("[data-lattice-run]"));
    expect($("[data-lattice-report]")?.textContent).toContain(
      `${live(scene, item.mesh).toLocaleString("en")} triangles`,
    );
    expectClean(scene, item.mesh);
    await rest();
  });

  it("finds the cavity of a hollowed mesh, and refuses one that is not there", async () => {
    vi.useFakeTimers();
    const { scene, store, app } = harness();
    const outer = makeBox(1);
    const inner = makeBox(0.75, true);
    const item = add(store, app, "Hollow", {
      positions: [...outer.positions, ...inner.positions],
      indices: [...outer.indices, ...inner.indices.map((v) => v + 8)],
    });
    const solid = add(store, app, "Solid", makeBox(1));
    store.select(item.mesh);
    const { container, render } = mount(store, section(store, app));
    await rest();
    const $ = <T extends Element>(s: string) => container.querySelector<T>(s);
    await type($<HTMLInputElement>("[data-lattice-cell]"), "0.75");
    await type($<HTMLInputElement>("[data-lattice-wall]"), "0.15");
    await type($<HTMLInputElement>("[data-lattice-voxel]"), "0.075");
    await click($("[data-lattice-shell]"));
    expect($("[data-lattice-plan]")?.textContent).toContain("Fills the cavity (3.38 mm³)");
    await click($("[data-lattice-run]"));
    expect($("[data-lattice-report]")?.textContent).toContain("of the cavity is solid");
    expectClean(scene, item.mesh);

    // A solid mesh has none.
    await act(async () => {
      const select = $<HTMLSelectElement>("[data-lattice-region]")!;
      Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value")!.set!.call(select, "cavity");
      select.dispatchEvent(new Event("change", { bubbles: true }));
    });
    store.select(solid.mesh);
    render();
    await rest();
    expect($("[data-lattice-plan]")?.textContent).toContain("no cavity");
    expect($<HTMLButtonElement>("[data-lattice-run]")?.disabled).toBe(true);
    // Its preview says so too, and shows nothing.
    await click($("[data-lattice-preview]"));
    expect($("[data-lattice-preview-line]")?.textContent).toContain("no cavity");
    expect(latticePreviewOf(app).getSnapshot()).toBeNull();
  });

  it("blocks a grid over the limit and bad lengths, and follows the mesh", async () => {
    vi.useFakeTimers();
    const { scene, store, app } = harness();
    const item = add(store, app, "Block", makeBox(1));
    const { container, unmount } = mount(store, section(store, app));
    await rest();
    const $ = <T extends Element>(s: string) => container.querySelector<T>(s);
    await type($<HTMLInputElement>("[data-lattice-voxel]"), "0.002");
    expect($("[data-lattice-plan]")?.textContent).toContain("over the limit");
    expect($("[data-lattice-plan]")?.className).toContain("error");
    expect($<HTMLButtonElement>("[data-lattice-run]")?.disabled).toBe(true);
    await click($("[data-lattice-voxel-auto]"));
    expect($<HTMLButtonElement>("[data-lattice-run]")?.disabled).toBe(false);
    await type($<HTMLInputElement>("[data-lattice-wall]"), "0.2");
    expect($("[data-lattice-plan]")?.textContent).toContain("a third of the cell size");
    expect($<HTMLButtonElement>("[data-lattice-run]")?.disabled).toBe(true);
    await click($("[data-lattice-auto]"));
    expect($("[data-lattice-plan]")?.textContent).toContain("Grid 105 × 105 × 105");

    // Scaling the mesh changes the default cell, once the change rests.
    await act(async () => {
      scene.setPlacement(item.mesh, {
        rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
        translation: [0, 0, 0],
        scale: [6, 6, 6],
      });
      store.commitFrame();
    });
    await rest();
    await rest();
    expect($<HTMLInputElement>("[data-lattice-cell]")?.value).toBe("2");
    expect($("[data-lattice-plan]")?.textContent).toContain("(1728 mm³)");

    // Another tab: nothing is planned, and the preview goes.
    await type($<HTMLInputElement>("[data-lattice-voxel]"), "0.15");
    await click($("[data-lattice-preview]"));
    expect(latticePreviewOf(app).getSnapshot()).not.toBeNull();
    await act(async () => app.setActiveTab("sculpt"));
    expect(latticePreviewOf(app).getSnapshot()).toBeNull();
    await act(async () => {
      scene.setPlacement(item.mesh, {
        rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
        translation: [0, 0, 0],
        scale: [3, 3, 3],
      });
      store.commitFrame();
    });
    await rest();
    expect($("[data-lattice-plan]")?.textContent).toContain("(1728 mm³)");
    await act(async () => app.setActiveTab("edit"));
    await rest();
    // Half as large: too coarse a voxel for the walls now.
    expect($("[data-lattice-plan]")?.textContent).toBe(
      "Lattice walls are thinner than two voxels; use a smaller voxel size or thicker walls",
    );
    await click($("[data-lattice-voxel-auto]"));
    expect($("[data-lattice-plan]")?.textContent).toContain("(216 mm³)");
    expect($("[data-lattice-preview]")?.getAttribute("aria-pressed")).toBe("false");

    // The panel goes, and the preview with it.
    await click($("[data-lattice-preview]"));
    expect(latticePreviewOf(app).getSnapshot()).not.toBeNull();
    unmount();
    expect(latticePreviewOf(app).getSnapshot()).toBeNull();
    await vi.advanceTimersByTimeAsync(500);
  });

  it("reports a mesh that is not closed as an error and leaves it", async () => {
    vi.useFakeTimers();
    const { scene, store, app } = harness();
    const open = makeBox(1);
    const item = add(store, app, "Open", { ...open, indices: open.indices.slice(3) });
    const { container } = mount(store, section(store, app));
    await rest();
    await click(container.querySelector("[data-lattice-run]"));
    const report = container.querySelector("[data-lattice-report]");
    expect(report?.className).toContain("error");
    expect(report?.textContent).toContain("closed");
    expect(live(scene, item.mesh)).toBe(11);
  });

  it("stands next to Hollow Mesh in the Modeling panel", async () => {
    vi.useFakeTimers();
    const { store, app } = harness();
    add(store, app, "Block", makeBox(1));
    const { container } = mount(store, () => createElement(ModelingPanel, { app }));
    await rest();
    const hollow = container.querySelector("[data-modeling-hollow-section]");
    expect(hollow?.nextElementSibling?.matches("[data-lattice-section]")).toBe(true);
    expect(container.querySelector("[data-lattice-plan]")?.textContent).toContain("Fills the inside");
  });
});
