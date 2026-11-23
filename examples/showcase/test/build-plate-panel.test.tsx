// @vitest-environment jsdom
// The Build Plate panel (#266) on the real engine: its fields follow the
// display unit and persist, the Home shelf and the command palette reach it,
// Arrange packs the selection or every visible mesh in one undo step and
// reports what it leaves, Multi-Clone adds copies, and the plate in the
// viewport takes no pointer rays.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

vi.mock("fast-mesh-wasm", async () => {
  return await import("../../../packages/fast-mesh-wasm/dist/web.js");
});

import { init, Scene } from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act } from "react";
import { createRoot } from "react-dom/client";
import * as THREE from "three";
import { AppState, type Item } from "../src/app/state/app-state";
import { buildPlateObject } from "../src/features/build-plate/ui/BuildPlateLayer";
import { BuildPlatePanel } from "../src/features/build-plate/ui/BuildPlatePanel";
import { DEFAULT_BUILD_PLATE } from "../src/features/build-plate/ux/plate";
import { readPlacedMesh } from "../src/features/transform/ux/placement-helpers";
import { worldBounds } from "../src/features/transform/ux/placement-math";
import { selectMesh } from "../src/features/select/ux/selection";
import { defaultSettings, parseSettings, settingsFromSnapshot } from "../src/shared/ux/settings";
import { WorkbenchRibbon } from "../src/shell/ui/Ribbon";
import { buildCommandPaletteItems } from "../src/shell/ux/command-registry";
import { tabMatchesSlot } from "../src/shell/ux/workbench-tabs";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeAll(() =>
  init(readFileSync(resolve(__dirname, "../../../packages/fast-mesh-wasm/dist/wasm/fast_mesh_wasm_bg.wasm"))),
);

const cleanups: (() => void)[] = [];
afterEach(() => {
  for (const fn of cleanups.splice(0).reverse()) {
    fn();
  }
});

/** A box of half extents `h` centred on `c`. */
function addBox(store: SceneStore, app: AppState, name: string, h: readonly number[], c: readonly number[]): Item {
  const positions: number[] = [];
  for (const z of [-1, 1]) {
    for (const [x, y] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      positions.push(c[0] + x * h[0], c[1] + y * h[1], c[2] + z * h[2]);
    }
  }
  return app.addMesh(store, {
    name,
    positions: Float64Array.from(positions),
    indices: Uint32Array.from([0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5]),
    center: [c[0], c[1], c[2]],
    radius: Math.hypot(...h),
    volumeId: null,
    preset: null,
  });
}

function setup() {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  cleanups.push(app.follow(store));
  const items = [0, 1, 2].map((i) => addBox(store, app, `Part ${i}`, [2 + i, 3, 1], [0, 0, 10 * i]));
  scene.clearHistory();
  const host = document.createElement("div");
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => {
    root.render(
      <SceneProvider store={store}>
        <WorkbenchRibbon
          app={app}
          mode="translate"
          canUndo={false}
          canRedo={false}
          wireframe={false}
          selection={null}
          hasItems
          hasVolume={false}
          onSetMode={() => {}}
          onResetView={() => {}}
          onFocusSelection={() => {}}
          onClearSelection={() => {}}
          onUndo={() => {}}
          onRedo={() => {}}
          onOpenHelp={() => {}}
        />
        <BuildPlatePanel app={app} />
      </SceneProvider>,
    );
  });
  cleanups.push(() => {
    act(() => root.unmount());
    host.remove();
    store.dispose();
    scene.free();
  });
  const select = async (item: Item, additive = false) =>
    act(async () =>
      selectMesh(app, (m) => store.select(m), store.getSnapshot().selection, item.mesh, additive),
    );
  return { scene, store, app, items, host, select };
}

const q = <T extends Element>(root: ParentNode, sel: string): T => {
  const el = root.querySelector<T>(sel);
  expect(el, sel).not.toBeNull();
  return el as T;
};

/** Sets a controlled field's value, as a user would. */
async function type(el: HTMLInputElement | HTMLSelectElement, value: string): Promise<void> {
  const proto = el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
  await act(async () => {
    Object.getOwnPropertyDescriptor(proto, "value")!.set!.call(el, value);
    el.dispatchEvent(new Event(el instanceof HTMLSelectElement ? "change" : "input", { bubbles: true }));
  });
}

async function click(el: HTMLElement): Promise<void> {
  await act(async () => el.click());
}

const bounds = (scene: Scene, item: Item) => worldBounds(readPlacedMesh(scene, item.mesh))!;

describe("Build Plate panel (#266)", () => {
  it("edits the plate in the display unit, shows it, and keeps it with the display settings", async () => {
    const { app, host } = setup();
    const field = (name: string) => q<HTMLInputElement>(host, `[data-build-plate="${name}"]`);
    expect(field("width").value).toBe("200");
    expect(field("origin-x").value).toBe("-100");
    await act(async () => app.setUnits("in"));
    await type(field("width"), "10");
    await type(field("depth"), "5");
    await type(field("origin-x"), "-1");
    await type(field("origin-y"), "2");
    await type(field("spacing"), "0.5");
    expect(host.textContent).toContain("Width X (in)");
    expect(host.textContent).toContain("Depth Y (in)");
    await type(q<HTMLSelectElement>(host, '[data-build-plate="up"]'), "y");
    expect(host.textContent).toContain("Depth Z (in)");
    await click(field("turn"));
    const plate = app.getSnapshot().buildPlate;
    expect(plate).toMatchObject({ width: 254, depth: 127, spacing: 12.7, up: "y", turn: false, keepClear: true, visible: false });
    expect(plate.origin[0]).toBeCloseTo(-25.4, 9);
    expect(plate.origin[1]).toBeCloseTo(50.8, 9);

    // The Home shelf's button and the panel's checkbox are one state.
    const shelf = q<HTMLButtonElement>(host, '[data-action="build-plate"]');
    expect(shelf.getAttribute("aria-pressed")).toBe("false");
    await click(shelf);
    expect(field("visible").checked).toBe(true);
    expect(shelf.getAttribute("aria-pressed")).toBe("true");

    const stored = parseSettings(JSON.parse(JSON.stringify(settingsFromSnapshot(app.getSnapshot()))));
    expect(stored.buildPlate).toEqual(app.getSnapshot().buildPlate);
    expect(new AppState(stored).getSnapshot().buildPlate).toEqual(stored.buildPlate);
    expect(parseSettings({ version: 1 }).buildPlate).toEqual(defaultSettings().buildPlate);
    await act(async () => app.resetDisplaySettings());
    expect(app.getSnapshot().buildPlate).toEqual(DEFAULT_BUILD_PLATE);
  });

  it("arranges the selected meshes in one undo step, showing the plate", async () => {
    const { scene, store, app, items, host, select } = setup();
    const arrange = q<HTMLButtonElement>(host, '[data-build-plate-action="arrange"]');
    expect(arrange.disabled).toBe(true);
    await select(items[0]);
    await select(items[2], true);
    expect(q(host, "[data-build-plate-scope]").textContent).toContain("Selected meshes (2)");
    expect(arrange.disabled).toBe(false);
    await act(async () => app.setBuildPlate({ width: 30, depth: 30, origin: [0, 0] }));
    const third = scene.placement(items[1].mesh);
    await click(arrange);
    await vi.waitFor(() => expect(q(host, "[data-build-plate-report]").textContent).toBe("Arranged 2 meshes"));
    expect(app.getSnapshot().buildPlate.visible).toBe(true);
    for (const i of [items[0], items[2]]) {
      const b = bounds(scene, i);
      expect(b.min[2]).toBe(0);
      expect(b.min[0]).toBeGreaterThanOrEqual(0);
      expect(b.max[0]).toBeLessThanOrEqual(30);
    }
    // Not selected: it stays, and is kept clear of.
    expect(scene.placement(items[1].mesh)).toEqual(third);
    await act(async () => store.undo());
    expect(scene.canUndo()).toBe(false);
    expect(bounds(scene, items[2]).min[2]).toBe(19);
  });

  it("arranges every visible mesh, and reports what the plate does not take", async () => {
    const { scene, store, app, items, host } = setup();
    await type(q<HTMLSelectElement>(host, "[data-build-plate-scope]"), "all");
    const option = () => q(host, '[data-build-plate-scope] option[value="all"]').textContent;
    expect(option()).toBe("All visible meshes (3)");
    await act(async () => store.setDisplay(items[0].mesh, { visible: false }));
    expect(option()).toBe("All visible meshes (2)");
    await act(async () => store.setDisplay(items[0].mesh, { visible: true }));
    // 8 x 6, 6 x 6 and 4 x 6 on 7 x 14: turned, the first and the last fit.
    await act(async () => app.setBuildPlate({ width: 7, depth: 14, origin: [0, 0], spacing: 1, keepClear: false }));
    await click(q<HTMLButtonElement>(host, '[data-build-plate-action="arrange"]'));
    await vi.waitFor(() =>
      expect(q(host, "[data-build-plate-report]").textContent).toContain("Arranged 2 of 3 meshes, 2 turned 90°"),
    );
    expect(q(host, "[data-build-plate-left]").textContent).toBe("No room left on the plate: Part 1");
    expect(bounds(scene, items[2]).max[1] - bounds(scene, items[2]).min[1]).toBeCloseTo(8, 9);

    await act(async () => app.setBuildPlate({ turn: false, width: 5 }));
    await act(async () => store.undo());
    await click(q<HTMLButtonElement>(host, '[data-build-plate-action="arrange"]'));
    await vi.waitFor(() =>
      expect(q(host, "[data-build-plate-left]").textContent).toBe("Too big for the plate: Part 1, Part 2"),
    );
    // A plate without a size says so and arranges nothing.
    await type(q<HTMLInputElement>(host, '[data-build-plate="depth"]'), "0");
    expect(host.textContent).toContain("must be above zero");
    expect(q<HTMLButtonElement>(host, '[data-build-plate-action="arrange"]').disabled).toBe(true);
  });

  it("makes copies of the selected mesh, in one undo step, and arranges them after", async () => {
    const { scene, store, app, items, host, select } = setup();
    const run = q<HTMLButtonElement>(host, '[data-multi-clone-action="run"]');
    expect(run.disabled).toBe(true);
    await select(items[1]);
    await type(q<HTMLInputElement>(host, '[data-multi-clone="copies"]'), "0");
    expect(q(host, "[data-multi-clone-problem]").textContent).toMatch(/Copies must be/);
    expect(run.disabled).toBe(true);
    await type(q<HTMLInputElement>(host, '[data-multi-clone="copies"]'), "4");
    expect(q<HTMLInputElement>(host, '[data-multi-clone="columns"]').placeholder).toBe("3");
    await type(q<HTMLSelectElement>(host, '[data-multi-clone="pattern"]'), "linear");
    await type(q<HTMLSelectElement>(host, '[data-multi-clone="axis"]'), "-x");
    await act(async () => app.setUnits("in"));
    await type(q<HTMLInputElement>(host, '[data-multi-clone="spacing"]'), "1");
    await click(run);
    await vi.waitFor(() =>
      expect(q(host, "[data-multi-clone-report]").textContent).toBe("Added 4 copies of Part 1, selected with it"),
    );
    const copies = app.getSnapshot().items.slice(3);
    expect(copies.map((c) => c.name)).toEqual(["Part 1 copy", "Part 1 copy 2", "Part 1 copy 3", "Part 1 copy 4"]);
    // 6 mm wide, an inch between.
    expect(bounds(scene, copies[3]).min[0]).toBeCloseTo(-3 - 4 * 31.4, 6);
    expect(q(host, "[data-build-plate-scope]").textContent).toContain("Selected meshes (5)");
    await click(q<HTMLButtonElement>(host, '[data-build-plate-action="arrange"]'));
    await vi.waitFor(() => expect(q(host, "[data-build-plate-report]").textContent).toBe("Arranged 5 meshes"));
    await act(async () => store.undo());
    await act(async () => store.undo());
    expect(scene.canUndo()).toBe(false);
    expect(app.getSnapshot().items).toEqual(items);
  });

  it("is on the Home tab, and in the command palette", async () => {
    const { scene, app, items, host, select } = setup();
    expect(tabMatchesSlot("home", "buildPlate")).toBe(true);
    expect(tabMatchesSlot("select", "buildPlate")).toBe(false);
    await select(items[0]);
    const ctx = {
      app,
      items: app.getSnapshot().items,
      selection: items[0].mesh as Item["mesh"] | null,
      canUndo: false,
      canRedo: false,
      ortho: false,
      units: "mm" as const,
      onSelectItem: () => {},
      onSetMode: () => {},
      onResetView: () => {},
      onUndo: () => {},
      onRedo: () => {},
      onOpenHelp: () => {},
    };
    const command = (id: string, over: Partial<typeof ctx> = {}) =>
      buildCommandPaletteItems({ ...ctx, ...over }).find((c) => c.id === id)!;
    await act(async () => command("action:build-plate-toggle").run());
    expect(app.getSnapshot().buildPlate.visible).toBe(true);
    expect(command("action:build-plate-toggle").title).toBe("Build Plate: Hide");
    await act(async () => app.setActiveTab("repair"));
    await act(async () => command("action:build-plate-arrange").run());
    expect(app.getSnapshot().activeTab).toBe("home");
    await vi.waitFor(() => expect(q(host, "[data-build-plate-report]").textContent).toBe("Arranged 1 mesh"));
    expect(bounds(scene, items[0]).min[2]).toBe(0);
    // jsdom has no layout to scroll.
    Element.prototype.scrollIntoView = () => {};
    await act(async () => command("tool:multi-clone").run());
    expect(document.activeElement?.getAttribute("data-multi-clone")).toBe("copies");
    expect(command("tool:multi-clone", { selection: null }).disabled).toBe(true);
    expect(command("action:build-plate-arrange", { items: [] }).disabled).toBe(true);
  });
});

describe("the plate in the viewport", () => {
  it("is a grid rectangle on the floor that takes no pointer rays", () => {
    const { root, dispose } = buildPlateObject({ width: 20, depth: 10, origin: [-10, -5], up: "z" });
    root.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(root);
    expect(box.min.toArray()).toEqual([-10, -5, 0]);
    expect(box.max.toArray()).toEqual([10, 5, 0]);
    const ray = new THREE.Raycaster(new THREE.Vector3(1.5, 0.5, 10), new THREE.Vector3(0, 0, -1));
    expect(ray.intersectObject(root, true)).toEqual([]);
    dispose();
  });
});
