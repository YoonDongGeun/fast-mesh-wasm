// @vitest-environment jsdom
// The transform panel's placement helpers (#256) on the real engine: the
// buttons move, lay flat and bake the selected mesh, each one undo step;
// the surface picks arm, prompt, cancel and land; the ribbon and command
// palette press the same buttons.
import { existsSync, readFileSync } from "node:fs";
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
import { AppState, type Item } from "../src/app/state/app-state";
import { viewportHint } from "../src/app/ux/viewport-hint";
import { TransformPanel } from "../src/features/transform/ui/TransformPanel";
import { readPlacedMesh } from "../src/features/transform/ux/placement-helpers";
import { worldBounds } from "../src/features/transform/ux/placement-math";
import { landPlacementHit } from "../src/features/transform/ux/use-placement-picks";
import { buildCommandPaletteItems } from "../src/shell/ux/command-registry";

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

/** A box of half extents (1, 2, 3) centred on (10, 20, 30); triangles 10 and 11 are its +X face. */
function addBox(store: SceneStore, app: AppState, name: string): Item {
  const [cx, cy, cz] = [10, 20, 30];
  const positions = Float64Array.from([
    cx - 1, cy - 2, cz - 3,
    cx + 1, cy - 2, cz - 3,
    cx + 1, cy + 2, cz - 3,
    cx - 1, cy + 2, cz - 3,
    cx - 1, cy - 2, cz + 3,
    cx + 1, cy - 2, cz + 3,
    cx + 1, cy + 2, cz + 3,
    cx - 1, cy + 2, cz + 3,
  ]);
  const indices = Uint32Array.from([
    0, 2, 1, 0, 3, 2,
    4, 5, 6, 4, 6, 7,
    0, 1, 5, 0, 5, 4,
    2, 3, 7, 2, 7, 6,
    0, 4, 7, 0, 7, 3,
    1, 2, 6, 1, 6, 5,
  ]);
  return app.addMesh(store, {
    name,
    positions,
    indices,
    center: [cx, cy, cz],
    radius: Math.hypot(1, 2, 3),
    volumeId: null,
    preset: null,
  });
}

function setup() {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  const item = addBox(store, app, "Part");
  store.select(item.mesh);
  scene.clearHistory();
  const host = document.createElement("div");
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => {
    root.render(
      <SceneProvider store={store}>
        <TransformPanel app={app} mode="translate" onMode={() => {}} />
      </SceneProvider>,
    );
  });
  cleanups.push(() => {
    act(() => root.unmount());
    host.remove();
    store.dispose();
    scene.free();
  });
  return { scene, store, app, item, host };
}

async function click(el: Element | null): Promise<void> {
  expect(el).not.toBeNull();
  await act(async () => {
    (el as HTMLElement).click();
    await Promise.resolve();
    await Promise.resolve();
  });
}

function bounds(scene: Scene, item: Item) {
  const b = worldBounds(readPlacedMesh(scene, item.mesh))!;
  return { min: b.min.map((v) => v + 0), max: b.max.map((v) => v + 0) };
}

describe("placement helpers panel (#256)", () => {
  it("moves the centre and the bottom centre to the origin, one undo step each, reporting in the unit", async () => {
    const { scene, store, app, item, host } = setup();
    await click(host.querySelector('[data-placement-action="origin-center"]'));
    expect(bounds(scene, item)).toEqual({ min: [-1, -2, -3], max: [1, 2, 3] });
    expect(host.querySelector("[data-placement-report]")?.textContent).toContain("mm");
    store.undo();
    expect(scene.placement(item.mesh).translation).toEqual([0, 0, 0]);
    expect(scene.canUndo()).toBe(false);

    await click(host.querySelector('[data-placement-action="origin-bottom"]'));
    expect(bounds(scene, item)).toEqual({ min: [-1, -2, 0], max: [1, 2, 6] });

    // With Y up the bottom is the lowest Y.
    const upSelect = host.querySelector<HTMLSelectElement>("[data-placement-up]")!;
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value")?.set;
      setter?.call(upSelect, "y");
      upSelect.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await click(host.querySelector('[data-placement-action="origin-bottom"]'));
    expect(bounds(scene, item)).toEqual({ min: [-1, 0, -3], max: [1, 4, 3] });
    expect(app.getSnapshot().placementTool.report?.error).toBe(false);
  });

  it("arms lay flat (disarming an inspect tool), prompts, cancels, and lands a face click", async () => {
    const { scene, store, app, item, host } = setup();
    app.setInspectTool("ruler");
    const lay = host.querySelector('[data-placement-action="lay-flat"]');
    await click(lay);
    expect(app.getSnapshot().inspectTool).toBeNull();
    expect(app.getSnapshot().placementTool.pick?.kind).toBe("layFlat");
    expect(lay?.getAttribute("aria-pressed")).toBe("true");
    expect(host.querySelector("[data-placement-prompt]")?.textContent).toContain("Click the face");
    expect(
      viewportHint({
        sculpting: false,
        maskPaint: false,
        selectTool: null,
        deformTool: null,
        printTool: null,
        picking: false,
        placementPick: true,
        selected: true,
        coarse: false,
      }),
    ).toContain("Esc cancels");

    await click(host.querySelector("[data-placement-cancel]"));
    expect(app.getSnapshot().placementTool.pick).toBeNull();

    await click(lay);
    await act(async () => {
      landPlacementHit(app, store, { mesh: item.mesh, triangle: 10, point: [11, 20, 30] });
    });
    expect(app.getSnapshot().placementTool.pick).toBeNull();
    // The +X face now faces down on Z = 0; the box turned about its centre.
    expect(bounds(scene, item)).toEqual({ min: [7, 18, 0], max: [13, 22, 2] });
    expect(host.querySelector("[data-placement-report]")?.textContent).toBe("Laid flat on Z = 0");
    store.undo();
    expect(scene.canUndo()).toBe(false);
  });

  it("stands a pick down when another tool arms or the selection moves", async () => {
    const { store, app, host } = setup();
    await click(host.querySelector('[data-placement-action="datum-321"]'));
    expect(app.getSnapshot().placementTool.pick?.kind).toBe("datum321");
    await act(async () => app.setInspectTool("query"));
    expect(app.getSnapshot().placementTool.pick).toBeNull();

    await click(host.querySelector('[data-placement-action="align-points"]'));
    expect(app.getSnapshot().placementTool.pick?.kind).toBe("alignPoints");
    await act(async () => store.select(null));
    expect(app.getSnapshot().placementTool.pick).toBeNull();
  });

  it("applies the transform as one undo step and reads 1x scale after", async () => {
    const { scene, store, item, host } = setup();
    store.setPlacement(item.mesh, { rotation: [0, -1, 0, 1, 0, 0, 0, 0, 1], translation: [5, -7, 2], scale: [2, 1, 1] });
    scene.clearHistory();
    const before = bounds(scene, item);
    const positions = Array.from(scene.views(item.mesh).positions());
    await click(host.querySelector('[data-placement-action="apply-transform"]'));
    await act(async () => {
      await new Promise((r) => setTimeout(r, 0));
    });
    expect(scene.placement(item.mesh)).toEqual({ rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0, 0, 0], scale: [1, 1, 1] });
    expect(bounds(scene, item)).toEqual(before);
    expect(host.querySelector<HTMLInputElement>('[data-transform-scale="x"]')?.value).toBe("1");
    expect(host.querySelector("[data-placement-report]")?.textContent).toContain("identity");
    store.undo();
    expect(scene.placement(item.mesh).translation).toEqual([5, -7, 2]);
    expect(Array.from(scene.views(item.mesh).positions())).toEqual(positions);
    expect(scene.canUndo()).toBe(false);
  });

  it("disables the helpers on a locked mesh", async () => {
    const { app, item, host } = setup();
    await act(async () => app.setMeshLocked(item.mesh, true));
    expect(host.querySelector<HTMLButtonElement>('[data-placement-action="origin-center"]')?.disabled).toBe(true);
    expect(host.querySelector<HTMLButtonElement>('[data-placement-action="lay-flat"]')?.disabled).toBe(true);
  });

  it("offers the helpers in the command palette, pressing the panel's buttons", async () => {
    const { scene, app, item } = setup();
    const ctx = {
      app,
      items: app.getSnapshot().items,
      selection: item.mesh,
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
    const cmds = buildCommandPaletteItems(ctx).filter((c) => c.id.startsWith("placement:"));
    expect(cmds.map((c) => c.id)).toEqual([
      "placement:origin-center",
      "placement:origin-bottom",
      "placement:lay-flat",
      "placement:align-points",
      "placement:datum-321",
      "placement:apply-transform",
    ]);
    await act(async () => cmds[0].run());
    expect(app.getSnapshot().activeTab).toBe("select");
    expect(bounds(scene, item)).toEqual({ min: [-1, -2, -3], max: [1, 2, 3] });
    expect(buildCommandPaletteItems({ ...ctx, selection: null }).find((c) => c.id === "placement:lay-flat")?.disabled).toBe(true);
  });
});
