// @vitest-environment jsdom
// The align panel's landmark pairs, coarse + ICP and symmetry plane (#253) on
// the real engine: picks land on the moving and fixed meshes in turn and an
// alignment brings a moved copy back; coarse + ICP reports both stages; the
// plane is detected and mirrored across. Each alignment, and the mirror, is
// one undo step. The command palette presses the same buttons.
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

vi.mock("fast-mesh-wasm", async () => {
  return await import("../../../packages/fast-mesh-wasm/dist/web.js");
});

import { init, type Placement, type PlacementInput, Scene } from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { AppState, type Item } from "../src/app/state/app-state";
import { viewportHint } from "../src/app/ux/viewport-hint";
import { mirrorMesh } from "../src/engine/workbench-ops";
import { AlignPanel } from "../src/features/transform/ui/AlignPanel";
import { planeToWorld } from "../src/features/transform/ux/align-tools";
import { readPlacedMesh } from "../src/features/transform/ux/placement-helpers";
import { forEachWorldVertex, worldPoint } from "../src/features/transform/ux/placement-math";
import { landLandmarkHit } from "../src/features/transform/ux/use-landmark-picks";
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

/** A closed wedge with no symmetry (distinct widths, a sloped top), offset by `at`. */
function addWedge(store: SceneStore, app: AppState, name: string, at: [number, number, number] = [0, 0, 0]): Item {
  const corners = [
    [0, 0, 0],
    [3, 0, 0],
    [3, 2, 0],
    [0, 2, 0],
    [0, 0, 1],
    [3, 0, 1.8],
    [2.4, 2, 1.1],
    [0, 2, 1.4],
  ];
  const positions = Float64Array.from(corners.flatMap((c) => [c[0] + at[0], c[1] + at[1], c[2] + at[2]]));
  const indices = Uint32Array.from([
    0, 2, 1, 0, 3, 2,
    4, 5, 6, 4, 6, 7,
    0, 1, 5, 0, 5, 4,
    1, 2, 6, 1, 6, 5,
    2, 3, 7, 2, 7, 6,
    3, 0, 4, 3, 4, 7,
  ]);
  return app.addMesh(store, {
    name,
    positions,
    indices,
    center: [1.5 + at[0], 1 + at[1], 0.9 + at[2]],
    radius: 2,
    volumeId: null,
    preset: null,
  });
}

/** A box of half extents (2, 1, 0.5) centred on (10, 20, 30). */
function addBox(store: SceneStore, app: AppState, name: string): Item {
  const [hx, hy, hz] = [2, 1, 0.5];
  const [cx, cy, cz] = [10, 20, 30];
  const positions = Float64Array.from([
    cx - hx, cy - hy, cz - hz,
    cx + hx, cy - hy, cz - hz,
    cx + hx, cy + hy, cz - hz,
    cx - hx, cy + hy, cz - hz,
    cx - hx, cy - hy, cz + hz,
    cx + hx, cy - hy, cz + hz,
    cx + hx, cy + hy, cz + hz,
    cx - hx, cy + hy, cz + hz,
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
    radius: Math.hypot(hx, hy, hz),
    volumeId: null,
    preset: null,
  });
}

const rotZ = (a: number): number[] => [Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a), 0, 0, 0, 1];

/** A fixed wedge and a moved copy of it (the selection), with no history. */
function setup(moved: PlacementInput) {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  const fixed = addWedge(store, app, "Fixed");
  const moving = addWedge(store, app, "Moving");
  store.setPlacement(moving.mesh, moved);
  store.select(moving.mesh);
  scene.clearHistory();
  const host = mount(store, app);
  cleanups.push(() => {
    store.dispose();
    scene.free();
  });
  return { scene, store, app, fixed, moving, host };
}

function mount(store: SceneStore, app: AppState): HTMLDivElement {
  const host = document.createElement("div");
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => {
    root.render(
      <SceneProvider store={store}>
        <AlignPanel app={app} />
      </SceneProvider>,
    );
  });
  cleanups.push(() => {
    act(() => root.unmount());
    host.remove();
  });
  return host;
}

async function click(el: Element | null): Promise<void> {
  expect(el).not.toBeNull();
  expect((el as HTMLButtonElement).disabled).toBe(false);
  await act(async () => {
    (el as HTMLElement).click();
    for (let i = 0; i < 4; i++) await Promise.resolve();
  });
}

/** Every live vertex's world position, in slot order. */
function worldVertices(scene: Scene, item: Item): number[][] {
  const out: number[][] = [];
  forEachWorldVertex(readPlacedMesh(scene, item.mesh), (_s, x, y, z) => out.push([x, y, z]));
  return out;
}

function expectClose(a: readonly (readonly number[])[], b: readonly (readonly number[])[], tol: number): void {
  expect(a.length).toBe(b.length);
  a.forEach((p, i) => p.forEach((v, k) => expect(Math.abs(v - b[i][k])).toBeLessThan(tol)));
}

const MOVED = { rotation: rotZ(0.75), translation: [4.5, -2, 1.25] as [number, number, number] };

describe("landmark alignment (#253)", () => {
  it("picks pairs on the moving then the fixed mesh, aligns the copy back and undoes in one step", async () => {
    const { scene, store, app, fixed, moving, host } = setup(MOVED);
    const before: Placement = scene.placement(moving.mesh);

    await click(host.querySelector("[data-align-pick]"));
    expect(app.getSnapshot().alignTool.landmarks?.picking).toBe(true);
    expect(host.querySelector("[data-align-landmark-prompt]")?.textContent).toContain("moving mesh (Moving)");
    expect(
      viewportHint({
        sculpting: false,
        maskPaint: false,
        selectTool: null,
        deformTool: null,
        printTool: null,
        picking: false,
        landmarkPick: true,
        selected: true,
        coarse: false,
      }),
    ).toContain("Esc stops");

    // A hit on the fixed mesh first is not the point asked for.
    const third = [1 / 3, 1 / 3, 1 / 3] as [number, number, number];
    act(() => landLandmarkHit(app, store, { mesh: fixed.mesh, triangle: 0, barycentric: third }));
    expect(app.getSnapshot().alignTool.landmarks?.fixedPoints).toHaveLength(0);

    // The same spots on both: triangle centroids of the bottom, top, right and back faces.
    for (const triangle of [0, 2, 6, 8]) {
      act(() => landLandmarkHit(app, store, { mesh: moving.mesh, triangle, barycentric: third }));
      act(() => landLandmarkHit(app, store, { mesh: fixed.mesh, triangle, barycentric: third }));
    }
    act(() => landLandmarkHit(app, store, { mesh: moving.mesh, triangle: 4, barycentric: third }));
    expect(host.querySelector("[data-align-landmark-count]")?.textContent).toBe("4 pairs + 1 point");
    await click(host.querySelector('[data-align-action="landmarks-undo"]'));
    expect(host.querySelector("[data-align-landmark-count]")?.textContent).toBe("4 pairs");

    await click(host.querySelector('[data-align-action="landmarks-run"]'));
    expect(host.querySelector("[data-align-landmark-error]")).toBeNull();
    expect(app.getSnapshot().alignTool.landmarks?.picking).toBe(false);
    expectClose(worldVertices(scene, moving), worldVertices(scene, fixed), 1e-6);
    expect(host.querySelector("[data-align-landmark-rmse]")?.textContent).toMatch(/ mm$/);
    const residuals = [...host.querySelectorAll("[data-align-landmark-residuals] li")];
    expect(residuals).toHaveLength(4);
    for (const li of residuals) {
      expect(li.textContent).toMatch(/ mm$/);
      expect(Number.parseFloat(li.textContent ?? "")).toBeLessThan(1e-9);
    }

    store.undo();
    expect(scene.placement(moving.mesh)).toEqual(before);
    expect(scene.canUndo()).toBe(false);
  });

  it("stops picking when another tool arms, keeps the points, and clears them with other meshes", async () => {
    const { store, app, fixed, moving, host } = setup(MOVED);
    app.setPlacementTool({ pick: { kind: "layFlat", mesh: moving.mesh, points: [], normal: null, up: "z", target: "+z" } });
    await click(host.querySelector("[data-align-pick]"));
    expect(app.getSnapshot().placementTool.pick).toBeNull();
    act(() => landLandmarkHit(app, store, { mesh: moving.mesh, triangle: 0, barycentric: [1, 0, 0] }));

    await act(async () => app.setInspectTool("ruler"));
    const set = app.getSnapshot().alignTool.landmarks;
    expect(set?.picking).toBe(false);
    expect(set?.movingPoints).toHaveLength(1);
    // Not picking: taps do nothing.
    act(() => landLandmarkHit(app, store, { mesh: fixed.mesh, triangle: 0, barycentric: [1, 0, 0] }));
    expect(app.getSnapshot().alignTool.landmarks?.fixedPoints).toHaveLength(0);

    // Resuming keeps the points; the selection moving away does not drop them.
    await click(host.querySelector("[data-align-pick]"));
    expect(app.getSnapshot().alignTool.landmarks?.movingPoints).toHaveLength(1);
    await act(async () => store.select(fixed.mesh));
    expect(app.getSnapshot().alignTool.landmarks?.movingPoints).toHaveLength(1);

    // Swapping the moving mesh drops them.
    const movingSelect = host.querySelector<HTMLSelectElement>("[data-align-moving]")!;
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value")?.set?.call(movingSelect, "0");
      movingSelect.dispatchEvent(new Event("change", { bubbles: true }));
    });
    expect(app.getSnapshot().alignTool.landmarks).toBeNull();
  });
});

describe("coarse + ICP (#253)", () => {
  it("pre-aligns a turned copy, refines it, reports both stages and undoes in one step", async () => {
    const { scene, store, fixed, moving, host } = setup({ rotation: rotZ(Math.PI / 2), translation: [12, -7, 4] });
    const before = scene.placement(moving.mesh);

    await click(host.querySelector('[data-align-action="multistage-run"]'));
    expect(host.querySelector("[data-align-multistage-error]")).toBeNull();
    const stages = [...host.querySelectorAll("[data-align-stage]")];
    expect(stages.map((s) => s.getAttribute("data-align-stage"))).toEqual(["coarse", "icp"]);
    expect(stages[0].textContent).toContain("Coarse (oriented bounding box)");
    expect(stages[1].textContent).toContain("ICP");
    expect(stages[1].textContent).toMatch(/\d+ iterations/);
    expectClose(worldVertices(scene, moving), worldVertices(scene, fixed), 1e-3);

    store.undo();
    expect(scene.placement(moving.mesh)).toEqual(before);
    expect(scene.canUndo()).toBe(false);
  });
});

describe("symmetry plane (#253)", () => {
  it("detects a box's plane, reports its fit, mirrors across it in one undo step and hides it", async () => {
    const scene = new Scene();
    const store = createSceneStore(scene);
    const app = new AppState();
    cleanups.push(() => {
      store.dispose();
      scene.free();
    });
    const box = addBox(store, app, "Box");
    store.setPlacement(box.mesh, { rotation: rotZ(Math.PI / 4), translation: [5, -3, 2] });
    scene.clearHistory();
    const host = mount(store, app);
    // One mesh is enough for the plane, not for the pairs.
    expect(host.querySelector("[data-align-pick]")).toBeNull();

    await click(host.querySelector('[data-align-action="symmetry-detect"]'));
    expect(host.querySelector("[data-align-symmetry-error]")).toBeNull();
    const shown = app.getSnapshot().alignTool.symmetry!;
    expect(shown.mesh).toBe(box.mesh);
    expect(shown.score).toBeGreaterThan(0.99);
    // In the box's own frame the plane is one of its three mid planes.
    expect(Math.max(...shown.normal.map(Math.abs))).toBeCloseTo(1, 4);
    expect(host.querySelector("[data-align-symmetry-error-rms]")?.textContent).toMatch(/ mm$/);
    expect(host.querySelector("[data-align-symmetry-normal]")?.textContent).toMatch(/^\(-?\d\.\d{3}, -?\d\.\d{3}, -?\d\.\d{3}\)$/);

    const before = worldVertices(scene, box);
    await click(host.querySelector('[data-align-action="symmetry-mirror"]'));
    expect(host.querySelector("[data-align-symmetry-mirrored]")).not.toBeNull();
    // A box mirrored across its mid plane covers itself: the same corners, in another order.
    const sort = (ps: number[][]) => ps.map((p) => p.map((v) => Math.round(v * 1e6) / 1e6)).sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
    expect(sort(worldVertices(scene, box))).toEqual(sort(before));
    store.undo();
    expectClose(worldVertices(scene, box), before, 1e-9);
    expect(scene.canUndo()).toBe(false);

    await click(host.querySelector('[data-align-action="symmetry-clear"]'));
    expect(app.getSnapshot().alignTool.symmetry).toBeNull();
  });

  it("mirrors a mesh across a world plane, keeping it closed and outward", async () => {
    const scene = new Scene();
    const store = createSceneStore(scene);
    const app = new AppState();
    cleanups.push(() => {
      store.dispose();
      scene.free();
    });
    const wedge = addWedge(store, app, "Wedge", [2, 0, 0]);
    store.setPlacement(wedge.mesh, { rotation: rotZ(0.3), translation: [1, 2, 3] });
    const volume = () => scene.inspect.measure(wedge.mesh).volume ?? Number.NaN;
    const v0 = volume();
    const before = worldVertices(scene, wedge);
    // The plane X = 1 in the world.
    await mirrorMesh(store, wedge.mesh, { point: [1, 0, 0], normal: [1, 0, 0] });
    expectClose(
      worldVertices(scene, wedge),
      before.map(([x, y, z]) => [2 - x, y, z]),
      1e-6,
    );
    expect(volume()).toBeCloseTo(v0, 6);
    expect(v0).toBeGreaterThan(0);
  });

  it("carries the plane with its mesh", () => {
    // On a mesh turned a quarter about Z and moved to (5, 0, 0), its local plane X = 0 is the world plane through (5, 0, 0) facing +Y.
    const matrix = Float64Array.from([0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 0, 5, 0, 0, 1]);
    const world = planeToWorld(matrix, [0, 0, 0], [1, 0, 0]);
    expect(world.point).toEqual(worldPoint(matrix, 0, 0, 0));
    expect(world.normal.map((v) => Math.round(v * 1e9) / 1e9 + 0)).toEqual([0, 1, 0]);
  });
});

describe("align commands (#253)", () => {
  it("offers the tools in the command palette, pressing the panel's buttons", async () => {
    const { app, moving, host } = setup(MOVED);
    void host;
    const ctx = {
      app,
      items: app.getSnapshot().items,
      selection: moving.mesh,
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
    const cmds = buildCommandPaletteItems(ctx).filter((c) => c.id.startsWith("align:"));
    expect(cmds.map((c) => c.id)).toEqual(["align:landmarks-pick", "align:multistage-run", "align:symmetry-detect"]);
    await act(async () => cmds[0].run());
    expect(app.getSnapshot().activeTab).toBe("align");
    expect(app.getSnapshot().alignTool.landmarks?.picking).toBe(true);
    const one = buildCommandPaletteItems({ ...ctx, items: ctx.items.slice(0, 1) });
    expect(one.find((c) => c.id === "align:landmarks-pick")?.disabled).toBe(true);
    expect(one.find((c) => c.id === "align:symmetry-detect")?.disabled).toBe(false);
  });
});
