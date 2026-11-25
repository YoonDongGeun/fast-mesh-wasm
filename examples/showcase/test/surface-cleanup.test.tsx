// @vitest-environment jsdom
// The surface cleanup section (#264): how its fields parse, the region, the
// report lines and readouts; and the Denoise and Delaunay flips buttons of
// the repair panel on a real Wasm Scene, SceneStore and AppState (one undo
// step each, the selected faces only, and the readouts an undo clears), and
// the ribbon and command palette entries that press them.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { MeshHandle, SceneLike } from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { generatePrimitive, init, Scene } from "../../../packages/fast-mesh-wasm/dist/web.js";
import { AppState, type Item } from "../src/app/state/app-state";
import { RepairPanel } from "../src/features/mesh-repair/ui/RepairPanel";
import {
  cleanupRegion,
  denoiseIterationsInput,
  denoiseStrengthInput,
  describeDelaunayFlips,
  describeDenoise,
  dihedralLimitInput,
  featureAngleInput,
  largestScale,
  SURFACE_CLEANUP_COMMANDS,
  surfaceReadouts,
} from "../src/features/mesh-repair/ux/surface-cleanup";
import { writeMeshFaceSelection } from "../src/features/select/ux/face-selection-ops";
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
});

const stats = (minAngle: number, meanNormalDeviation: number) => ({ triangles: 10, minAngle, meanNormalDeviation });

describe("surface cleanup fields and text", () => {
  it("clamps each field and reads an empty or invalid entry as the default", () => {
    expect(["", "x", "0", "3.6", "500"].map(denoiseIterationsInput)).toEqual([5, 5, 1, 4, 100]);
    expect(["", "0", "0.35", "7"].map(denoiseStrengthInput)).toEqual([0.5, 0.05, 0.35, 1]);
    expect(["", "0", "60.5", "900"].map(featureAngleInput)).toEqual([45, 1, 60.5, 180]);
    expect(["", "-3", "0", "12", "120"].map(dihedralLimitInput)).toEqual([10, 0, 0, 12, 90]);
  });

  it("limits the region to live selected faces, only when asked", () => {
    const indices = Uint32Array.from([0, 1, 2, 0, 0, 0, 2, 1, 3]);
    const selected = Uint32Array.from([0, 1, 2, 7]);
    expect(cleanupRegion(indices, selected, false)).toBeUndefined();
    expect(Array.from(cleanupRegion(indices, selected, true) ?? [])).toEqual([0, 2]);
    // Asked for, with nothing selected: nothing, not the whole mesh.
    expect(cleanupRegion(indices, undefined, true)).toHaveLength(0);
  });

  it("reports what each run did, in the display unit and world lengths", () => {
    const report = {
      moved: 1200,
      maxDisplacement: 0.5,
      iterations: 5,
      reverted: 0,
      before: stats(14.9, 9.468),
      after: stats(15.25, 0.571),
    };
    expect(describeDenoise(report, "mm")).toBe(
      "Denoise: moved 1,200 vertices (max 0.5 mm), mean normal deviation 9.47° → 0.57°, minimum angle 14.90° → 15.25°",
    );
    expect(describeDenoise({ ...report, maxDisplacement: 12.7 }, "in", 2)).toContain("(max 1 in)");
    expect(describeDenoise({ ...report, reverted: 1 }, "mm")).toContain("1 vertex left in place");
    expect(describeDenoise({ ...report, moved: 0 }, "mm")).toBe("Denoise: nothing to smooth");
    expect(largestScale({})).toBe(1);
    expect(largestScale({ scale: [1, 3, 2] })).toBe(3);

    const flips = { flips: 1, passes: 1, converged: true, before: stats(10, 2), after: stats(20, 2.5) };
    expect(describeDelaunayFlips(flips)).toBe(
      "Delaunay flips: 1 edge in 1 pass, mean normal deviation 2.00° → 2.50°, minimum angle 10.00° → 20.00°",
    );
    expect(describeDelaunayFlips({ ...flips, flips: 2000, passes: 3, converged: false })).toContain(
      "2,000 edges in 3 passes (stopped at the pass limit)",
    );
    expect(describeDelaunayFlips({ ...flips, flips: 0 })).toBe("Delaunay flips: nothing to flip");
    expect(surfaceReadouts(flips).map((r) => [r.key, r.before, r.after])).toEqual([
      ["normalDeviation", "2.00°", "2.50°"],
      ["minAngle", "10.00°", "20.00°"],
    ]);
  });
});

/** A cube of side 2 with `n` cells a side and noise of `noise` along each axis. */
function noisyCube(n: number, noise: number) {
  const { positions, indices } = generatePrimitive({ kind: "box", size: [2, 2, 2], segments: [n, n, n] });
  let state = 1;
  const next = () => {
    state = (state * 1103515245 + 12345) % 2147483648;
    return (2 * state) / 2147483648 - 1;
  };
  return { positions: Float64Array.from(positions, (p) => p + noise * next()), indices };
}

/** A flat grid of sheared cells, each split along its long diagonal. */
function shearedGrid(n: number) {
  const positions: number[] = [];
  const indices: number[] = [];
  for (let y = 0; y <= n; y++) {
    for (let x = 0; x <= n; x++) positions.push(x + 0.8 * y, y, 0);
  }
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      const a = y * (n + 1) + x;
      indices.push(a, a + 1, a + n + 2, a, a + n + 2, a + n + 1);
    }
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

interface Mounted {
  scene: Scene;
  store: SceneStore;
  app: AppState;
  item: Item;
  panel: HTMLDivElement;
}

function mountPanel(name: string, mesh: { positions: Float64Array; indices: Uint32Array }): Mounted {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  const item = app.addMesh(store, { name, ...mesh, center: [0, 0, 0], radius: 2, volumeId: null, preset: null });
  store.select(item.mesh);
  store.commitFrame();
  const panel = document.createElement("div");
  document.body.appendChild(panel);
  const root = createRoot(panel);
  act(() => {
    root.render(createElement(SceneProvider, { store }, createElement(RepairPanel, { app, scene })));
  });
  cleanups.push(() => {
    act(() => root.unmount());
    panel.remove();
    store.dispose();
    scene.free();
  });
  return { scene, store, app, item, panel };
}

/** Lets the panel's async work settle. */
const flush = () =>
  act(async () => {
    for (let i = 0; i < 8; i++) await Promise.resolve();
    await new Promise((r) => setTimeout(r, 0));
  });

async function press(panel: HTMLElement, op: string): Promise<void> {
  const button = panel.querySelector<HTMLButtonElement>(`[data-repair-op="${op}"]`);
  expect(button?.disabled).toBe(false);
  await act(async () => button!.click());
  await flush();
}

/** Types into a controlled input the way React sees it. */
function type(panel: HTMLElement, selector: string, value: string): void {
  const input = panel.querySelector<HTMLInputElement>(selector)!;
  act(() => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
}

const positionsOf = (scene: SceneLike, mesh: MeshHandle) => Array.from(scene.views(mesh).positions());
const indicesOf = (scene: SceneLike, mesh: MeshHandle) => Array.from(scene.views(mesh).indices());
const report = (panel: HTMLElement) => panel.querySelector("[data-repair-report]")?.textContent ?? "";
const readout = (panel: HTMLElement, key: string) =>
  panel.querySelector(`[data-surface-readout] [data-readout="${key}"] dd`)?.textContent ?? null;

describe("Denoise and Delaunay flips in the repair panel", () => {
  it("denoises the mesh as one undo step and shows the readouts", async () => {
    const { scene, store, item, panel } = mountPanel("Scan", noisyCube(8, 0.04));
    const start = positionsOf(scene, item.mesh);
    type(panel, '[data-denoise="iterations"]', "10");
    type(panel, '[data-denoise="strength"]', "1");
    await press(panel, "denoise");

    expect(report(panel)).toContain("Denoise: moved");
    expect(report(panel)).toContain("mm)");
    expect(panel.querySelector("[data-surface-readout]")?.getAttribute("data-surface-readout")).toBe("Denoise");
    const [before, after] = (readout(panel, "normalDeviation") ?? "").split(" → ").map(Number.parseFloat);
    expect(after).toBeLessThan(before / 1.5);
    const smooth = positionsOf(scene, item.mesh);
    expect(smooth).not.toEqual(start);

    // One undo step, which also clears the report and the readouts.
    await act(async () => {
      store.undo();
      store.commitFrame();
    });
    await flush();
    expect(positionsOf(scene, item.mesh)).toEqual(start);
    expect(scene.canUndo()).toBe(false);
    expect(report(panel)).toBe("");
    expect(readout(panel, "normalDeviation")).toBeNull();
    await act(async () => {
      store.redo();
      store.commitFrame();
    });
    expect(positionsOf(scene, item.mesh)).toEqual(smooth);
  });

  it("flips edges as one undo step, without moving a vertex", async () => {
    const { scene, store, item, panel } = mountPanel("Sheet", shearedGrid(6));
    const start = { positions: positionsOf(scene, item.mesh), indices: indicesOf(scene, item.mesh) };
    await press(panel, "delaunayFlips");

    expect(report(panel)).toContain("Delaunay flips:");
    expect(readout(panel, "minAngle")).toBe("22.29° → 49.97°");
    expect(positionsOf(scene, item.mesh)).toEqual(start.positions);
    expect(indicesOf(scene, item.mesh)).not.toEqual(start.indices);

    // Nothing left to flip: no new undo step.
    await press(panel, "delaunayFlips");
    expect(report(panel)).toContain("nothing to flip");
    await act(async () => {
      store.undo();
      store.commitFrame();
    });
    await flush();
    expect(indicesOf(scene, item.mesh)).toEqual(start.indices);
    expect(readout(panel, "minAngle")).toBeNull();
    expect(scene.canUndo()).toBe(false);
  });

  it("runs on the selected faces only, and not at all without any", async () => {
    const { scene, app, item, panel } = mountPanel("Sheet", shearedGrid(6));
    const start = indicesOf(scene, item.mesh);
    const button = (op: string) => panel.querySelector<HTMLButtonElement>(`[data-repair-op="${op}"]`)!;
    const box = panel.querySelector<HTMLInputElement>("[data-cleanup-selected]")!;
    await act(async () => box.click());
    expect([button("denoise").disabled, button("delaunayFlips").disabled]).toEqual([true, true]);

    // The first two rows of cells.
    act(() => writeMeshFaceSelection(app, item.id, Uint32Array.from({ length: 24 }, (_, t) => t)));
    expect(box.parentElement?.textContent).toContain("(24)");
    await press(panel, "delaunayFlips");
    expect(report(panel)).toContain("Delaunay flips:");
    const after = indicesOf(scene, item.mesh);
    expect(after.slice(72)).toEqual(start.slice(72));
    expect(after.slice(0, 72)).not.toEqual(start.slice(0, 72));
  });

  it("is offered on the ribbon shelf's and the palette's lists", () => {
    const { app, item } = mountPanel("Scan", noisyCube(2, 0));
    expect(SURFACE_CLEANUP_COMMANDS.map((c) => c.op)).toEqual(["denoise", "delaunayFlips"]);
    const noop = () => undefined;
    const ctx = {
      app,
      items: [item],
      selection: item.mesh,
      canUndo: false,
      canRedo: false,
      ortho: false,
      units: "mm" as const,
      onSelectItem: noop,
      onSetMode: noop,
      onResetView: noop,
      onUndo: noop,
      onRedo: noop,
      onOpenHelp: noop,
    };
    const find = (selection: MeshHandle | null) =>
      buildCommandPaletteItems({ ...ctx, selection }).filter((c) => c.id.startsWith("repair:"));
    expect(find(item.mesh).map((c) => [c.id, c.disabled])).toEqual([
      ["repair:denoise", false],
      ["repair:delaunayFlips", false],
    ]);
    expect(find(null).every((c) => c.disabled)).toBe(true);
    act(() => find(item.mesh)[1].run());
    expect(app.getSnapshot().activeTab).toBe("repair");
  });
});
