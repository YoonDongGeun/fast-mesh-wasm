// @vitest-environment jsdom
// Extrude Faces and Extrude Base (#262): the options their fields become and
// the report lines; and the Mesh Edit section wired to a real Wasm Scene,
// SceneStore and AppState: the face selection it extrudes, the unit setting,
// one undo step each, crossing faces, errors, and the Worker scene's path.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { MeshHandle } from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { init, Scene } from "../../../packages/fast-mesh-wasm/dist/web.js";
import { AppState, type Item } from "../src/app/state/app-state";
import { runExtrudeBase, runExtrudeFaces } from "../src/engine/workbench-ops";
import { ModelingPanel } from "../src/features/mesh-edit/ui/ModelingPanel";
import {
  BASE_DIRECTIONS,
  crossingNote,
  describeExtrudeBase,
  describeExtrudeFaces,
  extrudeBaseOptions,
  extrudeFacesBlocked,
  extrudeFacesOptions,
  signedAxisVector,
} from "../src/features/mesh-edit/ux/extrude";
import { writeMeshFaceSelection } from "../src/features/select/ux/face-selection-ops";
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

describe("extrude options and reports", () => {
  it("turns the direction fields into engine options", () => {
    expect(BASE_DIRECTIONS.map(signedAxisVector)).toEqual([
      [-1, 0, 0],
      [1, 0, 0],
      [0, -1, 0],
      [0, 1, 0],
      [0, 0, -1],
      [0, 0, 1],
    ]);
    expect(extrudeFacesOptions({ distance: 2, direction: "region" })).toEqual({ distance: 2, direction: "region" });
    expect(extrudeFacesOptions({ distance: -2, direction: "vertex" })).toEqual({ distance: -2, direction: "vertex" });
    expect(extrudeFacesOptions({ distance: -2, direction: "y" })).toEqual({ distance: -2, direction: [0, 1, 0] });
    expect(extrudeFacesBlocked(0, 1)).toBe(true);
    expect(extrudeFacesBlocked(3, 0)).toBe(true);
    expect(extrudeFacesBlocked(3, Number.NaN)).toBe(true);
    expect(extrudeFacesBlocked(3, -1)).toBe(false);
  });

  it("gives the base as an offset, or a coordinate along the axis as a height", () => {
    const fields = { plane: "at" as const, offset: 2, at: 5, allLoops: false, refine: true };
    // A height is measured along the opposite of the direction.
    expect(extrudeBaseOptions({ ...fields, direction: "-z" })).toEqual({
      direction: [0, 0, -1],
      height: 5,
      loops: "largest",
      refine: true,
    });
    expect(extrudeBaseOptions({ ...fields, direction: "+y" }).height).toBe(-5);
    expect(Object.is(extrudeBaseOptions({ ...fields, at: 0, direction: "+y" }).height, 0)).toBe(true);
    expect(extrudeBaseOptions({ ...fields, plane: "below", allLoops: true, refine: false, direction: "-x" })).toEqual({
      direction: [-1, 0, 0],
      offset: 2,
      loops: "all",
      refine: false,
    });
  });

  it("reports counts, lengths in the display unit, and crossing faces", () => {
    const faces = {
      regions: 1,
      capTriangles: 2,
      movedVertices: 0,
      addedVertices: 4,
      wallTriangles: new Uint32Array(8),
      selfIntersections: new Uint32Array(0),
    };
    expect(describeExtrudeFaces(faces, 25.4, "mm")).toBe("2 faces in 1 region by 25.4 mm, 8 wall triangles.");
    expect(describeExtrudeFaces({ ...faces, regions: 3, selfIntersections: new Uint32Array(5) }, 25.4, "in")).toBe(
      "2 faces in 3 regions by 1 in, 8 wall triangles. Warning: 5 faces touch or cross each other.",
    );
    expect(crossingNote(null)).toBe("");
    const base = {
      loops: [{ minVertex: 0, len: 4, hash: 1n }],
      height: -50.8,
      baseVertices: 4,
      capVertices: 0,
      wallTriangles: 8,
      capTriangles: 2,
      capped: false,
      flipped: false,
      selfIntersections: null,
    };
    expect(describeExtrudeBase(base, "-z", "mm")).toBe("1 loop to Z = -50.8 mm, 8 walls and 2 cap triangles.");
    expect(describeExtrudeBase({ ...base, flipped: true }, "+y", "in")).toBe(
      "1 loop to Y = 2 in, 8 walls and 2 cap triangles. The surface was turned to face outward.",
    );
  });
});

describe("the scene an extrusion runs on", () => {
  it("is the Worker's when the scene runs there, with one committed frame after it", async () => {
    const mesh = { __mesh: 1 } as unknown as MeshHandle;
    const faces = vi.fn(async () => ({ regions: 1 }));
    const base = vi.fn(async () => ({ height: 0 }));
    const commitFrame = vi.fn();
    const scene = { status: () => "idle", extrudeFaces: faces, extrudeBase: base };
    const store = { scene, commitFrame } as unknown as SceneStore;
    const triangles = Uint32Array.of(1, 2);
    expect(await runExtrudeFaces({ store, mesh, triangles, options: { distance: 1 } })).toEqual({ regions: 1 });
    expect(faces).toHaveBeenCalledWith(mesh, triangles, { distance: 1 });
    expect(commitFrame).toHaveBeenCalledTimes(1);
    expect(await runExtrudeBase({ store, mesh, options: { offset: 1 } })).toEqual({ height: 0 });
    expect(base).toHaveBeenCalledWith(mesh, { offset: 1 });
    expect(commitFrame).toHaveBeenCalledTimes(2);
  });
});

// Triangles 10 and 11 are the +z face of a box.
const BOX_TRIS = [
  0, 4, 6, 0, 6, 2, 1, 3, 7, 1, 7, 5, 0, 1, 5, 0, 5, 4, 2, 6, 7, 2, 7, 3, 0, 2, 3, 0, 3, 1, 4, 5, 7, 4, 7, 6,
];

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

function add(store: SceneStore, app: AppState, name: string, positions: number[], indices: number[]): Item {
  const item = app.addMesh(store, {
    name,
    positions: Float64Array.from(positions),
    indices: Uint32Array.from(indices),
    center: [0, 0, 0],
    radius: 5,
    volumeId: null,
    preset: null,
  });
  store.select(item.mesh);
  store.commitFrame();
  return item;
}

/** A 2 x 3 x 4 box with a corner at the origin. */
function addBox(store: SceneStore, app: AppState): Item {
  const p: number[] = [];
  for (let i = 0; i < 8; i++) {
    p.push(i & 1 ? 2 : 0, i & 2 ? 3 : 0, i & 4 ? 4 : 0);
  }
  return add(store, app, "Box", p, BOX_TRIS);
}

/** An open 4 x 4 sheet in the plane z = 1, facing +z. */
function addSheet(store: SceneStore, app: AppState): Item {
  const p: number[] = [];
  const idx: number[] = [];
  for (let y = 0; y <= 4; y++) {
    for (let x = 0; x <= 4; x++) p.push(x, y, 1);
  }
  for (let y = 0; y < 4; y++) {
    for (let x = 0; x < 4; x++) {
      const a = y * 5 + x;
      idx.push(a, a + 1, a + 6, a, a + 6, a + 5);
    }
  }
  return add(store, app, "Sheet", p, idx);
}

function mount(store: SceneStore, app: AppState): HTMLDivElement {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(createElement(SceneProvider, { store }, createElement(ModelingPanel, { app })));
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
    for (let i = 0; i < 6; i++) await Promise.resolve();
  });
}

async function setValue(el: HTMLInputElement | HTMLSelectElement | null, value: string): Promise<void> {
  expect(el).not.toBeNull();
  const proto = el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(proto, "value")?.set?.call(el, value);
  await act(async () => {
    el!.dispatchEvent(new Event(el instanceof HTMLSelectElement ? "change" : "input", { bubbles: true }));
  });
}

const reportOf = (panel: Element) => panel.querySelector("[data-extrude-report]");

describe("the Extrude section", () => {
  it("extrudes the active mesh's selected faces, which stay selected, as one undo step", async () => {
    const { scene, store, app } = harness();
    const box = addBox(store, app);
    const panel = mount(store, app);
    const run = panel.querySelector<HTMLButtonElement>("[data-extrude-faces-run]");
    expect(run?.disabled).toBe(true);
    expect(panel.querySelector("[data-extrude-faces-selection]")?.textContent).toContain("No faces selected on Box");

    // A dead slot in the selection does not count.
    act(() => writeMeshFaceSelection(app, box.id, Uint32Array.of(10, 11, 500)));
    expect(panel.querySelector("[data-extrude-faces-selection]")?.textContent).toBe("2 selected faces of Box.");
    expect(run?.disabled).toBe(false);
    await setValue(panel.querySelector("[data-extrude-faces-distance]"), "0.5");
    await click(run);
    expect(scene.measure(box.mesh).volume).toBeCloseTo(27, 5);
    expect(reportOf(panel)?.textContent).toBe(
      "Extrude faces: 2 faces in 1 region by 0.5 mm, 8 wall triangles.",
    );

    // Still selected, so it extrudes again: in inches now, along Z.
    await act(async () => app.setUnits("in"));
    await click(panel.querySelector('[data-extrude-faces-direction="z"]'));
    await setValue(panel.querySelector("[data-extrude-faces-distance]"), "1");
    await click(run);
    expect(scene.measure(box.mesh).volume).toBeCloseTo(27 + 6 * 25.4, 3);
    expect(reportOf(panel)?.textContent).toContain("by 1 in");

    store.undo();
    store.commitFrame();
    expect(scene.measure(box.mesh).volume).toBeCloseTo(27, 5);
    store.undo();
    store.commitFrame();
    expect(countLiveTriangles(scene.views(box.mesh).indices())).toBe(12);
    expect(scene.measure(box.mesh).volume).toBeCloseTo(24, 5);
    store.redo();
    store.commitFrame();
    expect(scene.measure(box.mesh).volume).toBeCloseTo(27, 5);
  });

  it("offers the crossing faces of a result as the new selection", async () => {
    const { scene, store, app } = harness();
    const box = addBox(store, app);
    const panel = mount(store, app);
    act(() => writeMeshFaceSelection(app, box.id, Uint32Array.of(10, 11)));
    await setValue(panel.querySelector("[data-extrude-faces-distance]"), "-6");
    await click(panel.querySelector("[data-extrude-faces-run]"));
    const report = reportOf(panel);
    expect(report?.className).not.toContain("error");
    expect(report?.textContent).toContain("Warning:");
    // Written all the same.
    expect(countLiveTriangles(scene.views(box.mesh).indices())).toBe(20);
    const select = panel.querySelector("[data-extrude-select-crossing]");
    await click(select);
    const selected = app.getSnapshot().wave6.select.selectedByMesh[box.id];
    expect(select?.textContent).toBe(`Select ${selected.length} crossing faces`);
    expect(selected.length).toBeGreaterThan(2);
    expect(panel.querySelector("[data-extrude-select-crossing]")).toBeNull();
  });

  it("extrudes an open mesh's base to a closed solid, as one undo step", async () => {
    const { scene, store, app } = harness();
    const sheet = addSheet(store, app);
    const panel = mount(store, app);
    const run = panel.querySelector("[data-extrude-base-run]");
    // 2 mm past the sheet, downward: a 4 x 4 x 2 block.
    await click(run);
    expect(scene.measure(sheet.mesh).volume).toBeCloseTo(32, 4);
    expect(reportOf(panel)?.textContent).toMatch(/^Extrude base: 1 loop to Z = -1 mm, 32 walls and \d+ cap triangles\.$/);
    store.undo();
    store.commitFrame();
    expect(scene.measure(sheet.mesh).volume).toBeNull();
    expect(countLiveTriangles(scene.views(sheet.mesh).indices())).toBe(32);

    // Upward to a coordinate, with the rim's vertices only.
    await setValue(panel.querySelector("[data-extrude-base-direction]"), "+z");
    await setValue(panel.querySelector("[data-extrude-base-plane]"), "at");
    await setValue(panel.querySelector("[data-extrude-base-at]"), "4");
    await click(panel.querySelector("[data-extrude-base-refine]"));
    await click(run);
    expect(scene.measure(sheet.mesh).volume).toBeCloseTo(48, 4);
    expect(reportOf(panel)?.textContent).toBe(
      "Extrude base: 1 loop to Z = 4 mm, 32 walls and 14 cap triangles. The surface was turned to face outward.",
    );
  });

  it("reports what the engine refuses as an error and changes nothing", async () => {
    const { scene, store, app } = harness();
    const box = addBox(store, app);
    const panel = mount(store, app);
    await click(panel.querySelector("[data-extrude-base-run]"));
    expect(reportOf(panel)?.className).toContain("error");
    expect(reportOf(panel)?.textContent).toContain("no boundary loop");
    // Every face of a closed box: no average normal.
    act(() => writeMeshFaceSelection(app, box.id, Uint32Array.from({ length: 12 }, (_, t) => t)));
    await click(panel.querySelector("[data-extrude-faces-run]"));
    expect(reportOf(panel)?.textContent).toContain("give a direction");
    expect(scene.canUndo()).toBe(false);
    expect(countLiveTriangles(scene.views(box.mesh).indices())).toBe(12);
  });
});
