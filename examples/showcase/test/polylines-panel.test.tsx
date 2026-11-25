// @vitest-environment jsdom
// The polyline wiring (#268), on the real engine: the Inspect panel's three
// Keep buttons, the rows in the mesh list, the polyline panel's tools with
// their units and reports, the palette entries, the tabs the panel is on, and
// what the viewport layer draws (and that rays pass through it).
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import * as THREE from "three";
import type { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { init, Scene } from "../../../packages/fast-mesh-wasm/dist/web.js";
import { AppState } from "../src/app/state/app-state";
import { MeshList } from "../src/features/home/ui/MeshList";
import { InspectPanel } from "../src/features/inspect/ui/InspectPanel";
import { queueSurfaceTap, surfacePathIdle } from "../src/features/inspect/ux/surface-path-actions";
import { PolylineLines } from "../src/features/polylines/ui/PolylineLayer";
import { PolylinePanel } from "../src/features/polylines/ui/PolylinePanel";
import { keepCurves } from "../src/features/polylines/ux/polyline-actions";
import { curveLength } from "../src/features/polylines/ux/polyline";
import { blobbySphere } from "../src/io/demo";
import { formatLength } from "../src/shared/ux/units";
import { WorkbenchRibbon } from "../src/shell/ui/Ribbon";
import { buildCommandPaletteItems } from "../src/shell/ux/command-registry";
import { tabMatchesSlot } from "../src/shell/ux/workbench-tabs";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// The rows' pictures draw on a canvas, which jsdom has not.
vi.mock("../src/features/home/ui/mesh-preview", () => ({ useMeshPreview: () => ({ url: null, empty: false }) }));

beforeAll(() =>
  init(readFileSync(resolve(__dirname, "../../../packages/fast-mesh-wasm/dist/wasm/fast_mesh_wasm_bg.wasm"))),
);

const cleanups: (() => Promise<void> | void)[] = [];
afterEach(async () => {
  for (const c of cleanups.splice(0).reverse()) {
    await c();
  }
  vi.restoreAllMocks();
});

const SQUARE = { points: [0, 0, 0, 25.4, 0, 0, 25.4, 25.4, 0, 0, 25.4, 0], closed: true };
const LINE = { points: [0, 0, 0, 50.8, 0, 0], closed: false };

async function mount() {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  app.follow(store);
  const d = blobbySphere([0, 0, 0], 1, 48, 32, 0);
  const item = app.addMesh(store, { name: "Ball", ...d, center: [0, 0, 0], radius: 1, volumeId: null, preset: null });
  store.select(item.mesh);
  store.commitFrame();
  scene.clearHistory();
  const focused: { center: readonly number[]; radius: number }[] = [];
  const container = document.createElement("div");
  document.body.appendChild(container);
  let root: Root;
  const nothing = () => {};
  await act(async () => {
    root = createRoot(container);
    root.render(
      createElement(
        SceneProvider,
        { store },
        createElement(WorkbenchRibbon, {
          app,
          mode: "translate",
          canUndo: false,
          canRedo: false,
          wireframe: false,
          selection: null,
          hasItems: true,
          hasVolume: false,
          onSetMode: nothing,
          onResetView: nothing,
          onFocusSelection: nothing,
          onClearSelection: nothing,
          onUndo: nothing,
          onRedo: nothing,
          onOpenHelp: nothing,
        }),
        createElement(MeshList, {
          app,
          onLoad: nothing,
          onCancelLoad: nothing,
          onLoadPhantom: nothing,
          onLoadSample: nothing,
          onLoadDemo: nothing,
          onFocus: nothing,
          onFocusBox: (box) => void focused.push(box),
        }),
        createElement(InspectPanel, { app, scene: scene as never }),
        createElement(PolylinePanel, { app }),
      ),
    );
  });
  cleanups.push(async () => {
    await act(async () => root.unmount());
    container.remove();
    store.dispose();
    scene.free();
  });
  return { app, store, scene, container, item, focused };
}

const q = <T extends Element = HTMLElement>(root: ParentNode, sel: string): T => {
  const el = root.querySelector<T>(sel);
  if (el === null) {
    throw new Error(`missing ${sel}`);
  }
  return el;
};

async function click(el: Element, shiftKey = false): Promise<void> {
  await act(async () => {
    el.dispatchEvent(new MouseEvent("click", { bubbles: true, shiftKey }));
    await new Promise((r) => setTimeout(r, 0));
  });
}

/** Types `value` into a controlled input or select, as a user would. */
async function type(el: HTMLInputElement | HTMLSelectElement, value: string): Promise<void> {
  await act(async () => {
    const proto = el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, "value")!.set!.call(el, value);
    el.dispatchEvent(new Event(el instanceof HTMLSelectElement ? "change" : "input", { bubbles: true }));
    await new Promise((r) => setTimeout(r, 0));
  });
}

const polylineAction = (root: ParentNode, name: string) => q<HTMLButtonElement>(root, `[data-polyline-action="${name}"]`);
const report = (root: ParentNode) => root.querySelector("[data-polyline-report]")?.textContent ?? null;
const undo = (store: SceneStore) => act(async () => store.undo());

describe("keeping from the Inspect panel", () => {
  it("keeps section contours, the section plane's contours and a surface path", async () => {
    const { app, scene, store, container } = await mount();
    expect(container.querySelector("[data-polyline-list]")).toBeNull();
    expect(container.querySelector("[data-panel=polylines]")!.textContent).toContain("Keep section contours");

    // Section contours: a stack of two.
    await type(q<HTMLInputElement>(container, "[data-inspect-contour-count]"), "2");
    await type(q<HTMLInputElement>(container, "[data-inspect-contour-step]"), "0.3");
    await click(q(container, '[data-action="inspect-slice-contours"]'));
    await click(q(container, '[data-action="inspect-keep-contours"]'));
    expect(app.getSnapshot().polylines.map((p) => p.name)).toEqual(["Ball contour 1", "Ball contour 2"]);
    expect(q(container, "[data-inspect-contours-kept]").textContent).toContain("Kept 2 polylines");
    expect(container.querySelectorAll("[data-polyline]")).toHaveLength(2);
    // Clearing the tool's slices leaves what was kept.
    await click(q(container, '[data-action="inspect-contours-clear"]'));
    expect(app.getSnapshot().polylines).toHaveLength(2);

    // The section plane.
    await act(async () => app.setInspectTool("section"));
    await click(q(container, '[data-action="section-enable"]'));
    await act(async () => void store.commitFrame());
    await click(q(container, '[data-action="section-keep"]'));
    expect(q(container, "[data-section-kept]").textContent).toBe('Kept as "Section".');
    await click(q(container, '[data-action="section-disable"]'));
    expect(app.getSnapshot().polylines.map((p) => p.name)).toContain("Section");

    // A surface path, once it has a leg.
    const keep = () => q<HTMLButtonElement>(container, '[data-action="path-keep"]');
    expect(keep().disabled).toBe(true);
    await click(q(container, '[data-action="path-pick"]'));
    await act(async () => {
      for (const dir of [[0.4, 0.8, 0.2], [-0.5, 0.7, 0.4], [0.1, 0.6, -0.8]]) {
        const n = Math.hypot(...dir);
        const u = dir.map((x) => x / n) as [number, number, number];
        const hit = scene.raycast([2 * u[0], 2 * u[1], 2 * u[2]], [-u[0], -u[1], -u[2]])!;
        void queueSurfaceTap(app, scene as never, Promise.resolve(hit));
      }
      await surfacePathIdle(app);
    });
    expect(keep().disabled).toBe(false);
    await click(keep());
    expect(q(container, "[data-path-status]").textContent).toBe('Kept as "Ball path".');
    const path = app.getSnapshot().polylines.find((p) => p.name === "Ball path")!;
    expect(path.closed).toBe(false);
    await click(q(container, '[data-action="path-close"]'));
    await act(async () => void (await surfacePathIdle(app)));
    await click(keep());
    expect(app.getSnapshot().polylines.find((p) => p.name === "Ball path 2")!.closed).toBe(true);
    // The path itself is still there, and each Keep is an undo step.
    expect(app.getSnapshot().surfacePath.path).not.toBeNull();
    await undo(store);
    await undo(store);
    expect(app.getSnapshot().polylines.map((p) => p.name)).toEqual(["Ball contour 1", "Ball contour 2", "Section"]);
    await undo(store);
    await undo(store);
    expect(container.querySelector("[data-polyline-list]")).toBeNull();
  });
});

describe("the rows in the mesh list", () => {
  it("show, hide, select, focus and remove, with lengths in the display unit", async () => {
    const { app, store, container, focused } = await mount();
    await act(async () => void keepCurves(app, store, [SQUARE, LINE], "Cut"));
    const row = (name: string) => q(container, `[data-polyline="${name}"]`);
    expect(q(row("Cut 1"), "[data-polyline-length]").textContent).toBe(formatLength(4 * 25.4, "mm"));
    await act(async () => app.setUnits("in"));
    expect(q(row("Cut 1"), "[data-polyline-length]").textContent).toBe("4 in");
    expect(q(row("Cut 2"), "[data-polyline-length]").textContent).toBe("2 in");
    expect(row("Cut 1").textContent).toContain("Closed");
    expect(row("Cut 2").textContent).toContain("Open");

    await click(q(row("Cut 2"), ".mesh-name"));
    expect(app.getSnapshot().polylineSelection).toEqual([2]);
    expect(row("Cut 2").classList.contains("selected")).toBe(true);
    await click(q(row("Cut 1"), ".mesh-name"), true);
    expect(app.getSnapshot().polylineSelection).toEqual([2, 1]);

    await click(q(row("Cut 1"), "[data-polyline-show]"));
    expect(app.getSnapshot().polylines[0].visible).toBe(false);
    const all = q<HTMLInputElement>(container, "[data-polyline-show-all]");
    expect(all.checked).toBe(false);
    await click(all);
    expect(app.getSnapshot().polylines.every((p) => p.visible)).toBe(true);

    await click(q(row("Cut 1"), "[data-polyline-focus]"));
    expect(focused).toHaveLength(1);
    expect(focused[0].center).toEqual([12.7, 12.7, 0]);
    expect(focused[0].radius).toBeCloseTo(12.7 * Math.SQRT2, 9);

    await click(q(row("Cut 1"), "[data-polyline-remove]"));
    expect(app.getSnapshot().polylines.map((p) => p.name)).toEqual(["Cut 2"]);
    await undo(store);
    expect(app.getSnapshot().polylines.map((p) => p.name)).toEqual(["Cut 1", "Cut 2"]);
  });
});

describe("the polyline panel", () => {
  it("works on the selection, in the display unit, with a report for each tool", async () => {
    const { app, store, container } = await mount();
    await act(async () => void keepCurves(app, store, [SQUARE, LINE], "Cut"));
    const panel = q(container, "[data-panel=polylines]");
    expect(q(panel, "[data-polyline-readout]").textContent).toContain("2 polylines");
    await click(q(container, '[data-polyline="Cut 1"] .mesh-name'));
    expect(q(panel, "[data-polyline-readout]").textContent).toContain("planar");
    await act(async () => app.setUnits("in"));
    expect(q(panel, "[data-polyline-readout]").textContent).toContain("4 in · 4 points · closed");

    // A tube of a tenth of an inch.
    await type(q<HTMLInputElement>(panel, '[data-polyline-length-input="radius"]'), "0.1");
    await type(q<HTMLInputElement>(panel, "[data-polyline-sides]"), "8");
    await click(polylineAction(panel, "tube"));
    expect(report(panel)).toBe("1 tube, 64 triangles.");
    const tube = app.getSnapshot().items.find((i) => i.name === "Cut 1 tube")!;
    const pos = store.scene.views(tube.mesh).positions();
    let z = 0;
    for (let i = 2; i < pos.length; i += 3) {
      z = Math.max(z, pos[i]);
    }
    expect(z).toBeCloseTo(2.54, 5);
    await undo(store);
    expect(app.getSnapshot().items.map((i) => i.name)).toEqual(["Ball"]);

    // An extrusion, its depth typed.
    await type(q<HTMLInputElement>(panel, '[data-polyline-length-input="depth"]'), "-0.5");
    await click(polylineAction(panel, "extrude"));
    expect(report(panel)).toBe("1 solid, 12 triangles.");
    expect(app.getSnapshot().items.map((i) => i.name)).toEqual(["Ball", "Cut 1 extrusion"]);
    const solid = app.getSnapshot().items[1];
    expect(solid.center[2]).toBeCloseTo(-6.35, 5);

    // An offset with sharp corners, then its points.
    await type(q<HTMLInputElement>(panel, '[data-polyline-length-input="offset"]'), "1");
    await type(q<HTMLSelectElement>(panel, "[data-polyline-corners]"), "sharp");
    await click(polylineAction(panel, "offset"));
    expect(report(panel)).toBe("Offset by 1 in: 1 polyline.");
    const offset = app.getSnapshot().polylines[2];
    expect(curveLength(offset)).toBeCloseTo(12 * 25.4, 6);
    expect(app.getSnapshot().polylineSelection).toEqual([offset.id]);
    await type(q<HTMLInputElement>(panel, '[data-polyline-length-input="spacing"]'), "0.5");
    await click(polylineAction(panel, "resample"));
    expect(report(panel)).toContain("Resample: 4 points →");
    await type(q<HTMLInputElement>(panel, '[data-polyline-length-input="tolerance"]'), "0.2");
    await click(polylineAction(panel, "simplify"));
    expect(report(panel)).toContain("Simplify:");

    // Its name and colour.
    await type(q<HTMLInputElement>(panel, "[data-polyline-name]"), "Rim");
    expect(app.getSnapshot().polylines[2].name).toBe("Rim");
    await type(q<HTMLInputElement>(panel, "[data-polyline-color]"), "#00ff00");
    expect(app.getSnapshot().polylines[2].color).toBe(0x00ff00);

    // What cannot be done is said.
    await click(q(container, '[data-polyline="Cut 2"] .mesh-name'));
    await click(polylineAction(panel, "extrude"));
    expect(report(panel)).toContain('"Cut 2" is open');
    expect(q(panel, "[data-polyline-report]").classList.contains("error")).toBe(true);
    await click(polylineAction(panel, "remove"));
    expect(report(panel)).toBe("Removed 1 polyline.");
    await click(polylineAction(panel, "select-all"));
    expect(app.getSnapshot().polylineSelection).toHaveLength(2);
  });

  it("exports the selected polylines, or every visible one", async () => {
    const { app, store, container } = await mount();
    const files: { name: string; text: string }[] = [];
    const blobs: Blob[] = [];
    vi.stubGlobal("URL", Object.assign(URL, { createObjectURL: (b: Blob) => `blob:${blobs.push(b)}`, revokeObjectURL: () => {} }));
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
      files.push({ name: this.download, text: "" });
    });
    await act(async () => void keepCurves(app, store, [SQUARE, { points: [0, 0, 9, 5, 0, 9], closed: false }], "Cut"));
    const panel = q(container, "[data-panel=polylines]");
    await click(q(container, '[data-polyline="Cut 1"] .mesh-name'));
    await click(polylineAction(panel, "export-svg"));
    expect(files[0].name).toBe("Cut 1.svg");
    expect(report(panel)).toBe("Exported 1 polyline to Cut 1.svg (mm).");
    // None selected: both, flattened onto the plane asked for.
    await act(async () => app.selectPolylines([]));
    await type(q<HTMLSelectElement>(panel, "[data-polyline-plane]"), "z");
    await click(polylineAction(panel, "export-dxf"));
    expect(files[1].name).toBe("polylines.dxf");
    expect(report(panel)).toBe("Exported 2 polylines to polylines.dxf (mm). Flattened: they span 9 mm across the plane.");
    await act(async () => app.setPolylineDisplay([app.getSnapshot().polylines[1].id], { visible: false }));
    await click(polylineAction(panel, "export-dxf"));
    expect(report(panel)).toBe("Exported 1 polyline to Cut 1.dxf (mm).");
    expect(blobs).toHaveLength(3);
    vi.unstubAllGlobals();
  });

  it("is on the Mesh Edit and Inspect tabs, with ribbon buttons and palette entries", async () => {
    const { app, store, container, item } = await mount();
    expect(tabMatchesSlot("edit", "polylines") && tabMatchesSlot("inspect", "polylines")).toBe(true);
    expect(tabMatchesSlot("sculpt", "polylines")).toBe(false);
    for (const tab of ["edit", "inspect"] as const) {
      await act(async () => app.setActiveTab(tab));
      q(container, '[data-ribbon-focus="polylines"]');
    }
    const commands = () =>
      buildCommandPaletteItems({
        app,
        items: [item],
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
    const command = (id: string) => commands().find((c) => c.id === `polyline:${id}`)!;
    expect(command("tube").disabled).toBe(true);
    await act(async () => void keepCurves(app, store, [SQUARE], "Cut"));
    expect(command("tube").disabled).toBe(false);
    await act(async () => app.setActiveTab("inspect"));
    await act(async () => command("offset").run());
    // It shows on the tab that was open.
    expect(app.getSnapshot().activeTab).toBe("inspect");
    expect(app.getSnapshot().polylines).toHaveLength(2);
    await act(async () => app.setActiveTab("home"));
    await act(async () => command("simplify").run());
    expect(app.getSnapshot().activeTab).toBe("edit");
    expect(report(container)).toContain("Simplify: ");
    const ids = commands().map((c) => c.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});

describe("the viewport lines", () => {
  it("draw each polyline in place and colour, and let rays through", () => {
    const app = new AppState();
    const lines = new PolylineLines();
    const a = { id: 1, name: "A", color: 0xff0000, visible: true, closed: true, points: Float64Array.from(SQUARE.points) };
    const b = { id: 2, name: "B", color: 0x00ff00, visible: false, closed: false, points: Float64Array.from(LINE.points) };
    lines.sync([a, b], [2]);
    expect(lines.size).toBe(2);
    const groupOf = (id: number) => lines.group.getObjectByName(`polyline:${id}`) as THREE.Group;
    const lineOf = (id: number) => groupOf(id).children[0] as LineSegments2;
    expect(groupOf(1).visible).toBe(true);
    expect(groupOf(2).visible).toBe(false);
    expect(lineOf(1).material.color.getHex()).toBe(0xff0000);
    expect(lineOf(2).material.linewidth).toBeGreaterThan(lineOf(1).material.linewidth);
    expect(lineOf(1).geometry.instanceCount).toBe(4);
    expect(lineOf(2).geometry.instanceCount).toBe(1);
    lines.group.updateMatrixWorld(true);
    const box = new THREE.Box3().copy(lineOf(1).geometry.boundingBox!).applyMatrix4(lineOf(1).matrixWorld);
    expect(box.min.toArray().map((x) => Math.round(x * 1e4) / 1e4)).toEqual([0, 0, 0]);
    expect(box.max.toArray().map((x) => Math.round(x * 1e4) / 1e4)).toEqual([25.4, 25.4, 0]);

    const caster = new THREE.Raycaster(new THREE.Vector3(10, 0, 5), new THREE.Vector3(0, 0, -1));
    caster.camera = new THREE.PerspectiveCamera();
    (caster.params as { Line2?: { threshold: number } }).Line2 = { threshold: 5 };
    expect(caster.intersectObject(lines.group, true)).toEqual([]);

    // An edit remakes its line, a display change does not; a removed one goes.
    const before = lineOf(1);
    lines.sync([{ ...a, color: 0x0000ff }, b], []);
    expect(lineOf(1)).toBe(before);
    expect(before.material.color.getHex()).toBe(0x0000ff);
    lines.sync([{ ...a, points: Float64Array.from(LINE.points), closed: false }], []);
    expect(lines.size).toBe(1);
    expect(lineOf(1)).not.toBe(before);
    expect(lineOf(1).geometry.instanceCount).toBe(1);
    lines.dispose();
    expect(lines.size).toBe(0);
    void app;
  });
});
