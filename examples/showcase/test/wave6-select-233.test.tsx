// @vitest-environment jsdom
// Integration & unit tests for Issue #233: SelectPanel, FaceSelectionLayer helpers,
// FaceSelectionMarquee overlay, and real Wasm Scene / SceneStore / AppState wiring.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { SceneLike } from "fast-mesh-wasm";
import { createSceneStore, meshState, type SceneStore } from "fast-mesh-wasm/store";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import * as THREE from "three";
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { init, Scene } from "../../../packages/fast-mesh-wasm/dist/web.js";
import { SelectPanel } from "../src/features/select/ui/SelectPanel";
import { AppState, type Item } from "../src/app/state/app-state";
import {
  applyScreenFaceSelection,
  buildScreenSelectOptions,
  buildSelectedFacesGeometry,
  clientPointToNdc,
  combineFaceSelection,
  countLiveTriangles,
  FaceSelectionMarquee,
  invertFaceSelection,
  raycastVisibleItem,
  setFaceSelectionMarquee,
  writeMeshFaceSelection,
} from "../src/features/select/ui/FaceSelectionLayer";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

beforeAll(() =>
  init(
    readFileSync(
      resolve(__dirname, "../../../packages/fast-mesh-wasm/dist/wasm/fast_mesh_wasm_bg.wasm"),
    ),
  ),
);

const cleanups: (() => void)[] = [];

afterEach(() => {
  setFaceSelectionMarquee(null);
  while (cleanups.length > 0) {
    cleanups.pop()?.();
  }
});

function makeBoxGeometry(
  hx: number,
  hy: number,
  hz: number,
  cx = 0,
  cy = 0,
  cz = 0,
): { positions: Float64Array; indices: Uint32Array } {
  const x0 = cx - hx;
  const x1 = cx + hx;
  const y0 = cy - hy;
  const y1 = cy + hy;
  const z0 = cz - hz;
  const z1 = cz + hz;
  const positions = Float64Array.from([
    x0, y0, z0,
    x1, y0, z0,
    x1, y1, z0,
    x0, y1, z0,
    x0, y0, z1,
    x1, y0, z1,
    x1, y1, z1,
    x0, y1, z1,
  ]);
  const indices = Uint32Array.from([
    0, 2, 1, 0, 3, 2, // -Z
    4, 5, 6, 4, 6, 7, // +Z
    0, 1, 5, 0, 5, 4, // -Y
    2, 3, 7, 2, 7, 6, // +Y
    0, 4, 7, 0, 7, 3, // -X
    1, 2, 6, 1, 6, 5, // +X
  ]);
  return { positions, indices };
}

function createHarness(): { scene: Scene; store: SceneStore; app: AppState } {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  cleanups.push(() => {
    store.dispose();
    scene.free();
  });
  return { scene, store, app };
}

function addBoxItem(
  store: SceneStore,
  app: AppState,
  name: string,
  opts: { hx?: number; hy?: number; hz?: number; cx?: number; cy?: number; cz?: number } = {},
): Item {
  const hx = opts.hx ?? 1;
  const hy = opts.hy ?? 1;
  const hz = opts.hz ?? 1;
  const { positions, indices } = makeBoxGeometry(hx, hy, hz, opts.cx ?? 0, opts.cy ?? 0, opts.cz ?? 0);
  const item = app.addMesh(store, {
    name,
    positions,
    indices,
    center: [opts.cx ?? 0, opts.cy ?? 0, opts.cz ?? 0],
    radius: Math.hypot(hx, hy, hz),
    volumeId: null,
    preset: null,
  });
  return item;
}

function mountReact(element: React.ReactElement): { container: HTMLDivElement; root: Root } {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(element);
  });
  cleanups.push(() => {
    act(() => {
      root.unmount();
    });
    container.remove();
  });
  return { container, root };
}

async function clickAndFlush(button: Element): Promise<void> {
  await act(async () => {
    button.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    await Promise.resolve();
    await Promise.resolve();
  });
}

async function setRangeValue(input: HTMLInputElement, value: string): Promise<void> {
  await act(async () => {
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
    setter?.call(input, value);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
    await Promise.resolve();
  });
}

function liveTriangles(scene: SceneLike, mesh: Item["mesh"]): number {
  return countLiveTriangles(scene.views(mesh).indices());
}

describe("Issue #233 Face Selection pure helpers & marquee overlay", () => {
  it("combines selections across replace, add, subtract, and intersect modes cleanly", () => {
    const a = Uint32Array.of(5, 1, 3, 1);
    const b = Uint32Array.of(3, 8, 2);

    expect(Array.from(combineFaceSelection(a, b, "replace"))).toEqual([2, 3, 8]);
    expect(Array.from(combineFaceSelection(a, b, "add"))).toEqual([1, 2, 3, 5, 8]);
    expect(Array.from(combineFaceSelection(a, b, "subtract"))).toEqual([1, 5]);
    expect(Array.from(combineFaceSelection(a, b, "intersect"))).toEqual([3]);
    expect(Array.from(combineFaceSelection(a, Uint32Array.of(), "intersect"))).toEqual([]);
  });

  it("inverts selection while skipping dead [0, 0, 0] triangle slots", () => {
    const indices = Uint32Array.of(
      0, 1, 2, // tri 0 (live)
      0, 0, 0, // tri 1 (dead)
      1, 2, 3, // tri 2 (live)
      2, 3, 4, // tri 3 (live)
    );
    expect(countLiveTriangles(indices)).toBe(3);
    expect(Array.from(invertFaceSelection(indices, Uint32Array.of(0)))).toEqual([2, 3]);
    expect(Array.from(invertFaceSelection(indices, Uint32Array.of()))).toEqual([0, 2, 3]);
  });

  it("builds non-indexed highlight BufferGeometry skipping dead and out-of-range triangles", () => {
    const { positions, indices } = makeBoxGeometry(1, 2, 3);
    const geom = buildSelectedFacesGeometry(positions, indices, [2, 2, 99, 3]);
    cleanups.push(() => geom.dispose());

    const posAttr = geom.getAttribute("position") as THREE.BufferAttribute;
    expect(posAttr).toBeDefined();
    expect(posAttr.count).toBe(6); // 2 unique triangles * 3 vertices
    expect(geom.boundingSphere).not.toBeNull();
    // Both triangles 2 and 3 lie on the +Z face (z = +3).
    for (let i = 0; i < posAttr.count; i++) {
      expect(posAttr.getZ(i)).toBeCloseTo(3, 5);
    }

    const emptyGeom = buildSelectedFacesGeometry(positions, indices, []);
    cleanups.push(() => emptyGeom.dispose());
    expect(emptyGeom.getAttribute("position").count).toBe(0);
  });

  it("converts DOM client coordinates to NDC and builds camera ScreenSelectOptions", () => {
    const rect = { left: 100, top: 50, width: 200, height: 100 };
    expect(clientPointToNdc(100, 50, rect)).toEqual([-1, 1]);
    expect(clientPointToNdc(300, 150, rect)).toEqual([1, -1]);
    expect(clientPointToNdc(200, 100, rect)).toEqual([0, 0]);

    const ortho = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 100);
    ortho.position.set(0, 0, 10);
    ortho.lookAt(0, 0, 0);
    const opts = buildScreenSelectOptions(ortho, true, true, true);
    expect(opts.isOrthographic).toBe(true);
    expect(opts.visibleOnly).toBe(true);
    expect(opts.rejectBackfaces).toBe(true);
    expect(opts.viewProj).toHaveLength(16);
    expect(opts.cameraOrigin).toEqual([0, 0, 10]);
    expect(opts.cameraForward[2]).toBeCloseTo(-1, 5);
  });

  it("renders FaceSelectionMarquee SVG box and lasso states and clears cleanly", () => {
    const { container } = mountReact(<FaceSelectionMarquee />);
    expect(container.querySelector("[data-select-marquee-overlay]")).toBeNull();

    act(() => {
      setFaceSelectionMarquee({
        kind: "box",
        x0: 120,
        y0: 90,
        x1: 20,
        y1: 30,
        width: 400,
        height: 300,
      });
    });
    const rectEl = container.querySelector('[data-select-marquee="box"]');
    expect(rectEl).not.toBeNull();
    expect(rectEl?.getAttribute("x")).toBe("20");
    expect(rectEl?.getAttribute("y")).toBe("30");
    expect(rectEl?.getAttribute("width")).toBe("100");
    expect(rectEl?.getAttribute("height")).toBe("60");

    act(() => {
      setFaceSelectionMarquee({
        kind: "lasso",
        points: [
          [10, 10],
          [80, 15],
          [50, 70],
        ],
        width: 400,
        height: 300,
      });
    });
    const polyEl = container.querySelector('[data-select-marquee="lasso"]');
    expect(polyEl).not.toBeNull();
    expect(polyEl?.getAttribute("points")).toBe("10,10 80,15 50,70");

    act(() => {
      setFaceSelectionMarquee(null);
    });
    expect(container.querySelector("[data-select-marquee-overlay]")).toBeNull();
  });
});

describe("Issue #233 Real-WASM SelectPanel & Viewport Selection Integration", () => {
  it("supports tool switching, combine modes, screen selection, grow, shrink, invert, smooth, hole fill, and mask conversion", async () => {
    const { scene, store, app } = createHarness();
    const box = addBoxItem(store, app, "Cube", { hx: 1, hy: 1, hz: 1 });
    store.select(box.mesh);

    const { container } = mountReact(<SelectPanel app={app} store={store} />);

    // Initial state: tool off, replace combine, 0 / 12 triangles.
    const readout = () => container.querySelector("[data-select-readout]");
    expect(readout()?.getAttribute("data-select-count")).toBe("0");
    expect(readout()?.getAttribute("data-select-total")).toBe("12");

    // Toggle selection tools & Off disarm button.
    await clickAndFlush(container.querySelector('[data-select-tool="box"]')!);
    expect(app.getSnapshot().wave6.select.activeTool).toBe("box");
    await clickAndFlush(container.querySelector('[data-select-tool="lasso"]')!);
    expect(app.getSnapshot().wave6.select.activeTool).toBe("lasso");
    await clickAndFlush(container.querySelector('[data-select-tool="off"]')!);
    expect(app.getSnapshot().wave6.select.activeTool).toBeNull();

    // Switch combine mode and adjacency.
    await clickAndFlush(container.querySelector('[data-select-combine="add"]')!);
    expect(app.getSnapshot().wave6.select.combineMode).toBe("add");
    await clickAndFlush(container.querySelector('[data-select-adjacency="vertex"]')!);
    expect(app.getSnapshot().wave6.select.adjacency).toBe("vertex");
    await clickAndFlush(container.querySelector('[data-select-adjacency="edge"]')!);
    await clickAndFlush(container.querySelector('[data-select-combine="replace"]')!);

    // Screen selection facing +Z front face.
    const camera = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 100);
    camera.position.set(0, 0, 10);
    camera.lookAt(0, 0, 0);
    const screenOpts = buildScreenSelectOptions(camera, true, true, true);

    await act(async () => {
      const selected = await applyScreenFaceSelection({
        app,
        store,
        item: box,
        shape: { kind: "rect", minNdc: [-0.25, -0.25], maxNdc: [0.25, 0.25] },
        options: screenOpts,
        combineMode: "replace",
      });
      expect(selected.length).toBeGreaterThan(0);
    });

    const initialCount = app.getSnapshot().wave6.select.selectedByMesh[box.id].length;
    expect(initialCount).toBe(2); // +Z face has 2 triangles
    expect(readout()?.getAttribute("data-select-count")).toBe("2");

    // Highlight geometry reflects exact 2 front triangles.
    const view = scene.views(box.mesh);
    const hlGeom = buildSelectedFacesGeometry(
      view.positions(),
      view.indices(),
      app.getSnapshot().wave6.select.selectedByMesh[box.id],
    );
    cleanups.push(() => hlGeom.dispose());
    expect(hlGeom.getAttribute("position").count).toBe(6);

    // Seed single triangle to verify grow/shrink cleanly.
    act(() => {
      writeMeshFaceSelection(app, box.id, Uint32Array.of(2));
    });
    expect(readout()?.getAttribute("data-select-count")).toBe("1");

    await clickAndFlush(container.querySelector('[data-select-action="grow"]')!);
    const grownCount = app.getSnapshot().wave6.select.selectedByMesh[box.id].length;
    expect(grownCount).toBeGreaterThan(1);
    expect(readout()?.getAttribute("data-select-count")).toBe(String(grownCount));

    await clickAndFlush(container.querySelector('[data-select-action="shrink"]')!);
    const shrunkCount = app.getSnapshot().wave6.select.selectedByMesh[box.id].length;
    expect(shrunkCount).toBeLessThan(grownCount);

    // Invert selection -> complement of shrunkCount out of 12.
    await clickAndFlush(container.querySelector('[data-select-action="invert"]')!);
    const invertedCount = app.getSnapshot().wave6.select.selectedByMesh[box.id].length;
    expect(invertedCount).toBe(12 - shrunkCount);
    expect(readout()?.getAttribute("data-select-count")).toBe(String(invertedCount));

    // Convert face selection -> sculpt mask -> face selection roundtrip.
    act(() => {
      writeMeshFaceSelection(app, box.id, Uint32Array.of(2, 3));
    });
    await clickAndFlush(container.querySelector('[data-select-action="to-mask"]')!);
    expect(app.getSnapshot().sculptMask?.mesh).toBe(box.mesh);
    expect(app.getSnapshot().sculptMask?.vertices.length).toBeGreaterThanOrEqual(4);

    await clickAndFlush(container.querySelector('[data-select-action="clear"]')!);
    expect(app.getSnapshot().wave6.select.selectedByMesh[box.id].length).toBe(0);

    await clickAndFlush(container.querySelector('[data-select-action="from-mask"]')!);
    expect(Array.from(app.getSnapshot().wave6.select.selectedByMesh[box.id])).toEqual([2, 3]);

    // Check smooth-boundary, fill-selection-holes, select-islands, checkboxes & sliders.
    await clickAndFlush(container.querySelector('[data-select-action="fill-selection-holes"]')!);
    expect(app.getSnapshot().wave6.select.selectedByMesh[box.id].length).toBeGreaterThanOrEqual(2);

    await clickAndFlush(container.querySelector('[data-select-action="smooth-boundary"]')!);
    expect(app.getSnapshot().wave6.select.selectedByMesh[box.id].length).toBeGreaterThan(0);

    await clickAndFlush(container.querySelector('[data-select-combine="replace"]')!);
    await clickAndFlush(container.querySelector('[data-select-action="select-islands"]')!);
    expect(app.getSnapshot().wave6.select.selectedByMesh[box.id].length).toBe(12);

    const visibleBox = container.querySelector<HTMLInputElement>("[data-select-visible-only]")!;
    const cullBox = container.querySelector<HTMLInputElement>("[data-select-backface-cull]")!;
    await clickAndFlush(visibleBox);
    await clickAndFlush(cullBox);
    expect(app.getSnapshot().wave6.select.visibleOnly).toBe(false);
    expect(app.getSnapshot().wave6.select.backfaceCull).toBe(false);

    await setRangeValue(container.querySelector<HTMLInputElement>("[data-select-brush-radius]")!, "0.85");
    await setRangeValue(container.querySelector<HTMLInputElement>("[data-select-crease-angle]")!, "42");
    await setRangeValue(container.querySelector<HTMLInputElement>("[data-select-island-max-faces]")!, "36");
    expect(app.getSnapshot().wave6.select.brushRadius).toBeCloseTo(0.85, 4);
    expect(app.getSnapshot().wave6.select.creaseAngleDeg).toBe(42);
    expect(app.getSnapshot().wave6.select.islandMaxFaces).toBe(36);
  });

  it("extracts selected faces into a visible adopted mesh, splits, fills holes, deletes, and clears", async () => {
    const { scene, store, app } = createHarness();
    const box = addBoxItem(store, app, "SourceBox", { hx: 1, hy: 1, hz: 1 });
    store.select(box.mesh);

    const { container } = mountReact(<SelectPanel app={app} store={store} />);
    const readout = () => container.querySelector("[data-select-readout]");

    // Select 2 triangles (+Z face) -> Extract creates a new visible adopted item.
    act(() => {
      writeMeshFaceSelection(app, box.id, Uint32Array.of(2, 3));
    });
    await clickAndFlush(container.querySelector('[data-select-action="extract"]')!);

    const itemsAfterExtract = app.getSnapshot().items;
    expect(itemsAfterExtract).toHaveLength(2);
    const extractedItem = itemsAfterExtract[1];
    expect(extractedItem.name).toBe("SourceBox (Extract)");
    expect(meshState(store.getSnapshot(), extractedItem.mesh)?.desired.visible).toBe(true);
    expect(liveTriangles(scene, extractedItem.mesh)).toBe(2);
    expect(liveTriangles(scene, box.mesh)).toBe(12);

    // Split 2 triangles (-Z face: slots 0, 1) -> clears source selection & adopts visible split mesh.
    act(() => {
      writeMeshFaceSelection(app, box.id, Uint32Array.of(0, 1));
    });
    await clickAndFlush(container.querySelector('[data-select-action="split"]')!);

    const itemsAfterSplit = app.getSnapshot().items;
    expect(itemsAfterSplit).toHaveLength(3);
    const splitItem = itemsAfterSplit[2];
    expect(splitItem.name).toBe("SourceBox (Split)");
    expect(meshState(store.getSnapshot(), splitItem.mesh)?.desired.visible).toBe(true);
    expect(liveTriangles(scene, splitItem.mesh)).toBe(2);
    expect(app.getSnapshot().wave6.select.selectedByMesh[box.id].length).toBe(0);
    expect(liveTriangles(scene, box.mesh)).toBe(10);
    expect(readout()?.getAttribute("data-select-total")).toBe("10");

    // Fill Mesh Holes closes the clean open boundary left by split back to 12 triangles.
    await clickAndFlush(container.querySelector('[data-select-action="fill-holes"]')!);
    expect(liveTriangles(scene, box.mesh)).toBe(12);
    expect(readout()?.getAttribute("data-select-total")).toBe("12");

    // Delete 2 triangles -> reduces source mesh to 10 and clears selection cleanly.
    const liveIndices = scene.views(box.mesh).indices();
    const nextTwo: number[] = [];
    for (let t = 0; t * 3 + 2 < liveIndices.length && nextTwo.length < 2; t++) {
      const a = liveIndices[3 * t];
      const b = liveIndices[3 * t + 1];
      const c = liveIndices[3 * t + 2];
      if (a !== 0 || b !== 0 || c !== 0) {
        nextTwo.push(t);
      }
    }
    act(() => {
      writeMeshFaceSelection(app, box.id, Uint32Array.from(nextTwo));
    });
    await clickAndFlush(container.querySelector('[data-select-action="delete"]')!);
    expect(app.getSnapshot().wave6.select.selectedByMesh[box.id].length).toBe(0);
    expect(liveTriangles(scene, box.mesh)).toBe(10);
    expect(readout()?.getAttribute("data-select-total")).toBe("10");
    expect(readout()?.getAttribute("data-select-count")).toBe("0");
  });

  it("skips hidden meshes during raycastVisibleItem and selects crease/shell/sphere faces cleanly", async () => {
    const { scene, store, app } = createHarness();
    const hiddenFront = addBoxItem(store, app, "HiddenFront", { cx: 0, cy: 0, cz: 5 });
    const visibleBack = addBoxItem(store, app, "VisibleBack", { cx: 0, cy: 0, cz: 0 });
    store.setDisplay(hiddenFront.mesh, { visible: false });

    const hit = await raycastVisibleItem(
      store,
      app.getSnapshot().items,
      [0, 0, 10],
      [0, 0, -1],
      true,
    );
    expect(hit).not.toBeNull();
    expect(hit?.item.id).toBe(visibleBack.id);

    const creaseTris = await Promise.resolve(
      scene.repair.selectByCreaseAngle(visibleBack.mesh, hit!.hit.triangle, (35 * Math.PI) / 180),
    );
    expect(creaseTris.length).toBe(2);

    const shellTris = await Promise.resolve(
      scene.repair.selectConnectedShell(visibleBack.mesh, hit!.hit.triangle),
    );
    expect(shellTris.length).toBe(12);

    const brushTris = await Promise.resolve(
      scene.repair.selectFacesInSphere(visibleBack.mesh, hit!.hit.point, 0.5),
    );
    expect(brushTris.length).toBeGreaterThan(0);
  });
});
