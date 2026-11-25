// @vitest-environment jsdom
// Face selection by criteria (#258) against the real engine: the engine-backed
// criteria (self-intersections, degenerate faces, contacts, components), the
// main-thread ones checked against the engine's own answers (overhang counts,
// the faces `orient` flips), and the select panel's wiring with every combine
// mode.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import type { MeshHandle, SceneLike } from "fast-mesh-wasm";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act } from "react";
import { createRoot } from "react-dom/client";
import * as THREE from "three";
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { init, Scene } from "../../../packages/fast-mesh-wasm/dist/web.js";
import { AppState, type Item } from "../src/app/state/app-state";
import { SelectPanel } from "../src/features/select/ui/SelectPanel";
import { buildEdgeAdjacency, type Pause, selectInconsistentOrientation, selectOverhangs } from "../src/features/select/ux/face-criteria";
import { writeMeshFaceSelection } from "../src/features/select/ux/face-selection-ops";
import {
  type CriteriaParams,
  DEFAULT_CRITERIA_PARAMS,
  expandSelectionToComponents,
  type FaceCriterion,
  findFacesByCriterion,
  snapshotGeometry,
} from "../src/features/select/ux/select-criteria";
import { view } from "../src/viewport/ux/view";

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

const noPause: Pause = () => Promise.resolve();

interface Buffers {
  positions: number[];
  indices: number[];
}

/** The closed box [c - h, c + h] wound outward: two triangles per face, -Z, +Z, -Y, +Y, -X, +X. */
function box(h: number, c: [number, number, number] = [0, 0, 0]): Buffers {
  const positions: number[] = [];
  for (const [x, y, z] of [
    [-1, -1, -1],
    [1, -1, -1],
    [1, 1, -1],
    [-1, 1, -1],
    [-1, -1, 1],
    [1, -1, 1],
    [1, 1, 1],
    [-1, 1, 1],
  ]) {
    positions.push(c[0] + h * x, c[1] + h * y, c[2] + h * z);
  }
  const indices = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5];
  return { positions, indices };
}

/** A closed UV sphere wound outward. */
function sphere(radius: number, stacks = 12, slices = 24, c: [number, number, number] = [0, 0, 0]): Buffers {
  const positions = [c[0], c[1], c[2] + radius];
  for (let i = 1; i < stacks; i++) {
    const phi = (Math.PI * i) / stacks;
    for (let j = 0; j < slices; j++) {
      const theta = (2 * Math.PI * j) / slices;
      positions.push(
        c[0] + radius * Math.sin(phi) * Math.cos(theta),
        c[1] + radius * Math.sin(phi) * Math.sin(theta),
        c[2] + radius * Math.cos(phi),
      );
    }
  }
  positions.push(c[0], c[1], c[2] - radius);
  const south = positions.length / 3 - 1;
  const ring = (i: number, j: number) => 1 + (i - 1) * slices + (j % slices);
  const indices: number[] = [];
  for (let j = 0; j < slices; j++) {
    indices.push(0, ring(1, j), ring(1, j + 1));
    indices.push(south, ring(stacks - 1, j + 1), ring(stacks - 1, j));
  }
  for (let i = 1; i < stacks - 1; i++) {
    for (let j = 0; j < slices; j++) {
      indices.push(ring(i, j), ring(i + 1, j), ring(i + 1, j + 1), ring(i, j), ring(i + 1, j + 1), ring(i, j + 1));
    }
  }
  return { positions, indices };
}

/** Several meshes as one: indices offset, triangles in order. */
function merge(...parts: Buffers[]): Buffers {
  const positions: number[] = [];
  const indices: number[] = [];
  for (const p of parts) {
    const base = positions.length / 3;
    positions.push(...p.positions);
    indices.push(...p.indices.map((v) => v + base));
  }
  return { positions, indices };
}

function flip(b: Buffers, tris: Iterable<number>): Buffers {
  const indices = b.indices.slice();
  for (const t of tris) {
    [indices[3 * t + 1], indices[3 * t + 2]] = [indices[3 * t + 2], indices[3 * t + 1]];
  }
  return { positions: b.positions, indices };
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

function addItem(store: SceneStore, app: AppState, name: string, b: Buffers, center: [number, number, number] = [0, 0, 0]): Item {
  return app.addMesh(store, {
    name,
    positions: Float64Array.from(b.positions),
    indices: Uint32Array.from(b.indices),
    center,
    radius: 1,
    volumeId: null,
    preset: null,
  });
}

function run(
  scene: SceneLike,
  mesh: MeshHandle,
  criterion: FaceCriterion,
  params: Partial<CriteriaParams> = {},
  others: MeshHandle[] = [],
): Promise<number[]> {
  return findFacesByCriterion(criterion, { ...DEFAULT_CRITERIA_PARAMS, ...params }, {
    scene,
    mesh,
    others,
    camera: null,
    adjacency: "edge",
    pause: noPause,
  }).then((r) => [...r]);
}

/** The triangle slots whose winding `repair.orient` reverses. */
function orientFlips(scene: Scene, mesh: MeshHandle): number[] {
  const before = scene.views(mesh).indices().slice();
  scene.repair.orient(mesh);
  const after = scene.views(mesh).indices();
  const out: number[] = [];
  for (let t = 0; t < before.length / 3; t++) {
    if (before[3 * t + 1] !== after[3 * t + 1] || before[3 * t + 2] !== after[3 * t + 2]) {
      out.push(t);
    }
  }
  return out;
}

describe("engine-backed criteria", () => {
  it("selects the self-intersecting faces of two crossing boxes in one mesh", async () => {
    const { scene, store, app } = harness();
    const crossed = addItem(store, app, "Crossed", merge(box(1), box(1, [1, 0.5, 0.3])));
    const single = addItem(store, app, "Single", box(1));
    const faces = await run(scene, crossed.mesh, "selfIntersecting");
    expect(faces).toEqual([...scene.inspect.selfIntersectingTriangles(crossed.mesh)]);
    expect(faces.some((t) => t < 12) && faces.some((t) => t >= 12)).toBe(true);
    // The faces inside the other box's span cross it; the far ones do not.
    expect(faces).not.toContain(8); // -X of the first box, at x = -1
    expect(faces).not.toContain(22); // +X of the second box, at x = 2
    expect(await run(scene, single.mesh, "selfIntersecting")).toEqual([]);
  });

  it("selects the faces analyze calls degenerate, and slivers by aspect", async () => {
    const { scene, store, app } = harness();
    // A box, then: a collinear triangle, a thin sliver, a repeated vertex id,
    // and a needle far thinner than the area tolerance.
    const b = box(1);
    const extra: Buffers = {
      positions: [3, 0, 0, 4, 0, 0, 5, 0, 0, 3, 2, 0, 13, 2, 0, 8, 2.05, 0, 3, 4, 0, 4, 4, 0, 20, 0, 0, 1020, 0, 0, 520, 1e-9, 0],
      indices: [0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10],
    };
    const item = addItem(store, app, "Mixed", merge(b, extra));
    const analysis = scene.inspect.analyze(item.mesh, { maxListed: 1000 });
    expect([...analysis.degenerateList]).toEqual([12, 14, 15]);
    // The sliver joins them; a higher sliver threshold leaves the analysis' list.
    expect(await run(scene, item.mesh, "degenerate")).toEqual([12, 13, 14, 15]);
    expect(await run(scene, item.mesh, "degenerate", { sliverAspect: 1e12 })).toEqual([...analysis.degenerateList]);
  });

  it("selects the faces touching or near another mesh", async () => {
    const { scene, store, app } = harness();
    const a = addItem(store, app, "A", box(1));
    // B overlaps A's +X side.
    const b = addItem(store, app, "B", box(1, [1.5, 0, 0]));
    const far = addItem(store, app, "Far", box(1, [0, 10, 0]));
    const touching = await run(scene, a.mesh, "contact", {}, [b.mesh, far.mesh]);
    expect(touching.length).toBeGreaterThan(0);
    // A's -X face (x = -1) is 1.5 from B, which starts at x = 0.5; its +X face lies inside B.
    expect(touching).not.toContain(8);
    expect(touching).not.toContain(9);
    expect(touching).toContain(10);
    // From B's side: B's +X face (x = 2.5) is outside A.
    const fromB = await run(scene, b.mesh, "contact", {}, [a.mesh]);
    expect(fromB).not.toContain(10);
    expect(fromB).not.toContain(11);
    // A gap of 1: none at margin 0, the facing sides within 1.5.
    const c = addItem(store, app, "C", box(1, [0, 0, 3]));
    expect(await run(scene, c.mesh, "contact", { contactMargin: 0 }, [a.mesh])).toEqual([]);
    const near = await run(scene, c.mesh, "contact", { contactMargin: 1.5 }, [a.mesh]);
    expect(near).toContain(0);
    expect(near).toContain(1);
    expect(near).not.toContain(2);
    await expect(run(scene, a.mesh, "contact", {}, [])).rejects.toThrow(/No other visible mesh/);
  });

  it("expands a selection to the components it touches", async () => {
    const { scene, store, app } = harness();
    const item = addItem(store, app, "Two", merge(box(1), box(1, [5, 0, 0]), box(1, [10, 0, 0])));
    expect([...(await expandSelectionToComponents(scene, item.mesh, [3]))]).toEqual(Array.from({ length: 12 }, (_, t) => t));
    expect([...(await expandSelectionToComponents(scene, item.mesh, [0, 30]))]).toEqual([
      ...Array.from({ length: 12 }, (_, t) => t),
      ...Array.from({ length: 12 }, (_, t) => 24 + t),
    ]);
  });
});

describe("main-thread criteria against the engine", () => {
  it("counts the overhangs inspect.overhangs counts", async () => {
    const { scene, store, app } = harness();
    const item = addItem(store, app, "Sphere", sphere(2, 16, 32, [1, 2, 3]), [1, 2, 3]);
    for (const [dir, angle] of [
      [[0, 0, 1], 45],
      [[0, 0, 1], 10],
      [[1, 0, 0], 30],
      [[0, -1, 0], 60],
    ] as const) {
      const report = scene.inspect.overhangs(item.mesh, { buildDirection: [...dir], criticalAngleDeg: angle });
      const mine = await selectOverhangs(snapshotGeometry(scene, item.mesh), [...dir], angle, noPause);
      expect(mine.length).toBe(report.overhangFaceCount);
      expect(mine.length).toBeGreaterThan(0);
    }
  });

  it("selects exactly the faces repair.orient reverses", async () => {
    const s = sphere(1, 10, 20);
    const tris = s.indices.length / 3;
    const all = Array.from({ length: tris }, (_, t) => t);
    // The sphere without its two caps (its first 2 x 20 triangles): open at both ends.
    const open: Buffers = { positions: s.positions, indices: s.indices.slice(3 * 40) };
    const cases: Buffers[] = [
      flip(s, [40, 41, 42, 60, 61]),
      flip(s, all),
      flip(s, all.filter((t) => t % 7 !== 0)),
      flip(open, [10, 11, 12, 13]),
      merge(flip(box(1), [3]), flip(box(1, [5, 0, 0]), Array.from({ length: 12 }, (_, t) => t))),
    ];
    for (const c of cases) {
      const { scene, store, app } = harness();
      const item = addItem(store, app, "Wound", c);
      const geom = snapshotGeometry(scene, item.mesh);
      const adj = await buildEdgeAdjacency(geom.indices, geom.positions.length / 3, noPause);
      const mine = [...(await selectInconsistentOrientation(geom, adj, noPause))];
      expect(mine).toEqual(orientFlips(scene, item.mesh));
    }
  });
});

/** Lets timers and the action's promises run, one act at a time so the DOM shows each update. */
async function settle(until: () => boolean): Promise<void> {
  for (let i = 0; i < 200 && !until(); i++) {
    await act(async () => {
      await new Promise((r) => setTimeout(r, 5));
    });
  }
}

function mount(app: AppState, store: SceneStore): HTMLDivElement {
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(<SelectPanel app={app} store={store} />));
  cleanups.push(() => {
    act(() => root.unmount());
    container.remove();
  });
  return container;
}

function change(el: HTMLSelectElement | HTMLInputElement, value: string): void {
  act(() => {
    const proto = el instanceof HTMLSelectElement ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(proto, "value")?.set?.call(el, value);
    el.dispatchEvent(new Event(el instanceof HTMLSelectElement ? "change" : "input", { bubbles: true }));
  });
}

function click(el: Element): void {
  act(() => {
    el.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
}

describe("select panel criteria wiring", () => {
  it("selects by criterion under every combine mode, and expands to components", async () => {
    const { store, app } = harness();
    const item = addItem(store, app, "Cube", box(1));
    store.select(item.mesh);
    const container = mount(app, store);
    const q = <T extends Element>(sel: string) => container.querySelector<T>(sel)!;
    const selected = () => [...(app.getSnapshot().wave6.select.selectedByMesh[item.id] ?? [])];
    const report = () => container.querySelector("[data-select-report]")?.textContent ?? "";
    const runCriterion = async (expected: number[]) => {
      const button = q<HTMLButtonElement>('[data-select-action="by-criterion"]');
      click(button);
      // Busy until the action lands.
      await settle(() => !button.disabled);
      expect(selected()).toEqual(expected);
    };

    // Overhangs with +Z up: the bottom.
    change(q<HTMLSelectElement>("[data-select-criterion]"), "overhang");
    await runCriterion([0, 1]);
    expect(report()).toMatch(/Overhangs: 2 faces matched \(2 selected\)/);

    // Add the top (build direction -Z).
    click(q('[data-select-combine="add"]'));
    change(q<HTMLSelectElement>("[data-select-build-axis]"), "-z");
    await runCriterion([0, 1, 2, 3]);

    // Subtract faces larger than 1 mm²: every face (each is 2 mm²).
    click(q('[data-select-combine="subtract"]'));
    change(q<HTMLSelectElement>("[data-select-criterion]"), "area");
    change(q<HTMLSelectElement>("[data-select-area-mode]"), "above");
    change(q<HTMLInputElement>("[data-select-area-threshold]"), "1");
    await runCriterion([]);

    // Intersect: of the sharp faces (all of them), keep what is selected.
    act(() => writeMeshFaceSelection(app, item.id, Uint32Array.of(4, 5)));
    click(q('[data-select-combine="intersect"]'));
    change(q<HTMLSelectElement>("[data-select-criterion]"), "sharp");
    await runCriterion([4, 5]);

    // Replace with the faces facing an orthographic camera looking down -Y.
    const camera = new THREE.OrthographicCamera(-5, 5, 5, -5, 0.1, 100);
    camera.position.set(0, -10, 0);
    camera.lookAt(0, 0, 0);
    const prevCamera = view.camera;
    view.camera = camera;
    cleanups.push(() => {
      view.camera = prevCamera;
    });
    click(q('[data-select-combine="replace"]'));
    change(q<HTMLSelectElement>("[data-select-criterion]"), "facing");
    await runCriterion([4, 5]);

    // A closed cube has no boundary: replace empties the selection.
    change(q<HTMLSelectElement>("[data-select-criterion]"), "boundary");
    await runCriterion([]);

    // Expand to components: one face grows to the whole cube.
    act(() => writeMeshFaceSelection(app, item.id, Uint32Array.of(7)));
    const expand = q<HTMLButtonElement>('[data-select-action="expand-components"]');
    expect(expand.disabled).toBe(false);
    click(expand);
    await settle(() => !expand.disabled);
    expect(selected()).toEqual(Array.from({ length: 12 }, (_, t) => t));
    expect(report()).toMatch(/Expanded to components \(12 faces\)/);
  });

  it("reads the area threshold in the display unit and reports failures", async () => {
    const { store, app } = harness();
    const item = addItem(store, app, "Cube", box(1));
    store.select(item.mesh);
    app.setUnits("in");
    const container = mount(app, store);
    const q = <T extends Element>(sel: string) => container.querySelector<T>(sel)!;
    const selected = () => [...(app.getSnapshot().wave6.select.selectedByMesh[item.id] ?? [])];

    // Below 0.01 in² (6.45 mm²): every 2 mm² face.
    change(q<HTMLSelectElement>("[data-select-area-mode]"), "below");
    change(q<HTMLInputElement>("[data-select-area-threshold]"), "0.01");
    expect(q<HTMLInputElement>("[data-select-area-threshold]").value).toBe("0.01");
    const button = q<HTMLButtonElement>('[data-select-action="by-criterion"]');
    click(button);
    await settle(() => !button.disabled);
    expect(selected().length).toBe(12);

    // No other mesh to touch: the error is reported and the selection kept.
    change(q<HTMLSelectElement>("[data-select-criterion]"), "contact");
    click(button);
    await settle(() => !button.disabled);
    expect(container.querySelector("[data-select-report]")?.textContent).toMatch(/No other visible mesh/);
    expect(selected().length).toBe(12);
  });
});
