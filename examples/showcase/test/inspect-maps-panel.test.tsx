// @vitest-environment jsdom
// The inspect maps section's wiring (#251) in the inspect panel, on the real
// engine: each map colours the selected mesh through the surface field
// overlay with a legend and the SDK's readouts; the tolerance restyles
// without a new query; units follow the setting; an edit and its undo
// refresh the map; the surface field and the maps take the overlay from each
// other; Clear and removing the reference clear it.
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
import { createRoot, type Root } from "react-dom/client";
import { AppState } from "../src/app/state/app-state";
import { InspectPanel } from "../src/features/inspect/ui/InspectPanel";
import { INSPECT_MAP_REFRESH_MS, inspectMapProbe } from "../src/features/inspect/ux/use-inspect-maps";
import { formatArea, formatLength } from "../src/shared/ux/units";
import { formatDistance } from "../src/features/collision/ux/collision-math";
import { view } from "../src/viewport/ux/view";

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

/** An axis-aligned cube of half-width `h`, outward-wound: 8 vertices, 12 triangles. */
function cube(h: number): { positions: Float64Array; indices: Uint32Array } {
  const p: number[] = [];
  for (const z of [-h, h]) {
    for (const y of [-h, h]) {
      for (const x of [-h, h]) {
        p.push(x, y, z);
      }
    }
  }
  const quads = [
    [0, 2, 3, 1],
    [4, 5, 7, 6],
    [0, 1, 5, 4],
    [2, 6, 7, 3],
    [0, 4, 6, 2],
    [1, 3, 7, 5],
  ];
  const t: number[] = [];
  for (const [a, b, c, d] of quads) {
    t.push(a, b, c, a, c, d);
  }
  return { positions: Float64Array.from(p), indices: Uint32Array.from(t) };
}

function setup() {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  const add = (name: string, h: number) => {
    const data = cube(h);
    return app.addMesh(store, {
      name,
      positions: data.positions,
      indices: data.indices,
      center: [0, 0, 0],
      radius: h * Math.sqrt(3),
      volumeId: null,
      preset: null,
    });
  };
  const a = add("inner", 1);
  const b = add("outer", 1.25);
  store.select(a.mesh);
  store.commitFrame();
  cleanups.push(() => {
    store.dispose();
  });
  return { scene, store, app, a, b };
}

function mount(app: AppState, store: SceneStore, scene: Scene): HTMLElement {
  const host = document.createElement("div");
  document.body.appendChild(host);
  const root: Root = createRoot(host);
  act(() => {
    root.render(
      <SceneProvider store={store}>
        <InspectPanel app={app} scene={scene} />
      </SceneProvider>,
    );
  });
  cleanups.push(() => {
    act(() => {
      root.unmount();
    });
    host.remove();
  });
  return host;
}

const q = <T extends Element = HTMLElement>(host: HTMLElement, selector: string): T => {
  const el = host.querySelector<T>(selector);
  if (el === null) {
    throw new Error(`missing ${selector}`);
  }
  return el;
};

async function click(el: HTMLElement): Promise<void> {
  await act(async () => {
    el.click();
    await new Promise((r) => setTimeout(r, 0));
  });
}

async function settle(ms = INSPECT_MAP_REFRESH_MS + 100): Promise<void> {
  await act(async () => {
    await new Promise((r) => setTimeout(r, ms));
  });
}

const readout = (host: HTMLElement) => host.querySelector("[data-inspect-map-readout]")?.textContent ?? null;

describe("inspect maps section (#251)", () => {
  it("shows a deviation map with the SDK's readouts, a legend and a tolerance band", async () => {
    const { scene, store, app, a, b } = setup();
    const host = mount(app, store, scene);
    expect(q(host, '[data-inspect-map-kind="deviation"]').getAttribute("aria-pressed")).toBe("true");

    const runs = inspectMapProbe.runs;
    await click(q(host, '[data-action="inspect-map-show"]'));
    const sdk = scene.inspect.meshDeviation(a.mesh, b.mesh, { signed: false });
    const overlay = app.getSnapshot().surfaceFieldValues;
    expect(overlay?.mesh).toBe(a.mesh);
    expect(Array.from(overlay?.values ?? [])).toEqual(Array.from(sdk.vertexDeviations));
    expect(overlay?.band ?? null).toBeNull();
    expect(readout(host)).toContain(`max ${formatLength(sdk.max, "mm")}`);
    expect(readout(host)).toContain(`mean ${formatLength(sdk.mean, "mm")}`);
    expect(readout(host)).toContain(`RMS ${formatLength(sdk.rms, "mm")}`);
    expect(readout(host)).toContain("to outer");
    expect(host.querySelector("[data-inspect-maps] .legend-bar")).not.toBeNull();
    expect(inspectMapProbe.runs).toBe(runs + 1);
    expect(inspectMapProbe.shown).toMatchObject({ kind: "deviation", mesh: "inner", reference: "outer" });

    // The tolerance band restyles without a new query.
    await click(q(host, "[data-inspect-map-tolerance-on]"));
    expect(app.getSnapshot().surfaceFieldValues?.band).toMatchObject({ lo: -0.1, hi: 0.1 });
    expect(app.getSnapshot().surfaceFieldValues?.values).toBe(overlay?.values);
    expect(q(host, "[data-legend-band]").textContent).toContain("within 0.1 mm");
    expect(inspectMapProbe.runs).toBe(runs + 1);

    // Units follow the setting.
    await act(async () => app.setUnits("in"));
    expect(readout(host)).toContain(`max ${formatLength(sdk.max, "in")}`);
    expect(q(host, "[data-legend-band]").textContent).toContain("in");

    // Signed runs the query again.
    await click(q(host, "[data-inspect-map-signed]"));
    const signed = scene.inspect.meshDeviation(a.mesh, b.mesh, { signed: true });
    expect(readout(host)).toContain(`max ${formatLength(signed.max, "in")}`);
    expect(readout(host)).toContain("signed, to outer");
    expect(inspectMapProbe.runs).toBe(runs + 2);

    // Clear removes it.
    await click(q(host, '[data-action="inspect-map-clear"]'));
    expect(app.getSnapshot().surfaceFieldValues).toBeNull();
    expect(readout(host)).toBeNull();
    expect(inspectMapProbe.shown).toBeNull();
  });

  it("refreshes on an edit and its undo, and clears when the reference goes", async () => {
    const { scene, store, app, a, b } = setup();
    const host = mount(app, store, scene);
    await click(q(host, '[data-action="inspect-map-show"]'));
    const before = scene.inspect.meshDeviation(a.mesh, b.mesh).max;
    expect(readout(host)).toContain(`max ${formatLength(before, "mm")}`);

    // Pull one corner far out of the reference: the map follows once the edit settles.
    const runs = inspectMapProbe.runs;
    await act(async () => {
      scene.edit(a.mesh, [{ op: "setPositions", vertices: [7], positions: Float64Array.from([3, 3, 3]) }]);
      store.commitFrame();
    });
    await settle();
    const after = scene.inspect.meshDeviation(a.mesh, b.mesh).max;
    expect(after).toBeGreaterThan(before);
    expect(inspectMapProbe.runs).toBe(runs + 1);
    expect(readout(host)).toContain(`max ${formatLength(after, "mm")}`);
    expect(Array.from(app.getSnapshot().surfaceFieldValues?.values ?? [])).toEqual(
      Array.from(scene.inspect.meshDeviation(a.mesh, b.mesh).vertexDeviations),
    );

    await act(async () => {
      scene.undo();
      store.commitFrame();
    });
    await settle();
    expect(inspectMapProbe.runs).toBe(runs + 2);
    expect(readout(host)).toContain(`max ${formatLength(before, "mm")}`);

    // A frame without a geometry change leaves it alone.
    await act(async () => {
      store.commitFrame();
    });
    await settle();
    expect(inspectMapProbe.runs).toBe(runs + 2);

    // Removing the reference clears the map.
    await act(async () => {
      app.removeMesh(store, b);
      store.commitFrame();
    });
    await settle(20);
    expect(app.getSnapshot().surfaceFieldValues).toBeNull();
    expect(readout(host)).toBeNull();
  });

  it("waits for a held stroke to be released before refreshing", async () => {
    const { scene, store, app, a } = setup();
    const host = mount(app, store, scene);
    await click(q(host, '[data-inspect-map-kind="quality"]'));
    await click(q(host, '[data-action="inspect-map-show"]'));
    const runs = inspectMapProbe.runs;
    view.stroking = true;
    try {
      await act(async () => {
        scene.edit(a.mesh, [{ op: "setPositions", vertices: [7], positions: Float64Array.from([1, 1, 3]) }]);
        store.commitFrame();
      });
      await settle(INSPECT_MAP_REFRESH_MS * 3);
      expect(inspectMapProbe.runs).toBe(runs);
    } finally {
      view.stroking = false;
    }
    await settle();
    expect(inspectMapProbe.runs).toBe(runs + 1);
    const quality = scene.inspect.triangleQuality(a.mesh, { metric: "aspectRatio", sliverThreshold: 10 });
    expect(readout(host)).toContain(`min ${formatDistance(quality.min)}`);
  });

  it("overhang and quality maps, one at a time with the surface field", async () => {
    const { scene, store, app, a } = setup();
    const host = mount(app, store, scene);

    await click(q(host, '[data-inspect-map-kind="overhang"]'));
    await click(q(host, '[data-action="inspect-map-show"]'));
    const over = scene.inspect.overhangs(a.mesh, { buildDirection: [0, 0, 1], criticalAngleDeg: 45 });
    expect(readout(host)).toContain(`overhang area ${formatArea(over.totalSupportArea, "mm")} · ${over.overhangFaceCount} faces`);
    expect(over.overhangFaceCount).toBe(2);
    expect(app.getSnapshot().surfaceFieldValues).toMatchObject({ mesh: a.mesh, min: 45, max: 90 });
    expect(q(host, "[data-legend-band]").textContent).toContain("no support");

    // While shown, a parameter change runs it again on the same mesh.
    const critical = q<HTMLInputElement>(host, "[data-inspect-map-critical]");
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
      setter?.call(critical, "95");
      critical.dispatchEvent(new Event("input", { bubbles: true }));
      await new Promise((r) => setTimeout(r, 0));
    });
    expect(readout(host)).toContain("critical 90°");
    expect(readout(host)).toContain("0 faces");

    // Switching the kind maps the same mesh.
    await click(q(host, '[data-inspect-map-kind="quality"]'));
    const quality = scene.inspect.triangleQuality(a.mesh, { metric: "aspectRatio", sliverThreshold: 10 });
    expect(readout(host)).toContain(`${quality.sliverTriangleCount} of ${quality.liveTriangleCount} triangles above 10`);
    expect(inspectMapProbe.shown?.kind).toBe("quality");

    // A surface field mode takes the overlay: the map goes.
    await click(q(host, '[data-inspect-field-mode="meanCurvature"]'));
    expect(app.getSnapshot().surfaceField.mode).toBe("meanCurvature");
    expect(readout(host)).toBeNull();
    expect(inspectMapProbe.shown).toBeNull();

    // Showing a map turns the surface field off.
    await click(q(host, '[data-action="inspect-map-show"]'));
    expect(app.getSnapshot().surfaceField.mode).toBe("none");
    expect(app.getSnapshot().surfaceFieldValues?.mesh).toBe(a.mesh);
    expect(readout(host)).toContain("triangles above 10");
    expect(scene.inspect.coloring(a.mesh)).toBeNull();
  });

  it("asks for a selection and a second mesh", async () => {
    const { scene, store, app, b } = setup();
    const host = mount(app, store, scene);
    await act(async () => {
      store.select(null);
      store.commitFrame();
    });
    expect(q<HTMLButtonElement>(host, '[data-action="inspect-map-show"]').disabled).toBe(true);
    expect(host.textContent).toContain("Select a mesh to map.");

    await act(async () => {
      app.removeMesh(store, b);
      store.commitFrame();
      store.select(app.getSnapshot().items[0].mesh);
      store.commitFrame();
    });
    expect(q<HTMLButtonElement>(host, '[data-action="inspect-map-show"]').disabled).toBe(true);
    expect(host.textContent).toContain("Load a second mesh");
    await click(q(host, '[data-inspect-map-kind="quality"]'));
    expect(q<HTMLButtonElement>(host, '[data-action="inspect-map-show"]').disabled).toBe(false);
  });
});
