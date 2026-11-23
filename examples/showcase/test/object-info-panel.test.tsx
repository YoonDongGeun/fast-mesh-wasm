// @vitest-environment jsdom
// The object info box's wiring (#257) on the real engine: the selected
// mesh's numbers are the SDK's (counts, area, volume) in the unit setting;
// the world/local toggle; a placement move, an edit and their undo refresh
// it once they settle; a collapsed box reads nothing; the overlay controls
// set the app state; the quality report downloads with the shown map's
// readouts; and the angle tool's clicks and its readout in the inspect panel.
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
import { ObjectInfoBox } from "../src/features/inspect/ui/ObjectInfoBox";
import { inspectMapProbe } from "../src/features/inspect/ux/use-inspect-maps";
import {
  LARGE_MEASURE_DELAY_MS,
  LARGE_MESH_TRIANGLES,
  OBJECT_INFO_REFRESH_MS,
  objectInfoProbe,
} from "../src/features/inspect/ux/use-object-info";
import { formatArea, formatLength, formatPoint, formatVolume } from "../src/shared/ux/units";
import { useMeshClick } from "../src/viewport/ux/use-selection-clicks";
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

/** An axis-aligned box from `lo` to `hi`, outward-wound: 8 vertices, 12 triangles. */
function box(lo: [number, number, number], hi: [number, number, number]): { positions: Float64Array; indices: Uint32Array } {
  const p: number[] = [];
  for (const z of [lo[2], hi[2]]) {
    for (const y of [lo[1], hi[1]]) {
      for (const x of [lo[0], hi[0]]) {
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
  const data = box([0, 0, 0], [4, 2, 1]);
  const a = app.addMesh(store, {
    name: "part",
    positions: data.positions,
    indices: data.indices,
    center: [2, 1, 0.5],
    radius: Math.hypot(2, 1, 0.5),
    volumeId: null,
    preset: null,
  });
  store.select(a.mesh);
  store.commitFrame();
  cleanups.push(() => {
    store.dispose();
  });
  return { scene, store, app, a };
}

function mount(app: AppState, store: SceneStore, scene: Scene, withInspect = false): HTMLElement {
  const host = document.createElement("div");
  document.body.appendChild(host);
  const root: Root = createRoot(host);
  act(() => {
    root.render(
      <SceneProvider store={store}>
        <ObjectInfoBox app={app} scene={scene} />
        {withInspect ? <InspectPanel app={app} scene={scene} /> : null}
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

async function settle(ms = OBJECT_INFO_REFRESH_MS + 100): Promise<void> {
  await act(async () => {
    await new Promise((r) => setTimeout(r, ms));
  });
}

async function click(el: HTMLElement): Promise<void> {
  await act(async () => {
    el.click();
    await new Promise((r) => setTimeout(r, 0));
  });
}

const value = (host: HTMLElement, key: string) => host.querySelector(`[data-info="${key}"]`)?.textContent ?? null;

describe("object info box (#257)", () => {
  it("shows the selected mesh's numbers, the SDK's measurements among them, in the unit setting", async () => {
    const { scene, store, app, a } = setup();
    const host = mount(app, store, scene);
    await settle(20);
    const sdk = scene.inspect.measure(a.mesh);
    expect(value(host, "triangles")).toBe(String(scene.triangleCount(a.mesh)));
    expect(value(host, "vertices")).toBe(String(scene.vertexCount(a.mesh)));
    expect(value(host, "edges")).toBe("18");
    expect(value(host, "components")).toBe(String(scene.inspect.analyze(a.mesh).components.length));
    expect(value(host, "area")).toBe(formatArea(sdk.area, "mm"));
    expect(value(host, "volume")).toBe(formatVolume(sdk.volume!, "mm"));
    expect(value(host, "size")).toBe("4 × 2 × 1 mm");
    expect(value(host, "min")).toBe(formatPoint([0, 0, 0], "mm"));
    expect(value(host, "max")).toBe(formatPoint([4, 2, 1], "mm"));
    expect(objectInfoProbe.mesh).toBe("part");
    expect(objectInfoProbe.info?.area).toBe(sdk.area);

    await act(async () => app.setUnits("in"));
    expect(value(host, "area")).toBe(formatArea(sdk.area, "in"));
    expect(value(host, "volume")).toBe(formatVolume(sdk.volume!, "in"));
    expect(value(host, "mean-edge")).toBe(formatLength(objectInfoProbe.info!.meanEdge, "in"));
    expect(value(host, "max")).toBe(formatPoint([4, 2, 1], "in"));
  });

  it("follows a placement move and its undo, world-aligned or local", async () => {
    const { scene, store, app, a } = setup();
    const host = mount(app, store, scene);
    await settle(20);
    // A quarter turn about z and a move.
    await act(async () => {
      store.setPlacement(a.mesh, { rotation: [0, -1, 0, 1, 0, 0, 0, 0, 1], translation: [10, 0, 0] });
      store.commitFrame();
    });
    await settle();
    expect(value(host, "size")).toBe("2 × 4 × 1 mm");
    // x' = -y + 10 and y' = x: the eight corners through the engine's own placement.
    const placed = store.scene.placement(a.mesh);
    const corners = [0, 1, 2, 3, 4, 5, 6, 7].map((i) => {
      const p = [i & 1 ? 4 : 0, i & 2 ? 2 : 0, i & 4 ? 1 : 0];
      return [0, 1, 2].map((r) => [0, 1, 2].reduce((sum, c) => sum + placed.rotation[3 * r + c] * p[c], placed.translation[r]));
    });
    const world = objectInfoProbe.info!.world!;
    for (const k of [0, 1, 2]) {
      expect(world.min[k]).toBeCloseTo(Math.min(...corners.map((c) => c[k])), 6);
      expect(world.max[k]).toBeCloseTo(Math.max(...corners.map((c) => c[k])), 6);
    }
    expect(world.min.map((x) => Math.round(x))).toEqual([8, 0, 0]);
    expect(world.max.map((x) => Math.round(x))).toEqual([10, 4, 1]);
    // The local frame keeps the mesh's own box.
    await click(q(host, '[data-info-space="local"]'));
    expect(app.getSnapshot().objectInfo.space).toBe("local");
    expect(value(host, "size")).toBe("4 × 2 × 1 mm");
    expect(value(host, "min")).toBe(formatPoint([0, 0, 0], "mm"));
    await click(q(host, '[data-info-space="world"]'));

    await act(async () => {
      store.undo();
      store.commitFrame();
    });
    await settle();
    expect(value(host, "size")).toBe("4 × 2 × 1 mm");
    expect(value(host, "min")).toBe(formatPoint([0, 0, 0], "mm"));
  });

  it("refreshes after an edit and its undo once they settle, and reads nothing while collapsed", async () => {
    const { scene, store, app, a } = setup();
    const host = mount(app, store, scene);
    await settle(20);
    const refreshes = objectInfoProbe.refreshes;
    await act(async () => {
      scene.edit(a.mesh, [{ op: "setPositions", vertices: [7], positions: Float64Array.from([8, 2, 1]) }]);
      store.commitFrame();
    });
    // Not before the burst settles.
    expect(value(host, "max")).toBe(formatPoint([4, 2, 1], "mm"));
    await settle();
    expect(value(host, "max")).toBe(formatPoint([8, 2, 1], "mm"));
    expect(value(host, "area")).toBe(formatArea(scene.inspect.measure(a.mesh).area, "mm"));
    expect(objectInfoProbe.refreshes).toBe(refreshes + 1);

    await act(async () => {
      scene.undo();
      store.commitFrame();
    });
    await settle();
    expect(value(host, "max")).toBe(formatPoint([4, 2, 1], "mm"));

    // Collapsed: no readouts, and edits cost nothing.
    await click(q(host, '[data-action="object-info-toggle"]'));
    expect(q(host, '[data-action="object-info-toggle"]').getAttribute("aria-expanded")).toBe("false");
    expect(host.querySelector("[data-object-info]")).toBeNull();
    const collapsed = objectInfoProbe.refreshes;
    await act(async () => {
      scene.edit(a.mesh, [{ op: "setPositions", vertices: [7], positions: Float64Array.from([9, 2, 1]) }]);
      store.commitFrame();
    });
    await settle();
    expect(objectInfoProbe.refreshes).toBe(collapsed);
    // Opening reads the mesh as it is now.
    await click(q(host, '[data-action="object-info-toggle"]'));
    await settle(20);
    expect(value(host, "max")).toBe(formatPoint([9, 2, 1], "mm"));

    // Without a selection it asks for one.
    await act(async () => {
      store.select(null);
      store.commitFrame();
    });
    expect(host.textContent).toContain("Select a mesh");
  });

  it("keeps the topology through moved vertices, and waits for a held stroke", async () => {
    const { scene, store, app, a } = setup();
    const host = mount(app, store, scene);
    await settle(20);
    const builds = objectInfoProbe.topologyBuilds;
    const edit = async (x: number) => {
      await act(async () => {
        scene.edit(a.mesh, [{ op: "setPositions", vertices: [7], positions: Float64Array.from([x, 2, 1]) }]);
        store.commitFrame();
      });
    };
    await edit(8);
    await settle();
    expect(value(host, "max")).toBe(formatPoint([8, 2, 1], "mm"));
    expect(objectInfoProbe.topologyBuilds).toBe(builds);
    // Removed triangles change the counts: built again.
    await act(async () => {
      scene.edit(a.mesh, [{ op: "removeTriangles", triangles: [0, 1] }]);
      store.commitFrame();
    });
    await settle();
    expect(value(host, "triangles")).toBe("10");
    expect(value(host, "volume")).toBe("none (open mesh)");
    expect(objectInfoProbe.topologyBuilds).toBe(builds + 1);

    view.stroking = true;
    try {
      await edit(9);
      await settle(3 * OBJECT_INFO_REFRESH_MS);
      expect(value(host, "max")).toBe(formatPoint([8, 2, 1], "mm"));
    } finally {
      view.stroking = false;
    }
    await settle();
    expect(value(host, "max")).toBe(formatPoint([9, 2, 1], "mm"));
  });

  it("on a large mesh waits for the brush to be put down, and measures after a longer rest", async () => {
    const { scene, store, app } = setup();
    // A flat grid just past the limit.
    const n = Math.ceil(Math.sqrt(LARGE_MESH_TRIANGLES / 2)) + 1;
    const positions = new Float64Array((n + 1) * (n + 1) * 3);
    const indices = new Uint32Array(n * n * 6);
    for (let j = 0, k = 0; j <= n; j++) {
      for (let i = 0; i <= n; i++, k++) {
        positions.set([i, j, 0], 3 * k);
        if (i < n && j < n) {
          indices.set([k, k + 1, k + n + 2, k, k + n + 2, k + n + 1], 6 * (j * n + i));
        }
      }
    }
    const big = app.addMesh(store, { name: "big", positions, indices, center: [n / 2, n / 2, 0], radius: n, volumeId: null, preset: null });
    store.select(big.mesh);
    store.commitFrame();
    const host = mount(app, store, scene);
    await settle(20);
    // The buffers' numbers are there at once, the measurement after the rest.
    expect(value(host, "triangles")).toBe((2 * n * n).toLocaleString("en"));
    expect(value(host, "components")).toBe("1");
    expect(value(host, "area")).toBe("measuring…");
    expect(q(host, "[data-object-info]").getAttribute("aria-busy")).toBe("true");
    await settle(LARGE_MEASURE_DELAY_MS + 300);
    expect(value(host, "area")).toBe(formatArea(scene.inspect.measure(big.mesh).area, "mm"));
    expect(q(host, "[data-object-info]").getAttribute("aria-busy")).toBe("false");

    // A brush in hand: the stroke's change waits, and the box says so.
    const measured = value(host, "area");
    const refreshes = objectInfoProbe.refreshes;
    await act(async () => app.setMaskPaint(true));
    expect(host.textContent).toContain("once the brush is put down");
    await act(async () => {
      scene.edit(big.mesh, [{ op: "setPositions", vertices: [0], positions: Float64Array.from([-50, 0, 0]) }]);
      store.commitFrame();
    });
    await settle(LARGE_MEASURE_DELAY_MS + 300);
    expect(value(host, "min")).toBe(formatPoint([0, 0, 0], "mm"));
    expect(objectInfoProbe.refreshes).toBe(refreshes);
    await act(async () => app.setMaskPaint(false));
    await settle();
    expect(value(host, "min")).toBe(formatPoint([-50, 0, 0], "mm"));
    // Until it is measured again the last area shows, dimmed.
    expect(value(host, "area")).toBe(measured);
    expect(q(host, '[data-info="area"]').className).toBe("stale");
    await settle(LARGE_MEASURE_DELAY_MS + 300);
    expect(q(host, '[data-info="area"]').className).toBe("");
    expect(objectInfoProbe.info?.area).toBe(scene.inspect.measure(big.mesh).area);
    expect(host.textContent).not.toContain("once the brush is put down");
  }, 20_000);

  it("drives the bounding box and normals overlays through the app state", async () => {
    const { scene, store, app } = setup();
    const host = mount(app, store, scene);
    await settle(20);
    await click(q(host, "[data-object-info-box]"));
    expect(app.getSnapshot().objectInfo.showBox).toBe(true);
    expect(host.querySelector("[data-normals-length]")).toBeNull();
    await click(q(host, '[data-normals-mode="vertex"]'));
    expect(app.getSnapshot().objectInfo.normals).toBe("vertex");
    const slider = q<HTMLInputElement>(host, "[data-normals-length]");
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
      setter.call(slider, "0.1");
      slider.dispatchEvent(new Event("input", { bubbles: true }));
    });
    expect(app.getSnapshot().objectInfo.normalsScale).toBeCloseTo(0.1, 12);
    // The length reads in the unit setting.
    expect(host.textContent).toContain(formatLength(0.1 * Math.hypot(2, 1, 0.5), "mm"));
    await click(q(host, '[data-normals-mode="off"]'));
    expect(app.getSnapshot().objectInfo.normals).toBe("off");
  });

  it("downloads the quality report with the repair analysis and the shown map's readouts", async () => {
    const { scene, store, app } = setup();
    const host = mount(app, store, scene);
    await settle(20);
    const blobs: Blob[] = [];
    const created = vi.fn((b: Blob) => {
      blobs.push(b);
      return "blob:report";
    });
    const savedCreate = URL.createObjectURL;
    const savedRevoke = URL.revokeObjectURL;
    URL.createObjectURL = created as typeof URL.createObjectURL;
    URL.revokeObjectURL = () => {};
    const clicked = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => {});
    const savedMap = inspectMapProbe.shown;
    inspectMapProbe.shown = {
      kind: "quality",
      mesh: "part",
      reference: null,
      readout: { past: 0, total: 12, min: 2, max: 2, mean: 2 },
      params: { metric: "aspectRatio", threshold: 10 },
      range: [1, 10],
      band: null,
    };
    cleanups.push(() => {
      URL.createObjectURL = savedCreate;
      URL.revokeObjectURL = savedRevoke;
      clicked.mockRestore();
      inspectMapProbe.shown = savedMap;
    });

    await click(q(host, '[data-action="object-info-report-csv"]'));
    await settle(20);
    expect(created).toHaveBeenCalledTimes(1);
    expect(objectInfoProbe.lastReport).toMatchObject({ format: "csv", fileName: "part-quality.csv" });
    const csv = await blobs[0].text();
    expect(csv).toContain("Object,Triangles,12,");
    expect(csv).toContain(`Object,Surface area,${scene.inspect.measure(app.getSnapshot().items[0].mesh).area},mm^2`);
    expect(csv).toContain("Repair analysis,Clean,yes,");
    expect(csv).toContain("Inspect map,Metric,Aspect ratio,");
    expect(csv).toContain("Inspect map,Triangles measured,12,");

    await click(q(host, '[data-action="object-info-report-html"]'));
    await settle(20);
    expect(objectInfoProbe.lastReport).toMatchObject({ format: "html", fileName: "part-quality.html" });
    const html = await blobs[1].text();
    expect(html).toContain("<h2>Inspect map</h2>");
    expect(blobs[1].type).toBe("text/html");
  });

  it("the angle tool's clicks snap to vertices in the world, never select, and a fourth starts again", async () => {
    const { scene, store, app, a } = setup();
    const host = document.createElement("div");
    const root: Root = createRoot(host);
    cleanups.push(() => act(() => root.unmount()));
    const onPick = vi.fn();
    let onClick: ReturnType<typeof useMeshClick> | null = null;
    function Clicker(props: { tool: "angle" | null }) {
      onClick = useMeshClick({
        app,
        mesh: a.mesh,
        scene,
        tool: props.tool,
        picking: false,
        sculpted: false,
        suppressClick: false,
        onPick,
        onPickPoint: () => {},
      });
      return null;
    }
    const show = (tool: "angle" | null) => act(() => root.render(<Clicker tool={tool} />));
    /** A click looking down at the top face, at world `x, y`. */
    const clickAt = async (x: number, y: number) => {
      await act(async () => {
        onClick!({
          stopPropagation() {},
          delta: 0,
          nativeEvent: { shiftKey: false },
          ray: { origin: { x, y, z: 50 }, direction: { x: 0, y: 0, z: -1 } },
        } as never);
        await new Promise((r) => setTimeout(r, 0));
      });
    };
    // Moved, so the points must be in the world, not the mesh's frame.
    store.setPlacement(a.mesh, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [10, 0, 0] });
    store.commitFrame();
    show("angle");
    await clickAt(13.9, 0.05);
    await clickAt(10.1, 0.02);
    await clickAt(10.05, 1.95);
    expect(app.getSnapshot().objectInfo.angle).toEqual([
      [14, 0, 1],
      [10, 0, 1],
      [10, 2, 1],
    ]);
    // A miss changes nothing; a fourth click starts a new angle.
    await clickAt(99, 99);
    expect(app.getSnapshot().objectInfo.angle).toHaveLength(3);
    await clickAt(13.9, 1.95);
    expect(app.getSnapshot().objectInfo.angle).toEqual([[14, 2, 1]]);
    expect(onPick).not.toHaveBeenCalled();
    // Undoing the move leaves the picked point where it is.
    store.undo();
    store.commitFrame();
    expect(app.getSnapshot().objectInfo.angle).toEqual([[14, 2, 1]]);
    // Disarmed, a click selects as before.
    show(null);
    await clickAt(1, 1);
    expect(onPick).toHaveBeenCalledTimes(1);
  });

  it("reads the angle out in the inspect panel", async () => {
    const { scene, store, app } = setup();
    const host = mount(app, store, scene, true);
    await act(async () => app.setInspectTool("angle"));
    expect(q(host, '[data-action="inspect-angle"]').getAttribute("aria-pressed")).toBe("true");
    expect(q(host, '[data-readout="angle"]').textContent).toContain("Click three vertices");
    await act(async () => app.setObjectInfo({ angle: [[4, 0, 0]] }));
    expect(q(host, '[data-readout="angle"]').textContent).toContain("1 of 3 points");
    await act(async () =>
      app.setObjectInfo({
        angle: [
          [4, 0, 0],
          [0, 0, 0],
          [0, 2, 0],
        ],
      }),
    );
    expect(q(host, "[data-angle-degrees]").textContent).toBe("90.0°");
    expect(q(host, '[data-readout="angle"]').textContent).toContain(`${formatLength(4, "mm")}, ${formatLength(2, "mm")}`);
    await click(q(host, '[data-action="angle-clear"]'));
    expect(app.getSnapshot().objectInfo.angle).toEqual([]);
  });
});
