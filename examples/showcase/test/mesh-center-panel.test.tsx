// @vitest-environment jsdom
// What the panels centre on a mesh whose origin is far from the world origin
// (#277), on the real engine: the section plane, the clip box and contour
// slices of the inspect panel, and the collision probe's nudge.
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

vi.mock("fast-mesh-wasm", async () => {
  return await import("../../../packages/fast-mesh-wasm/dist/web.js");
});

import { init, type PlacementInput, Scene, type Vec3 } from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { AppState, type Item } from "../src/app/state/app-state";
import { CollisionPanel } from "../src/features/collision/ui/CollisionPanel";
import { InspectPanel } from "../src/features/inspect/ui/InspectPanel";
import { roundedBox } from "../src/io/demo";
import { bounds } from "../src/io/mesh-data";

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

const FAR: Vec3 = [1000, 2000, 3000];

function setup() {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  cleanups.push(() => {
    store.dispose();
    scene.free();
  });
  return { scene, store, app };
}

/** A 2 x 2 x 2 box around `centre`, its origin. */
function addBox(app: AppState, store: SceneStore, name: string, centre: Vec3, placement?: PlacementInput): Item {
  const { positions, indices } = roundedBox(centre, [1, 1, 1], 0.08, 4);
  const item = app.addMesh(store, { name, positions, indices, ...bounds(positions), volumeId: null, preset: null });
  if (placement !== undefined) {
    store.setPlacement(item.mesh, placement);
  }
  return item;
}

function mount(store: SceneStore, panel: ReactNode): HTMLElement {
  const host = document.createElement("div");
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => root.render(<SceneProvider store={store}>{panel}</SceneProvider>));
  cleanups.push(() => {
    act(() => root.unmount());
    host.remove();
  });
  return host;
}

async function click(host: HTMLElement, selector: string): Promise<void> {
  const button = host.querySelector<HTMLButtonElement>(selector);
  expect(button, selector).not.toBeNull();
  await act(async () => {
    button!.click();
    await new Promise((r) => setTimeout(r, 0));
  });
}

function expectPoint(got: Readonly<Vec3>, want: Readonly<Vec3>): void {
  for (let k = 0; k < 3; k++) {
    expect(got[k]).toBeCloseTo(want[k], 3);
  }
}

describe("a mesh far from the world origin", () => {
  it.each([
    ["by its origin", undefined, FAR],
    [
      "by its origin, a turn about z and a scale",
      { rotation: [0, -1, 0, 1, 0, 0, 0, 0, 1], translation: [5, 0, -7], scale: [1, 1, 2] },
      [-1995, 1000, 5993],
    ],
  ] as [string, PlacementInput | undefined, Vec3][])(
    "has the section, the clip box and contour slices through its middle: %s",
    async (_, placement, middle) => {
      const { scene, store, app } = setup();
      const item = addBox(app, store, "Far", FAR, placement);
      store.select(item.mesh);
      store.commitFrame();
      const host = mount(store, <InspectPanel app={app} scene={scene} />);

      await click(host, "[data-action=inspect-section]");
      await click(host, "[data-action=section-enable]");
      expectPoint(app.getSnapshot().inspectSection!.plane.origin, middle);
      await click(host, "[data-axis=x]");
      expectPoint(app.getSnapshot().inspectSection!.plane.origin, middle);
      expect(app.getSnapshot().inspectSection!.plane.normal).toEqual([1, 0, 0]);

      await click(host, "[data-action=inspect-clip]");
      await click(host, "[data-action=clip-fit]");
      const { min, max } = app.getSnapshot().clipBox!;
      expectPoint([(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2], middle);

      await click(host, "[data-action=inspect-slice-contours]");
      const [slice] = app.getSnapshot().wave6.metrology.contourSlices!;
      expectPoint(slice.planeOrigin, middle);
      expect(slice.totalArea).toBeGreaterThan(3.9);
    },
  );

  it("has another mesh nudged toward it", async () => {
    const { scene, store, app } = setup();
    addBox(app, store, "A", FAR);
    const b = addBox(app, store, "B", [FAR[0], FAR[1] + 8, FAR[2]]);
    store.commitFrame();
    // The first two meshes are the pair.
    const host = mount(store, <CollisionPanel app={app} scene={scene} />);
    await click(host, "[data-collision-nudge]");
    expectPoint(scene.placement(b.mesh).translation, [0, -4, 0]);
  });
});
