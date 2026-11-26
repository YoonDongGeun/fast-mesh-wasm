// @vitest-environment jsdom
// The repair panel comes after what it follows (#285): its chunk may land after
// a mesh was selected, or several were. It analyses the selected mesh, and the
// Join source is seeded from the queue as if it had been there.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { init, Scene } from "../../../packages/fast-mesh-wasm/dist/web.js";
import { AppState, type Item } from "../src/app/state/app-state";
import { RepairPanel } from "../src/features/mesh-repair/ui/RepairPanel";
import { lazyPanel } from "../src/shared/ui/LazyPart";
import { LazyChunk } from "../src/shared/ux/lazy-chunk";

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

function quad(app: AppState, store: SceneStore, name: string, z: number): Item {
  return app.addMesh(store, {
    name,
    positions: Float64Array.from([0, 0, z, 1, 0, z, 1, 1, z, 0, 1, z]),
    indices: Uint32Array.from([0, 1, 2, 0, 2, 3]),
    center: [0, 0, 0],
    radius: 1,
    volumeId: null,
    preset: null,
  });
}

describe("the repair panel, landing late", () => {
  it("analyses the selected mesh and keeps the Join source its queue seeds", async () => {
    const scene = new Scene();
    const store = createSceneStore(scene);
    const app = new AppState();
    const off = app.follow(store);
    const a = quad(app, store, "A", 0);
    const b = quad(app, store, "B", 1);
    const c = quad(app, store, "C", 2);
    let land!: (m: { RepairPanel: typeof RepairPanel }) => void;
    const chunk = new LazyChunk("repair", () => new Promise<{ RepairPanel: typeof RepairPanel }>((done) => (land = done)));
    const Lazy = lazyPanel(chunk, (m) => m.RepairPanel, { title: "Repair", height: 40 });

    const host = document.createElement("div");
    document.body.appendChild(host);
    const root = createRoot(host);
    act(() =>
      root.render(
        <SceneProvider store={store}>
          <Lazy app={app} scene={scene} />
        </SceneProvider>,
      ),
    );
    cleanups.push(() => {
      act(() => root.unmount());
      host.remove();
      off();
      store.dispose();
      scene.free();
    });
    act(() => {
      store.select(a.mesh);
      app.setMulti([a.mesh, b.mesh]);
      void chunk.load();
    });
    expect(host.querySelector("[data-lazy-panel=repair]")).not.toBeNull();
    expect(app.getSnapshot().repairAnalysis).toBeNull();

    await act(async () => land({ RepairPanel }));
    const source = () => host.querySelector<HTMLSelectElement>("[data-join-source]");
    await act(async () => {
      await vi.waitFor(() => expect(app.getSnapshot().repairAnalysis?.mesh).toBe(a.mesh), { timeout: 5000 });
    });
    expect(host.querySelector("[data-repair]")?.getAttribute("data-repair")).toBe("A");
    expect(source()?.value).toBe(String(b.id));

    // Another mesh on its own: no source, as ever.
    await act(async () => {
      store.select(c.mesh);
      app.setMulti([c.mesh]);
    });
    expect(host.querySelector("[data-repair]")?.getAttribute("data-repair")).toBe("C");
    expect(source()?.value).toBe("");
  });
});
