// @vitest-environment jsdom
// Cancelling geodesic queries (#274): the inspect panel's Cancel buttons for
// the map and the path, a newer ask ending the query under way, what
// undoing and clearing a path cancels, and the operators freed with the
// panel. The queries are stand-ins that wait for an answer or their signal,
// except at the end, where a Worker scene runs real ones.
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { Worker as NodeThread } from "node:worker_threads";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";

vi.mock("fast-mesh-wasm", async () => {
  return await import("../../../packages/fast-mesh-wasm/dist/web.js");
});

import { init, type MeshHandle, type RayHit, Scene, type SceneLike } from "fast-mesh-wasm";
import { SceneProvider } from "fast-mesh-wasm/react";
import { createSceneStore, type SceneStore } from "fast-mesh-wasm/store";
import { createWorkerScene, type WorkerLike } from "fast-mesh-wasm/worker";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { AppState } from "../src/app/state/app-state";
import { InspectPanel } from "../src/features/inspect/ui/InspectPanel";
import { runGeodesicMap } from "../src/features/inspect/ux/geodesic-map";
import { cancelSurfaceLeg, queueSurfaceTap, surfacePathIdle } from "../src/features/inspect/ux/surface-path-actions";
import { traceLeg } from "../src/features/inspect/ux/surface-paths";
import { geodesicMapProbe } from "../src/features/inspect/ux/use-geodesic-map";
import { blobbySphere } from "../src/io/demo";

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

type V3 = [number, number, number];

function setup() {
  const scene = new Scene();
  const store = createSceneStore(scene);
  const app = new AppState();
  const d = blobbySphere([0, 0, 0], 1, 32, 16, 0);
  const item = app.addMesh(store, {
    name: "Ball",
    positions: d.positions,
    indices: d.indices,
    center: [0, 0, 0],
    radius: 1,
    volumeId: null,
    preset: null,
  });
  store.select(item.mesh);
  store.commitFrame();
  scene.clearHistory();
  cleanups.push(() => {
    store.dispose();
    scene.free();
  });
  return { scene, store, app, item };
}

function mount(app: AppState, store: SceneStore, scene: Scene): { host: HTMLElement; root: Root } {
  const host = document.createElement("div");
  document.body.appendChild(host);
  const root = createRoot(host);
  act(() => {
    root.render(
      <SceneProvider store={store}>
        <InspectPanel app={app} scene={scene} />
      </SceneProvider>,
    );
  });
  cleanups.push(() => {
    act(() => root.unmount());
    host.remove();
  });
  return { host, root };
}

/** Replaces a query of `scene` by one that waits for an answer, or rejects when its signal aborts. */
function waiting(scene: Scene, name: "geodesicDistances" | "shortestSurfacePath") {
  const inspect = scene.inspect as unknown as Record<string, (...args: unknown[]) => unknown>;
  const real = inspect[name];
  const asks: { args: unknown[]; signal: AbortSignal; answer: () => Promise<void> }[] = [];
  inspect[name] = (...args: unknown[]) =>
    new Promise((res, rej) => {
      const options = args[args.length - 1] as { signal: AbortSignal };
      options.signal.addEventListener("abort", () => rej(options.signal.reason));
      asks.push({
        args,
        signal: options.signal,
        answer: () =>
          act(async () => {
            res(real(...args.slice(0, -1), { ...options, signal: undefined }));
            await new Promise((r) => setTimeout(r, 0));
          }),
      });
    });
  return asks;
}

const find = (host: HTMLElement, name: string) => host.querySelector<HTMLButtonElement>(`[data-action="${name}"]`);

async function click(host: HTMLElement, name: string): Promise<void> {
  const el = find(host, name);
  if (el === null) {
    throw new Error(`missing ${name}`);
  }
  await act(async () => {
    el.click();
    await new Promise((r) => setTimeout(r, 0));
  });
}

function hitToward(scene: Scene, dir: V3): RayHit {
  const n = Math.hypot(...dir);
  const u: V3 = [dir[0] / n, dir[1] / n, dir[2] / n];
  const hit = scene.raycast([1.5 * u[0], 1.5 * u[1], 1.5 * u[2]], [-u[0], -u[1], -u[2]]);
  if (hit === null) {
    throw new Error("missed");
  }
  return hit;
}

/** Queues a tap; with `wait`, until the queue is done. */
async function tap(app: AppState, scene: Scene, dir: V3, wait = true): Promise<void> {
  await act(async () => {
    const landed = queueSurfaceTap(app, scene, Promise.resolve(hitToward(scene, dir)));
    if (wait) {
      await landed;
    }
    await new Promise((r) => setTimeout(r, 0));
  });
}

const errors = (host: HTMLElement) => [...host.querySelectorAll(".report.error")].map((e) => e.textContent);

describe("cancelling the geodesic map (#274)", () => {
  it("Cancel stops the query and keeps the sources, and Show map asks again", async () => {
    const { scene, store, app } = setup();
    const { host } = mount(app, store, scene);
    const asks = waiting(scene, "geodesicDistances");
    const runs = geodesicMapProbe.runs;
    expect(find(host, "geodesic-cancel")).toBeNull();
    await click(host, "geodesic-pick");
    await tap(app, scene, [0, 1, 0]);
    expect(asks).toHaveLength(1);
    expect(asks[0].signal.aborted).toBe(false);
    expect(find(host, "geodesic-show")).toBeNull();

    await click(host, "geodesic-cancel");
    expect(asks[0].signal.aborted).toBe(true);
    expect(host.textContent).not.toContain("Computing surface distances");
    expect(find(host, "geodesic-cancel")).toBeNull();
    expect(errors(host)).toEqual([]);
    expect(app.getSnapshot().surfacePath.sources).toHaveLength(1);
    expect(app.getSnapshot().surfaceFieldValues).toBeNull();
    expect(geodesicMapProbe.runs).toBe(runs);

    await click(host, "geodesic-show");
    expect(asks).toHaveLength(2);
    await asks[1].answer();
    expect(geodesicMapProbe.runs).toBe(runs + 1);
    expect(geodesicMapProbe.shown?.sources).toHaveLength(1);
    expect(find(host, "geodesic-show")).toBeNull();

    // A second source cancelled: the first's map stays, marked as one to ask for again.
    await tap(app, scene, [1, 0, 0]);
    expect(asks).toHaveLength(3);
    await click(host, "geodesic-cancel");
    expect(geodesicMapProbe.shown?.sources).toHaveLength(1);
    expect(app.getSnapshot().surfaceFieldValues).not.toBeNull();
    await click(host, "geodesic-show");
    expect(asks).toHaveLength(4);
    await asks[3].answer();
    expect(geodesicMapProbe.shown?.sources).toHaveLength(2);
    expect(find(host, "geodesic-show")).toBeNull();
  });

  it("a newer ask cancels the query under way instead of waiting for it", async () => {
    const { scene, store, app } = setup();
    const { host } = mount(app, store, scene);
    const asks = waiting(scene, "geodesicDistances");
    await click(host, "geodesic-pick");
    await tap(app, scene, [0, 1, 0]);
    await tap(app, scene, [1, 0, 0]);
    expect(asks.map((a) => a.signal.aborted)).toEqual([true, false]);
    expect((asks[1].args[1] as number[]).length).toBe(2);
    expect(host.textContent).toContain("Computing surface distances");
    // The method too.
    await act(async () => {
      host.querySelector<HTMLButtonElement>('[data-geodesic-method="heatMethod"]')?.click();
      await new Promise((r) => setTimeout(r, 0));
    });
    expect(asks.map((a) => a.signal.aborted)).toEqual([true, true, false]);
    await asks[2].answer();
    expect(geodesicMapProbe.shown).toMatchObject({ method: "heatMethod" });
    expect(errors(host)).toEqual([]);

    // Clearing the sources ends the query and the map.
    await tap(app, scene, [0, 0, 1]);
    await click(host, "geodesic-clear");
    expect(asks.map((a) => a.signal.aborted)).toEqual([true, true, false, true]);
    expect(app.getSnapshot().surfaceFieldValues).toBeNull();
    expect(host.textContent).not.toContain("Computing surface distances");
  });

  it("frees what the accurate method keeps when the panel goes", async () => {
    const { scene, store, app } = setup();
    const { root } = mount(app, store, scene);
    const release = vi.spyOn(scene.inspect, "releaseGeodesics");
    act(() => root.unmount());
    expect(release).toHaveBeenCalledTimes(1);
  });
});

describe("cancelling a path leg (#274)", () => {
  it("Cancel stops the leg and keeps the path; the next tap goes on from it", async () => {
    const { scene, store, app } = setup();
    const { host } = mount(app, store, scene);
    const asks = waiting(scene, "shortestSurfacePath");
    const path = () => app.getSnapshot().surfacePath.path;
    await click(host, "path-pick");
    await tap(app, scene, [0, 1, 0]);
    expect(find(host, "path-cancel")).toBeNull();
    await tap(app, scene, [1, 1, 0], false);
    expect(asks).toHaveLength(1);
    await click(host, "path-cancel");
    await act(() => surfacePathIdle(app));
    expect(asks[0].signal.aborted).toBe(true);
    expect(path()?.points).toHaveLength(1);
    expect(host.textContent).not.toContain("Tracing the path");
    expect(errors(host)).toEqual([]);

    await tap(app, scene, [1, 0, 0], false);
    expect(asks).toHaveLength(2);
    await asks[1].answer();
    await act(() => surfacePathIdle(app));
    expect(path()?.points).toHaveLength(2);
    expect(path()?.legs).toHaveLength(1);
    cancelSurfaceLeg(app); // nothing to cancel
    expect(path()?.points).toHaveLength(2);
  });

  it("undoing a point and clearing the path end the leg being traced", async () => {
    const { scene, store, app } = setup();
    const { host } = mount(app, store, scene);
    const asks = waiting(scene, "shortestSurfacePath");
    await click(host, "path-pick");
    for (const [button, left] of [
      ["path-undo", null],
      ["path-clear", null],
    ] as const) {
      await tap(app, scene, [0, 1, 0]);
      await tap(app, scene, [1, 0, 0], false);
      await click(host, button);
      await act(() => surfacePathIdle(app));
      expect(asks[asks.length - 1].signal.aborted).toBe(true);
      expect(app.getSnapshot().surfacePath.path).toBe(left);
      expect(app.getSnapshot().surfacePath.tracing).toBe(0);
      expect(errors(host)).toEqual([]);
    }
  });
});

/** The SDK's Node worker-thread stand-in for the browser engine Worker. */
class NodeWorker implements WorkerLike {
  readonly worker = new NodeThread(
    [
      resolve(process.cwd(), "../../packages/fast-mesh-wasm/test/harness/node-worker.mjs"),
      resolve(process.cwd(), "packages/fast-mesh-wasm/test/harness/node-worker.mjs"),
    ].find((p) => existsSync(p)) ?? "",
    { workerData: { serve: true } },
  );
  readonly #message: ((e: { data: unknown }) => void)[] = [];
  readonly #error: ((e: unknown) => void)[] = [];

  constructor() {
    this.worker.on("message", (data: unknown) => {
      for (const cb of this.#message) {
        cb({ data });
      }
    });
    this.worker.on("error", (e) => {
      for (const cb of this.#error) {
        cb(e);
      }
    });
  }

  postMessage(msg: unknown, transfer?: Transferable[]): void {
    this.worker.postMessage(msg, transfer as never);
  }

  addEventListener(type: "message" | "error", cb: (e: never) => void): void {
    (type === "message" ? this.#message : this.#error).push(cb as (e: unknown) => void);
  }

  terminate(): void {
    void this.worker.terminate();
  }
}

describe("on a Worker scene (#274)", () => {
  it("a map and a leg reject when their signal aborts, and the next ones land", async () => {
    const scene = await createWorkerScene({ createWorker: () => new NodeWorker() });
    cleanups.push(() => scene.terminate());
    const d = blobbySphere([0, 0, 0], 1, 256, 128, 0);
    const mesh: MeshHandle = scene.addMesh(d.positions, d.indices);
    await scene.measure(mesh);
    const like = scene as unknown as SceneLike;
    const request = { mesh, sources: [0], method: "heatMethod" } as const;
    for (const after of [0, 5]) {
      const c = new AbortController();
      const map = runGeodesicMap(like, request, c.signal);
      const leg = traceLeg(
        like,
        mesh,
        { slot: 0, point: [0, 1, 0] },
        { slot: 500, point: [1, 0, 0] },
        "heatMethod",
        c.signal,
      );
      const outcome = Promise.allSettled([map, leg]);
      await new Promise((r) => setTimeout(r, after));
      c.abort();
      expect((await outcome).map((o) => (o.status === "rejected" ? (o.reason as Error).name : "landed"))).toEqual([
        "AbortError",
        "AbortError",
      ]);
    }
    const report = await runGeodesicMap(like, request, new AbortController().signal);
    // Pole to pole, nearly half the circumference.
    expect(report.max).toBeGreaterThan(3.08);
    expect(report.max).toBeLessThan(3.2);
  });
});
