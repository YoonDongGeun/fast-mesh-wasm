// The Worker's artifact selection: the threaded engine on cross-origin
// isolated pages (with its rayon pool started), the single-threaded one with
// `threads: 1`, on other pages and after any threaded start-up failure. Unit
// tests of `loadEngine` with fake loaders, then `WorkerScene`s over the Node
// harness, which can pose as an isolated page with a fake or failing threaded
// glue (Node itself is not isolated: the plain harness is single-threaded).
import { existsSync } from "node:fs";
import { Worker } from "node:worker_threads";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createWorkerScene, type WorkerLike, type WorkerScene } from "../src/worker/index.js";
import { defaultThreads, type Glue, loadEngine, type ThreadedGlue } from "../src/worker/load.js";

const hasThreaded = existsSync(new URL("../dist/wasm-threads/fast_mesh_wasm.js", import.meta.url));

/** A fake glue: records what was called, in order. */
function fakeGlue(log: string[], name: string, fail: { init?: unknown; pool?: unknown; hang?: boolean } = {}) {
  const memory = new WebAssembly.Memory({ initial: 1 });
  class WasmScene {}
  return {
    memory,
    glue: {
      default: async () => {
        log.push(`${name}.init`);
        if (fail.init !== undefined) {
          throw fail.init;
        }
        return { memory };
      },
      WasmScene,
      initThreadPool: (n: number) => {
        log.push(`${name}.pool(${n})`);
        if (fail.hang) {
          return new Promise<never>(() => {});
        }
        return fail.pool !== undefined ? Promise.reject(fail.pool) : Promise.resolve();
      },
    } as unknown as ThreadedGlue,
  };
}

function loaders(log: string[], fail: { import?: unknown; init?: unknown; pool?: unknown; hang?: boolean } = {}) {
  const threaded = fakeGlue(log, "threaded", fail);
  const single = fakeGlue(log, "single");
  return {
    threaded,
    single,
    importThreaded: async () => {
      log.push("importThreaded");
      if (fail.import !== undefined) {
        throw fail.import;
      }
      return threaded.glue;
    },
    importSingle: async () => {
      log.push("importSingle");
      return single.glue as Glue;
    },
  };
}

describe("loadEngine", () => {
  it("isolated_loads_threaded", async () => {
    const log: string[] = [];
    const l = loaders(log);
    const warn = vi.fn();
    const got = await loadEngine({ threads: 4, isolated: true, ...l, warn });
    expect(got.threads).toBe(4);
    expect(got.wasm).toBe(l.threaded.glue);
    expect(got.memory).toBe(l.threaded.memory);
    expect(log).toEqual(["importThreaded", "threaded.init", "threaded.pool(4)"]);
    expect(warn).not.toHaveBeenCalled();
  });

  it("single_thread_never_loads_threaded", async () => {
    for (const [threads, isolated] of [
      [1, true],
      [4, false],
      [1, false],
    ] as const) {
      const log: string[] = [];
      const l = loaders(log);
      const got = await loadEngine({ threads, isolated, ...l, warn: () => {} });
      expect(got.threads).toBe(1);
      expect(got.wasm).toBe(l.single.glue);
      expect(got.memory).toBe(l.single.memory);
      expect(log).toEqual(["importSingle", "single.init"]);
    }
  });

  it("threaded_start_failure_falls_back", async () => {
    const cases = {
      import: { import: new TypeError("Failed to fetch dynamically imported module") },
      // A 4 GiB shared memory can fail to allocate (32-bit or constrained browsers).
      instantiate: { init: new RangeError("WebAssembly.Memory(): could not allocate memory") },
      pool: { pool: new Error("nested Worker failed") },
      hang: { hang: true },
    };
    for (const [name, fail] of Object.entries(cases)) {
      const log: string[] = [];
      const l = loaders(log, fail);
      const warn = vi.fn();
      const got = await loadEngine({ threads: 4, isolated: true, ...l, warn, poolTimeoutMs: 50 });
      expect(got.threads, name).toBe(1);
      expect(got.wasm, name).toBe(l.single.glue);
      expect(got.memory, name).toBe(l.single.memory);
      expect(log.slice(-2), name).toEqual(["importSingle", "single.init"]);
      expect(warn, name).toHaveBeenCalledTimes(1);
      expect(warn.mock.calls[0][0], name).toMatch(/single-threaded/);
      expect(warn.mock.calls[0][0], name).toMatch(/Multithreading \(COOP\/COEP\)/);
      // Without `warn` (not a development build) it falls back silently.
      const quiet = loaders([], fail);
      expect((await loadEngine({ threads: 4, isolated: true, ...quiet, poolTimeoutMs: 50 })).threads, name).toBe(1);
    }
  });

  it("non_isolated_warns_once_in_dev", async () => {
    const warn = vi.fn();
    await loadEngine({ threads: 4, isolated: false, ...loaders([]), warn });
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0][0]).toMatch(/cross-origin isolated/);
    expect(warn.mock.calls[0][0]).toMatch(/Multithreading \(COOP\/COEP\)/);
    // `threads: 1` asked for one thread: nothing to warn about.
    const none = vi.fn();
    await loadEngine({ threads: 1, isolated: false, ...loaders([]), warn: none });
    expect(none).not.toHaveBeenCalled();
  });

  it("default_threads_capped", () => {
    expect(defaultThreads({ hardwareConcurrency: 20 })).toBe(8);
    expect(defaultThreads({ hardwareConcurrency: 2 })).toBe(2);
    expect(defaultThreads({ hardwareConcurrency: 1 })).toBe(1);
    expect(defaultThreads({})).toBe(4);
    expect(defaultThreads(undefined)).toBe(4);
  });
});

/** Harness options for posing as a page (see test/harness/node-worker.mjs). */
interface Pose {
  /** `crossOriginIsolated` as a browser page would report it. */
  isolated?: boolean;
  /** The threaded glue: a fake that starts, or a real/failing start-up. */
  threaded?: "fake" | "real" | "memFail" | "importFail";
}

/** The harness Worker as a `WorkerLike`, keeping its own `pool` reports apart. */
class NodeWorker implements WorkerLike {
  readonly worker: Worker;
  readonly sent: unknown[] = [];
  readonly pools: number[] = [];
  terminated = false;
  readonly #message: ((e: { data: unknown }) => void)[] = [];

  constructor(pose: Pose) {
    this.worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), {
      workerData: { serve: true, ...pose },
    });
    this.worker.on("message", (data: { kind?: string; threads?: number }) => {
      if (data.kind === "pool") {
        this.pools.push(data.threads as number);
        return;
      }
      for (const cb of this.#message) {
        cb({ data });
      }
    });
  }

  postMessage(msg: unknown, transfer?: Transferable[]): void {
    this.sent.push(msg);
    this.worker.postMessage(msg, transfer as never);
  }

  addEventListener(type: "message" | "error" | "messageerror", cb: (e: any) => void): void {
    if (type === "message") {
      this.#message.push(cb);
    }
  }

  terminate(): void {
    this.terminated = true;
    void this.worker.terminate();
  }
}

const opened: WorkerScene[] = [];
afterEach(() => {
  for (const s of opened.splice(0)) {
    s.terminate();
  }
  vi.restoreAllMocks();
});

async function start(pose: Pose = {}, options: { threads?: number; debug?: boolean } = {}) {
  const workers: NodeWorker[] = [];
  const scene = await createWorkerScene({
    ...options,
    createWorker: () => {
      const w = new NodeWorker(pose);
      workers.push(w);
      return w;
    },
  });
  opened.push(scene);
  return { scene, workers };
}

/** A unit tetrahedron, translated. */
function tetra(dx: number) {
  return {
    positions: new Float64Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1].map((v, i) => (i % 3 === 0 ? v + dx : v))),
    indices: new Uint32Array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]),
  };
}

/** Every kind of call still works on the scene. */
async function works(scene: WorkerScene): Promise<void> {
  const a = tetra(0);
  const b = tetra(0.5);
  const ma = scene.addMesh(a.positions, a.indices);
  const mb = scene.addMesh(b.positions, b.indices);
  expect(await scene.intersects(ma, mb)).toBe(true);
  const hit = await scene.closestPoint([-1, 0.1, 0.1]);
  expect(hit?.distance).toBeCloseTo(1, 6);
}

describe("WorkerScene threads", () => {
  it("threads_option_reported", async () => {
    // Node is not cross-origin isolated: single-threaded, whatever was asked.
    const { scene, workers } = await start({}, { threads: 4 });
    expect(scene.threads).toBe(1);
    expect(workers[0].sent[0]).toEqual({ kind: "init", threads: 4, debug: expect.any(Boolean) });
    expect(workers[0].pools).toEqual([]);
    await works(scene);
  });

  it("threads_option_validated", async () => {
    for (const threads of [0, -1, 1.5, Number.NaN]) {
      await expect(createWorkerScene({ threads, createWorker: () => new NodeWorker({}) })).rejects.toMatchObject({
        code: "INVALID_ARGUMENT",
      });
    }
  });

  it("isolated_harness_starts_pool", async () => {
    const { scene, workers } = await start({ isolated: true, threaded: "fake" }, { threads: 4 });
    expect(scene.threads).toBe(4);
    expect(workers[0].pools).toEqual([4]);
    await works(scene);
  });

  it("threads_one_forces_single", async () => {
    const { scene, workers } = await start({ isolated: true, threaded: "fake" }, { threads: 1 });
    expect(scene.threads).toBe(1);
    expect(workers[0].pools).toEqual([]);
  });

  it("restart_reinitialises_pool", async () => {
    const { scene, workers } = await start({ isolated: true, threaded: "fake" }, { threads: 3 });
    expect(scene.threads).toBe(3);
    const a = tetra(0);
    const mesh = scene.addMesh(a.positions, a.indices);
    await scene.restart();
    expect(workers).toHaveLength(2);
    expect(workers[0].terminated).toBe(true);
    // The new Worker got its own init and started its own pool, once.
    expect(workers[1].sent[0]).toMatchObject({ kind: "init", threads: 3 });
    expect(workers[1].pools).toEqual([3]);
    expect(scene.threads).toBe(3);
    expect(await scene.intersects(mesh, mesh)).toBe(true);
  });

  it("non_isolated_warns_once_in_dev", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const { scene } = await start({ isolated: false }, { threads: 4, debug: true });
    expect(scene.threads).toBe(1);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][0])).toMatch(/Multithreading \(COOP\/COEP\)/);
    // Once per scene: a restart's Worker warns no more.
    await scene.restart();
    expect(warn).toHaveBeenCalledTimes(1);
  });

  it("threaded_start_failure_falls_back", async () => {
    // An import that fails (as with no dist/wasm-threads). With the threaded
    // artifact built, the real threaded glue: once it instantiates but its
    // pool cannot start (no nested `Worker` in Node, as on an isolated page
    // that blocks them), once its shared memory fails to allocate (`new
    // WebAssembly.Memory` throws a RangeError).
    const causes = { importFail: /TypeError/, real: /Worker is not defined/, memFail: /RangeError/ };
    const poses = hasThreaded ? (["importFail", "real", "memFail"] as const) : (["importFail"] as const);
    for (const threaded of poses) {
      const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
      const { scene, workers } = await start({ isolated: true, threaded }, { threads: 4, debug: true });
      expect(scene.threads, threaded).toBe(1);
      expect(warn, threaded).toHaveBeenCalledTimes(1);
      expect(String(warn.mock.calls[0][0]), threaded).toMatch(/single-threaded/);
      expect(String(warn.mock.calls[0][0]), threaded).toMatch(causes[threaded]);
      expect(workers[0].pools, threaded).toEqual([]);
      await works(scene);
      scene.terminate();
      warn.mockRestore();
    }
  });
});
