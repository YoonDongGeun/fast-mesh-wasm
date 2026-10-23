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
import type { InitMessage, Request } from "../src/worker/protocol.js";
import {
  defaultThreads,
  type Glue,
  loadEngine,
  PoolStartError,
  startEngine,
  type ThreadedGlue,
} from "../src/worker/load.js";

const hasThreaded = existsSync(new URL("../dist/wasm-threads/fast_mesh_wasm.js", import.meta.url));

/** A fake glue: records what was called, in order. */
function fakeGlue(
  log: string[],
  name: string,
  fail: { init?: unknown; pool?: unknown; hang?: boolean; watch?: unknown } = {},
) {
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
      watchPoolCrashes: (channel: string) => {
        log.push(`${name}.watch(${channel})`);
        if (fail.watch !== undefined) {
          throw fail.watch;
        }
      },
    } as unknown as ThreadedGlue,
  };
}

function loaders(
  log: string[],
  fail: { import?: unknown; init?: unknown; pool?: unknown; hang?: boolean; watch?: unknown } = {},
) {
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

  it("threaded_watches_pool_crashes", async () => {
    // The scene's channel goes to the pool once it runs: its threads report
    // a crash there (the engine Worker itself is then blocked).
    const log: string[] = [];
    const l = loaders(log);
    const got = await loadEngine({ threads: 4, isolated: true, ...l, crashChannel: "ch" });
    expect(got.threads).toBe(4);
    expect(log).toEqual(["importThreaded", "threaded.init", "threaded.pool(4)", "threaded.watch(ch)"]);
    // Single-threaded, there is no pool to watch.
    const single: string[] = [];
    await loadEngine({ threads: 1, isolated: true, ...loaders(single), crashChannel: "ch" });
    expect(single).toEqual(["importSingle", "single.init"]);
    // A watch that cannot start leaves a working threaded engine (unwatched),
    // with a development warning.
    const failed: string[] = [];
    const warn = vi.fn();
    const unwatched = await loadEngine({
      threads: 2,
      isolated: true,
      ...loaders(failed, { watch: new Error("no BroadcastChannel") }),
      crashChannel: "ch",
      warn,
    });
    expect(unwatched.threads).toBe(2);
    expect(failed.at(-1)).toBe("threaded.watch(ch)");
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0][0]).toMatch(/crash detection unavailable \(Error: no BroadcastChannel\)/);
    // Not in production (no `warn`): silent.
    const quiet = await loadEngine({
      threads: 2,
      isolated: true,
      ...loaders([], { watch: new Error("no BroadcastChannel") }),
      crashChannel: "ch",
    });
    expect(quiet.threads).toBe(2);
    // startEngine passes the init message's channel.
    const started: string[] = [];
    await startEngine({ kind: "init", threads: 3, debug: false, crashChannel: "x" }, loaders(started), true);
    expect(started.at(-1)).toBe("threaded.watch(x)");
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
    // Nothing threaded is left behind: the Worker falls back itself.
    const cases = {
      import: { import: new TypeError("Failed to fetch dynamically imported module") },
      // A 4 GiB shared memory can fail to allocate (32-bit or constrained browsers).
      instantiate: { init: new RangeError("WebAssembly.Memory(): could not allocate memory") },
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

  it("pool_failure_rejects_for_replacement", async () => {
    // The threaded instance and nested Workers are live: the Worker must go.
    for (const [name, fail] of Object.entries({ pool: { pool: new Error("nested Worker failed") }, hang: { hang: true } })) {
      const log: string[] = [];
      const warn = vi.fn();
      const got = loadEngine({ threads: 4, isolated: true, ...loaders(log, fail), warn, poolTimeoutMs: 50 });
      await expect(got, name).rejects.toBeInstanceOf(PoolStartError);
      await expect(got, name).rejects.toThrow(/Multithreading \(COOP\/COEP\)/);
      expect(log, name).toEqual(["importThreaded", "threaded.init", "threaded.pool(4)"]);
      expect(warn, name).not.toHaveBeenCalled();
    }
  });

  it("broken_shared_growth_skips_threaded", async () => {
    // The probe saw the browser trap on a memory another thread grew (WebKit
    // 26): the threaded files are never fetched and nothing threaded is left
    // behind, so this Worker runs single-threaded itself, without waiting for
    // a pool that would never start.
    const log: string[] = [];
    const l = loaders(log);
    const warn = vi.fn();
    const probe = vi.fn(async () => "fill traps after another thread grew the memory");
    const got = await loadEngine({ threads: 4, isolated: true, ...l, warn, probe });
    expect(probe).toHaveBeenCalledTimes(1);
    expect(got.threads).toBe(1);
    expect(got.wasm).toBe(l.single.glue);
    expect(log).toEqual(["importSingle", "single.init"]);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(warn.mock.calls[0][0]).toMatch(/fill traps after another thread grew the memory/);
    expect(warn.mock.calls[0][0]).toMatch(/single-threaded/);
    expect(warn.mock.calls[0][0]).toMatch(/Multithreading \(COOP\/COEP\)/);
    // Silent without `warn`.
    expect((await loadEngine({ threads: 4, isolated: true, ...loaders([]), probe })).threads).toBe(1);
  });

  it("inconclusive_probe_tries_threaded", async () => {
    const probes: Record<string, () => Promise<string | undefined>> = {
      healthy: async () => undefined,
      failed: async () => {
        throw new Error("probe crashed");
      },
    };
    for (const [name, probe] of Object.entries(probes)) {
      const log: string[] = [];
      const warn = vi.fn();
      const got = await loadEngine({ threads: 4, isolated: true, ...loaders(log), warn, probe });
      expect(got.threads, name).toBe(4);
      expect(log, name).toEqual(["importThreaded", "threaded.init", "threaded.pool(4)"]);
      expect(warn, name).not.toHaveBeenCalled();
    }
  });

  it("probe_only_before_threaded", async () => {
    // Not run when the threaded engine is not tried at all.
    const probe = vi.fn(async () => "broken");
    await loadEngine({ threads: 1, isolated: true, ...loaders([]), probe });
    await loadEngine({ threads: 4, isolated: false, ...loaders([]), probe });
    expect(probe).not.toHaveBeenCalled();
    // startEngine passes an injected probe on.
    const log: string[] = [];
    const got = await startEngine({ kind: "init", threads: 4, debug: true }, { ...loaders(log), probe }, true);
    expect(got.threads).toBe(1);
    expect(got.warning).toMatch(/broken/);
    expect(log).toEqual(["importSingle", "single.init"]);
  });

  it("start_engine_warnings", async () => {
    // The production path: no `debug`, not isolated, so single-threaded and
    // silent.
    expect(await startEngine({ kind: "init", threads: 4, debug: false }, loaders([]), false)).toEqual({ threads: 1 });
    // Development on a page: the reason is reported.
    const dev = await startEngine({ kind: "init", threads: 4, debug: true }, loaders([]), false);
    expect(dev.warning).toMatch(/cross-origin isolated/);
    // Not a page (no `crossOriginIsolated`, as in Node): nothing to fix.
    expect(await startEngine({ kind: "init", threads: 4, debug: true }, loaders([]), undefined)).toEqual({ threads: 1 });
    // A pool failure carries its warning in development only.
    const pool = { pool: new Error("nested Worker failed") };
    const quiet = await startEngine({ kind: "init", threads: 4, debug: false }, loaders([], pool), true).catch((e) => e);
    expect(quiet).toBeInstanceOf(PoolStartError);
    expect(quiet.warning).toBeUndefined();
    const loud = await startEngine({ kind: "init", threads: 4, debug: true }, loaders([], pool), true).catch((e) => e);
    expect(loud.warning).toMatch(/single-threaded/);
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
  threaded?: "fake" | "poolFail" | "real" | "memFail" | "importFail";
  /** `analyze` posts a pool thread's crash report, then blocks the engine. */
  poolCrash?: boolean;
}

/** The harness Worker as a `WorkerLike`, keeping its own `pool` reports apart. */
class NodeWorker implements WorkerLike {
  readonly worker: Worker;
  readonly sent: unknown[] = [];
  readonly pools: number[] = [];
  /** Channels the fake threaded glue was asked to watch for pool crashes. */
  readonly watched: string[] = [];
  terminated = false;
  /** While set, `fallback` messages are kept in `fallbacks` until `release()`. */
  holdFallbacks = false;
  readonly fallbacks: { kind: "fallback"; batches: unknown[] }[] = [];
  readonly #message: ((e: { data: unknown }) => void)[] = [];
  readonly #error: ((e: unknown) => void)[] = [];

  constructor(pose: Pose) {
    this.worker = new Worker(new URL("./harness/node-worker.mjs", import.meta.url), {
      workerData: { serve: true, ...pose },
    });
    this.worker.on("message", (data: { kind?: string; threads?: number; channel?: string }) => {
      if (data.kind === "pool") {
        this.pools.push(data.threads as number);
        return;
      }
      if (data.kind === "watch") {
        this.watched.push(data.channel as string);
        return;
      }
      if (data.kind === "fallback" && this.holdFallbacks) {
        this.fallbacks.push(data as never);
        return;
      }
      this.#deliver(data);
    });
  }

  #deliver(data: unknown): void {
    for (const cb of this.#message) {
      cb({ data });
    }
  }

  /** Delivers the first `n` held `fallback` messages. */
  release(n = this.fallbacks.length): void {
    for (const m of this.fallbacks.splice(0, n)) {
      this.#deliver(m);
    }
  }

  /** Batches given back in held `fallback` messages. */
  givenBack(): number {
    return this.fallbacks.reduce((n, m) => n + m.batches.length, 0);
  }

  postMessage(msg: unknown, transfer?: Transferable[]): void {
    this.sent.push(msg);
    this.worker.postMessage(msg, transfer as never);
  }

  addEventListener(type: "message" | "error" | "messageerror", cb: (e: any) => void): void {
    if (type === "message") {
      this.#message.push(cb);
    } else if (type === "error") {
      this.#error.push(cb);
    }
  }

  /** Fires `error`, as a Worker that crashes does. */
  fail(): void {
    for (const cb of this.#error) {
      cb({ message: "boom" });
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
  vi.unstubAllGlobals();
});

/** A scene over harness Workers; `pose` may differ per Worker (by index). */
async function start(pose: Pose | ((n: number) => Pose) = {}, options: { threads?: number; debug?: boolean } = {}) {
  const workers: NodeWorker[] = [];
  const scene = await createWorkerScene({
    ...options,
    createWorker: () => {
      const w = new NodeWorker(typeof pose === "function" ? pose(workers.length) : pose);
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
    for (const threads of [1.5, Number.NaN, Number.POSITIVE_INFINITY]) {
      await expect(createWorkerScene({ threads, createWorker: () => new NodeWorker({}) })).rejects.toMatchObject({
        code: "INVALID_ARGUMENT",
      });
    }
  });

  it("threads_option_clamped", async () => {
    for (const [threads, sent] of [
      [0, 1],
      [-3, 1],
      [64, 64],
      [1000, 64],
    ]) {
      const { scene, workers } = await start({ isolated: true, threaded: "fake" }, { threads });
      expect(workers[0].sent[0], String(threads)).toMatchObject({ kind: "init", threads: sent });
      expect(scene.threads, String(threads)).toBe(sent);
      scene.terminate();
    }
  });

  it("isolated_harness_starts_pool", async () => {
    const { scene, workers } = await start({ isolated: true, threaded: "fake" }, { threads: 4 });
    expect(scene.threads).toBe(4);
    expect(workers[0].pools).toEqual([4]);
    await works(scene);
  });

  it("threads_one_forces_single", async () => {
    vi.stubGlobal("crossOriginIsolated", true);
    const { scene, workers } = await start({ isolated: true, threaded: "fake" }, { threads: 1 });
    expect(scene.threads).toBe(1);
    expect(workers[0].pools).toEqual([]);
    // No pool, so no crash channel.
    expect(workers[0].sent[0]).toEqual({ kind: "init", threads: 1, debug: expect.any(Boolean) });
  });

  it("pool_crash_poisons", async () => {
    // A pool thread crashes: the engine Worker blocks and never answers; the
    // crash report on the scene's channel poisons the scene, as a Worker
    // error does, and a restart recovers with a new pool.
    vi.stubGlobal("crossOriginIsolated", true);
    const { scene, workers } = await start({ isolated: true, threaded: "fake", poolCrash: true }, { threads: 4 });
    expect(scene.threads).toBe(4);
    const channel = (workers[0].sent[0] as InitMessage).crashChannel;
    expect(channel).toEqual(expect.any(String));
    expect(workers[0].watched).toEqual([channel]);
    const a = tetra(0);
    const mesh = scene.addMesh(a.positions, a.indices);
    expect(await scene.intersects(mesh, mesh)).toBe(true);
    const t0 = performance.now();
    const crashed = scene.analyze(mesh);
    const after = scene.intersects(mesh, mesh);
    await expect(crashed).rejects.toMatchObject({
      code: "INSTANCE_POISONED",
      message: expect.stringMatching(/pool.*test pool panic/),
    });
    await expect(after).rejects.toMatchObject({ code: "INSTANCE_POISONED" });
    expect(performance.now() - t0).toBeLessThan(2000);
    expect(workers[0].terminated).toBe(true);
    await expect(scene.intersects(mesh, mesh)).rejects.toMatchObject({ code: "INSTANCE_POISONED" });
    await scene.restart();
    expect(workers).toHaveLength(2);
    const next = (workers[1].sent[0] as InitMessage).crashChannel;
    expect(next).toEqual(expect.any(String));
    expect(next).not.toBe(channel);
    expect(workers[1].watched).toEqual([next]);
    expect(scene.threads).toBe(4);
    expect(await scene.intersects(mesh, mesh)).toBe(true);
  });

  it("old_crash_channel_ignored_after_restart", async () => {
    // A report on the retired Worker's channel (a late one from its pool)
    // must not poison the scene the restart brought back.
    vi.stubGlobal("crossOriginIsolated", true);
    const { scene, workers } = await start({ isolated: true, threaded: "fake" }, { threads: 2 });
    const old = (workers[0].sent[0] as InitMessage).crashChannel as string;
    const a = tetra(0);
    const mesh = scene.addMesh(a.positions, a.indices);
    await scene.restart();
    const sender = new BroadcastChannel(old);
    sender.postMessage({ kind: "poolCrash", detail: "late report (test)" });
    sender.close();
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(workers[1].terminated).toBe(false);
    expect(await scene.intersects(mesh, mesh)).toBe(true);
    // The current channel still poisons.
    const current = new BroadcastChannel((workers[1].sent[0] as InitMessage).crashChannel as string);
    current.postMessage({ kind: "poolCrash", detail: "current report (test)" });
    current.close();
    await vi.waitFor(() => expect(workers[1].terminated).toBe(true));
    await expect(scene.intersects(mesh, mesh)).rejects.toMatchObject({ code: "INSTANCE_POISONED" });
  });

  it("pool_failure_watches_nothing", async () => {
    // The pool failed and the Worker was replaced by a single-threaded one:
    // its init has no crash channel, and the scene works.
    vi.stubGlobal("crossOriginIsolated", true);
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const { scene, workers } = await start({ isolated: true, threaded: "poolFail" }, { threads: 4 });
    expect(scene.threads).toBe(1);
    expect(workers[1].sent[0]).toEqual({ kind: "init", threads: 1, debug: expect.any(Boolean) });
    await works(scene);
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

  it("pool_failure_replaces_worker", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const { scene, workers } = await start({ isolated: true, threaded: "poolFail" }, { threads: 4, debug: true });
    // The Worker with the half-started pool is gone; a single-threaded one runs.
    expect(workers).toHaveLength(2);
    expect(workers[0].terminated).toBe(true);
    expect(workers[0].sent).toEqual([{ kind: "init", threads: 4, debug: true }]);
    expect(workers[1].sent[0]).toEqual({ kind: "init", threads: 1, debug: true });
    expect(scene.threads).toBe(1);
    expect(warn).toHaveBeenCalledTimes(1);
    expect(String(warn.mock.calls[0][0])).toMatch(/nested Worker failed/);
    await works(scene);
    // Later restarts stay single-threaded: no new wait for a pool.
    await scene.restart();
    expect(workers).toHaveLength(3);
    expect(workers[2].sent[0]).toMatchObject({ kind: "init", threads: 1 });
    expect(scene.threads).toBe(1);
    expect(warn).toHaveBeenCalledTimes(1);
  });

  it("pool_failure_on_restart_hands_batches_over", async () => {
    // The first Worker is threaded; the restart's pool fails, while batches
    // are sent to it, before and after it gives up.
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const { scene, workers } = await start((n) => ({ isolated: true, threaded: n === 1 ? "poolFail" : "fake" }), {
      threads: 2,
    });
    expect(scene.threads).toBe(2);
    const a = tetra(0);
    const ma = scene.addMesh(a.positions, a.indices);
    expect(await scene.intersects(ma, ma)).toBe(true);
    const restarted = scene.restart();
    workers[1].holdFallbacks = true;
    // Sent to the second Worker: it gives the replay and these back.
    const b = tetra(0.5);
    const mb = scene.addMesh(b.positions, b.indices);
    const hit = scene.intersects(ma, mb);
    await vi.waitFor(() => expect(workers[1].givenBack()).toBe(2));
    // The fake pool fails within the init's microtasks, before the replay's
    // message is handled: the first give-back is empty, the two batches
    // follow one message each.
    expect(workers[1].fallbacks.map((m) => m.batches.length)).toEqual([0, 1, 1]);
    // The first give-back arrives: a third Worker starts; batches sent while
    // the rest is outstanding are held, so the order holds.
    workers[1].release(1);
    expect(workers).toHaveLength(3);
    const c = tetra(3);
    const mc = scene.addMesh(c.positions, c.indices);
    const apart = scene.intersects(ma, mc);
    // Let the batch be flushed (a microtask): it is held, not posted.
    await new Promise((resolve) => setTimeout(resolve, 0));
    // Only the init so far (lengths only: posted batches hold detached buffers).
    expect(workers[2].sent.length).toBe(1);
    workers[1].release();
    // The given-back batches first (the replay, then `b`'s), the held one last.
    const adds = (i: number) =>
      (workers[2].sent[i] as Request[]).flatMap((r) => (r.kind === "addMesh" ? [r.id] : []));
    expect(workers[2].sent).toHaveLength(4);
    expect(adds(1)).toEqual([scene.idOf(ma)]);
    expect(adds(2)).toEqual([scene.idOf(mb)]);
    expect(adds(3)).toEqual([scene.idOf(mc)]);
    await restarted;
    expect(await hit).toBe(true);
    expect(await apart).toBe(false);
    expect(workers[1].terminated).toBe(true);
    expect(workers[2].sent[0]).toEqual({ kind: "init", threads: 1, debug: expect.any(Boolean) });
    expect(workers[2].sent.length).toBe(4);
    expect(scene.threads).toBe(1);
    expect(await scene.intersects(mb, mc)).toBe(false);
  });

  it("handed_off_worker_error_poisons", async () => {
    // The restart's Worker fails its pool and crashes while it still owes
    // batches: the held ones must not wait forever.
    vi.spyOn(console, "warn").mockImplementation(() => {});
    const { scene, workers } = await start((n) => ({ isolated: true, threaded: n === 1 ? "poolFail" : "fake" }), {
      threads: 2,
    });
    const a = tetra(0);
    const ma = scene.addMesh(a.positions, a.indices);
    expect(await scene.intersects(ma, ma)).toBe(true);
    const restarted = scene.restart();
    workers[1].holdFallbacks = true;
    const b = tetra(0.5);
    const mb = scene.addMesh(b.positions, b.indices);
    const hit = scene.intersects(ma, mb);
    await vi.waitFor(() => expect(workers[1].givenBack()).toBe(2));
    expect(workers[1].fallbacks.map((m) => m.batches.length)).toEqual([0, 1, 1]);
    // The handoff starts and owes two batches; a query is held behind them.
    workers[1].release(1);
    expect(workers).toHaveLength(3);
    const held = scene.intersects(ma, ma);
    await new Promise((resolve) => setTimeout(resolve, 0));
    workers[1].fail();
    await expect(restarted).rejects.toMatchObject({ code: "INSTANCE_POISONED" });
    await expect(hit).rejects.toMatchObject({ code: "INSTANCE_POISONED" });
    await expect(held).rejects.toMatchObject({ code: "INSTANCE_POISONED" });
    expect(workers[1].terminated).toBe(true);
    expect(workers[2].terminated).toBe(true);
    // A restart recovers, single-threaded.
    await scene.restart();
    expect(workers).toHaveLength(4);
    expect(workers[3].sent[0]).toMatchObject({ kind: "init", threads: 1 });
    expect(await scene.intersects(ma, ma)).toBe(true);
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
      // The real glue's pool failed after it instantiated: that Worker was
      // replaced. The others fell back in their Worker.
      expect(workers, threaded).toHaveLength(threaded === "real" ? 2 : 1);
      expect(workers[0].terminated, threaded).toBe(threaded === "real");
      await works(scene);
      scene.terminate();
      warn.mockRestore();
    }
  });
});
