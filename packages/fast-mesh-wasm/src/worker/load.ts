// Which WebAssembly artifact the Worker runs: the threaded one (dist/wasm-threads,
// with a rayon pool of Workers sharing its memory) on cross-origin isolated
// pages, else the single-threaded one (dist/wasm). A browser that cannot share
// a growing memory between threads (WebKit 26, see probe.ts) is detected while
// the threaded glue is imported, before it is instantiated, and gets the
// single-threaded engine right away (the ready message says so, and the scene
// asks later Workers for one thread). A failure importing or instantiating
// the threaded glue (say its 4 GiB-maximum shared memory cannot be allocated)
// falls back to the single-threaded engine in this Worker too. A pool that
// fails or never starts leaves a live threaded instance and nested Workers
// behind, so it rejects with a `PoolStartError` instead: the Worker reports
// it, and the scene replaces the Worker with a single-threaded one (see
// `serveEngine`). Once a pool runs, its threads are watched for crashes
// (`watchPoolCrashes`) when the scene gave a channel. The glue modules come
// from injected loaders, so this module imports neither: only the Worker
// entry names them, and single-threaded mode never fetches wasm-threads/.
import { setMemory } from "../core.js";
import { useWasmScene } from "../scene.js";
import type { WasmScene } from "../wasm/fast_mesh_wasm.js";
import { probeSharedGrowth } from "./probe.js";
import type { InitMessage } from "./protocol.js";

/** What the engine needs of a wasm-bindgen glue module. */
export interface Glue {
  /** The glue's async init (fetches and instantiates its .wasm). */
  default(): Promise<{ memory: WebAssembly.Memory }>;
  WasmScene: typeof WasmScene;
}

/** The threaded glue: also starts wasm-bindgen-rayon's pool. */
export interface ThreadedGlue extends Glue {
  initThreadPool(threads: number): Promise<unknown>;
  /**
   * Makes the pool's threads report a panic or a trap on the
   * `BroadcastChannel` named `channel` (a `PoolCrash` message). Call once,
   * after the pool started.
   */
  watchPoolCrashes(channel: string): void;
}

/** Options of {@link loadEngine}. */
export interface LoadOptions {
  /** The threads asked for: the threaded engine is tried unless 1. */
  threads: number;
  /** `crossOriginIsolated`: shared memory (and so threads) is available. */
  isolated: boolean;
  importThreaded: () => Promise<ThreadedGlue>;
  importSingle: () => Promise<Glue>;
  /**
   * Development only: called once when the engine is single-threaded though
   * more threads were asked for, or when its pool cannot be watched for crashes.
   */
  warn?: (message: string) => void;
  /** How long the pool may take to start: {@link POOL_TIMEOUT_MS} by default. */
  poolTimeoutMs?: number;
  /** The channel the pool's threads report a crash on (see {@link ThreadedGlue.watchPoolCrashes}). */
  crashChannel?: string;
  /**
   * Run before the threaded engine is tried: resolves with why it must not
   * run in this browser, or `undefined` when it may (see
   * {@link probeSharedGrowth}, {@link startEngine}'s default). A rejection
   * counts as `undefined`.
   */
  probe?: () => Promise<string | undefined>;
}

/**
 * How long `initThreadPool` may take before the threaded engine is given up.
 * Its nested Workers report ready by message; one that fails to load (blocked
 * by the page, or a bundler that broke its import) never does, and the pool
 * would wait forever.
 */
export const POOL_TIMEOUT_MS = 10_000;

const SEE = 'See "Multithreading (COOP/COEP)" in the fast-mesh-wasm README.';

/**
 * The threaded glue instantiated but its pool did not start (it rejected or
 * timed out). The Worker holds leftovers (the instance, nested Workers), so
 * it is replaced by a single-threaded one rather than falling back itself.
 * The message is the warning text; `warning` is set (by {@link startEngine})
 * in development only.
 */
export class PoolStartError extends Error {
  override name = "PoolStartError";
  warning?: string;
}

/** The default pool size: the logical cores, at most 8 (4 when unknown). */
export function defaultThreads(nav: { hardwareConcurrency?: number } | undefined): number {
  return Math.min(nav?.hardwareConcurrency ?? 4, 8);
}

/**
 * Loads and initialises the glue to run: the threaded one with its pool
 * started when `isolated` and `threads` is not 1, else (or when importing or
 * instantiating it fails) the single-threaded one. Returns the glue, its
 * memory and the threads the engine computes with (1 when single-threaded);
 * `probeFailed` is set when the probe ruled the threaded engine out. The
 * probe runs while the threaded glue is imported; only its instantiation
 * waits for the probe's answer.
 * Rejects with a {@link PoolStartError} when the pool fails or times out
 * after the threaded glue instantiated (`warn` is then not called). A
 * started pool is watched for crashes on `crashChannel`, when given.
 */
export async function loadEngine(
  options: LoadOptions,
): Promise<{ wasm: Glue; memory: WebAssembly.Memory; threads: number; probeFailed?: true }> {
  const { threads, warn } = options;
  if (threads !== 1 && options.isolated) {
    let started: { wasm: ThreadedGlue; memory: WebAssembly.Memory } | undefined;
    const probing = options.probe?.().catch(() => undefined);
    const imported = new Promise<ThreadedGlue>((resolve) => resolve(options.importThreaded())).then(
      (glue) => ({ glue }),
      (error: unknown) => ({ error }),
    );
    const broken = await probing;
    if (broken !== undefined) {
      // Nothing threaded instantiated: this Worker runs single-threaded itself.
      warn?.(
        `fast-mesh-wasm: the multithreaded engine cannot run in this browser (${broken}); running single-threaded. ${SEE}`,
      );
      const glue = await options.importSingle();
      return { wasm: glue, memory: (await glue.default()).memory, threads: 1, probeFailed: true };
    }
    try {
      const got = await imported;
      if ("error" in got) {
        throw got.error;
      }
      started = { wasm: got.glue, memory: (await got.glue.default()).memory };
    } catch (e) {
      warn?.(failed(e));
    }
    if (started !== undefined) {
      try {
        await timeout(started.wasm.initThreadPool(threads), options.poolTimeoutMs ?? POOL_TIMEOUT_MS);
      } catch (e) {
        throw new PoolStartError(failed(e));
      }
      if (options.crashChannel !== undefined) {
        try {
          started.wasm.watchPoolCrashes(options.crashChannel);
        } catch (e) {
          // The engine works unwatched: a crash on a pool thread then hangs
          // the scene instead of poisoning it, as before the watch existed.
          const cause = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
          warn?.(
            `fast-mesh-wasm: crash detection unavailable (${cause}): a crash on a thread of the pool will hang the Worker scene instead of failing its calls with INSTANCE_POISONED. ${SEE}`,
          );
        }
      }
      return { ...started, threads };
    }
  } else if (threads !== 1) {
    warn?.(
      "fast-mesh-wasm: the page is not cross-origin isolated (no COOP/COEP headers), so the Worker scene " +
        `runs single-threaded. ${SEE}`,
    );
  }
  const glue = await options.importSingle();
  const { memory } = await glue.default();
  return { wasm: glue, memory, threads: 1 };
}

/** The warning for a threaded start that failed with `e`. */
function failed(e: unknown): string {
  const cause = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
  return `fast-mesh-wasm: the multithreaded engine could not start (${cause}); running single-threaded. ${SEE}`;
}

/** `promise`, or a rejection once `ms` have passed. */
function timeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const late = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`the thread pool did not start within ${ms} ms`)), ms);
  });
  return Promise.race([promise, late]).finally(() => clearTimeout(timer));
}

/**
 * Starts the Worker's engine for its {@link InitMessage}: loads the glue
 * ({@link loadEngine}, probing with {@link probeSharedGrowth} unless
 * `loaders` names a probe) and makes scenes use it. `isolated` defaults to the
 * global `crossOriginIsolated`; development warnings are collected only on a
 * page (where it is a boolean; Node `worker_threads` has none) and when
 * `debug` is set. Resolves with what the ready message reports (`probeFailed`
 * in production too: the scene caches it); rejects
 * with a {@link PoolStartError} (its `warning` set in development) when the
 * Worker must be replaced by a single-threaded one.
 */
export async function startEngine(
  init: InitMessage,
  loaders: Pick<LoadOptions, "importThreaded" | "importSingle" | "poolTimeoutMs" | "probe">,
  isolated: boolean | undefined = (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated,
): Promise<{ threads: number; warning?: string; probeFailed?: true }> {
  let warning: string | undefined;
  const dev = init.debug && typeof isolated === "boolean";
  let loaded: Awaited<ReturnType<typeof loadEngine>>;
  try {
    loaded = await loadEngine({
      probe: () => probeSharedGrowth(),
      ...loaders,
      // `navigator` is missing in some hosts (older Node `worker_threads`).
      threads: init.threads ?? defaultThreads((globalThis as { navigator?: { hardwareConcurrency?: number } }).navigator),
      isolated: isolated === true,
      crashChannel: init.crashChannel,
      warn: dev
        ? (message) => {
            warning ??= message;
          }
        : undefined,
    });
  } catch (e) {
    if (e instanceof PoolStartError && dev) {
      e.warning = e.message;
    }
    throw e;
  }
  const { wasm, memory, threads, probeFailed } = loaded;
  useWasmScene(wasm.WasmScene);
  setMemory(memory);
  return {
    threads,
    ...(warning === undefined ? {} : { warning }),
    ...(probeFailed === undefined ? {} : { probeFailed }),
  };
}
