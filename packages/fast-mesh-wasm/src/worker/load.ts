// Which WebAssembly artifact the Worker runs: the threaded one (dist/wasm-threads,
// with a rayon pool of Workers sharing its memory) on cross-origin isolated
// pages, else the single-threaded one (dist/wasm). Any failure starting the
// threaded engine (the import, instantiating its 4 GiB-maximum shared memory,
// the pool) falls back to the single-threaded engine. The glue modules come
// from injected loaders, so this module imports neither: only the Worker
// entry names them, and single-threaded mode never fetches wasm-threads/.
import { setMemory } from "../core.js";
import { useWasmScene } from "../scene.js";
import type { WasmScene } from "../wasm/fast_mesh_wasm.js";
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
}

/** Options of {@link loadEngine}. */
export interface LoadOptions {
  /** The threads asked for: the threaded engine is tried unless 1. */
  threads: number;
  /** `crossOriginIsolated`: shared memory (and so threads) is available. */
  isolated: boolean;
  importThreaded: () => Promise<ThreadedGlue>;
  importSingle: () => Promise<Glue>;
  /** Development only: called once when the engine is single-threaded though more threads were asked for. */
  warn?: (message: string) => void;
  /** How long the pool may take to start: {@link POOL_TIMEOUT_MS} by default. */
  poolTimeoutMs?: number;
}

/**
 * How long `initThreadPool` may take before the threaded engine is given up.
 * Its nested Workers report ready by message; one that fails to load (blocked
 * by the page, or a bundler that broke its import) never does, and the pool
 * would wait forever.
 */
export const POOL_TIMEOUT_MS = 10_000;

const SEE = 'See "Multithreading (COOP/COEP)" in the fast-mesh-wasm README.';

/** The default pool size: the logical cores, at most 8 (4 when unknown). */
export function defaultThreads(nav: { hardwareConcurrency?: number } | undefined): number {
  return Math.min(nav?.hardwareConcurrency ?? 4, 8);
}

/**
 * Loads and initialises the glue to run: the threaded one with its pool
 * started when `isolated` and `threads` is not 1, else (or when that fails)
 * the single-threaded one. Returns the glue, its memory and the threads the
 * engine computes with (1 when single-threaded).
 */
export async function loadEngine(
  options: LoadOptions,
): Promise<{ wasm: Glue; memory: WebAssembly.Memory; threads: number }> {
  const { threads, warn } = options;
  if (threads !== 1 && options.isolated) {
    try {
      const glue = await options.importThreaded();
      const { memory } = await glue.default();
      await timeout(glue.initThreadPool(threads), options.poolTimeoutMs ?? POOL_TIMEOUT_MS);
      return { wasm: glue, memory, threads };
    } catch (e) {
      const cause = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
      warn?.(`fast-mesh-wasm: the multithreaded engine could not start (${cause}); running single-threaded. ${SEE}`);
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
 * ({@link loadEngine}) and makes scenes use it. `isolated` defaults to the
 * global `crossOriginIsolated`; development warnings are collected only on a
 * page (where it is a boolean; Node `worker_threads` has none) and when
 * `debug` is set. Resolves with what the ready message reports.
 */
export async function startEngine(
  init: InitMessage,
  loaders: Pick<LoadOptions, "importThreaded" | "importSingle" | "poolTimeoutMs">,
  isolated: boolean | undefined = (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated,
): Promise<{ threads: number; warning?: string }> {
  let warning: string | undefined;
  const { wasm, memory, threads } = await loadEngine({
    ...loaders,
    // `navigator` is missing in some hosts (older Node `worker_threads`).
    threads: init.threads ?? defaultThreads((globalThis as { navigator?: { hardwareConcurrency?: number } }).navigator),
    isolated: isolated === true,
    warn:
      init.debug && typeof isolated === "boolean"
        ? (message) => {
            warning ??= message;
          }
        : undefined,
  });
  useWasmScene(wasm.WasmScene);
  setMemory(memory);
  return warning === undefined ? { threads } : { threads, warning };
}
