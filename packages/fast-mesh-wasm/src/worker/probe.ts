// A check, before the threaded engine is tried, that this browser can share a
// growing WebAssembly memory between threads. WebKit 26 (Safari 26) cannot:
// once another thread grows a shared memory, this thread's bulk memory
// operations (`memory.fill`, `memory.copy`) still bound-check against the old
// size and trap, while plain loads and stores see the new one. The threaded
// engine's pool Workers grow its memory as they start, so there the start
// traps on a pool Worker (which then never reports ready, and the pool waits
// for the 10 s timeout) or on the engine's thread. Any later grow on a pool
// thread would trap the engine the same way, so the threaded engine must not
// run at all. The probe reproduces exactly that, in a few milliseconds: a
// nested Worker grows a tiny shared memory by one page, then this thread
// fills a byte of the new page. Only a trap there rules threads out; a probe
// that cannot run or gets no answer says nothing, and the pool is tried as
// before (with its own timeout).

/** How long the probe's nested Worker may take to answer before the probe gives up (and threads are tried). */
export const PROBE_TIMEOUT_MS = 1000;

/** The parts of a nested Worker the probe uses. */
export interface ProbeWorker {
  postMessage(msg: unknown): void;
  terminate(): void;
  onmessage: ((e: { data: unknown }) => void) | null;
  onerror: ((e: unknown) => void) | null;
}

/** Starts a nested Worker running the classic script `source`; may throw. */
export type ProbeSpawn = (source: string) => ProbeWorker;

/**
 * A module importing a shared memory `e.m` (1 page, at most 2), exporting
 * `grow(pages)` (`memory.grow`: the old size in pages, or -1) and
 * `fill(address)` (`memory.fill` of one byte there; returns 0).
 */
const MODULE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 6, 1, 96, 1, 127, 1, 127, 2, 9, 1, 1, 101, 1, 109, 2, 3, 1, 2, 3, 3, 2, 0, 0, 7, 15,
  2, 4, 103, 114, 111, 119, 0, 0, 4, 102, 105, 108, 108, 0, 1, 10, 22, 2, 6, 0, 32, 0, 64, 0, 11, 13, 0, 32, 0, 65, 0,
  65, 1, 252, 11, 0, 65, 0, 11,
]);

const PAGE = 65536;

/** The nested Worker: instantiates the module on the memory it is sent, grows it a page, answers `grow`'s result. */
const SCRIPT = `self.onmessage = (e) => {
  let old = -1;
  try {
    old = new WebAssembly.Instance(e.data.module, { e: { m: e.data.memory } }).exports.grow(1);
  } catch {}
  self.postMessage(old);
};`;

/** Nested classic Workers from a Blob URL, when this scope has what that takes. */
function browserSpawn(): ProbeSpawn | undefined {
  const g = globalThis as {
    Worker?: new (url: string) => ProbeWorker;
    Blob?: new (parts: string[], options: { type: string }) => object;
    URL?: { createObjectURL?(blob: object): string; revokeObjectURL?(url: string): void };
  };
  const { Worker: NestedWorker, Blob: ScriptBlob, URL: Urls } = g;
  if (typeof NestedWorker !== "function" || typeof ScriptBlob !== "function" || Urls?.createObjectURL === undefined) {
    return undefined;
  }
  const createUrl = Urls.createObjectURL.bind(Urls);
  return (source) => {
    const url = createUrl(new ScriptBlob([source], { type: "text/javascript" }));
    const revoke = () => Urls.revokeObjectURL?.(url);
    let worker: ProbeWorker;
    try {
      worker = new NestedWorker(url);
    } catch (e) {
      revoke();
      throw e;
    }
    const terminate = worker.terminate.bind(worker);
    worker.terminate = () => {
      terminate();
      revoke();
    };
    return worker;
  };
}

/**
 * Resolves with why the threaded engine must not run in this browser (a
 * shared memory grown on a nested Worker cannot be filled here), or
 * `undefined` when it may: the probe passed, or could not tell (no nested
 * Workers, a Worker that failed or did not answer within `timeoutMs`). Never
 * rejects. `spawn` defaults to Blob-URL Workers, and to no probe where there
 * are none (Node).
 */
export async function probeSharedGrowth(
  options: { spawn?: ProbeSpawn; timeoutMs?: number } = {},
): Promise<string | undefined> {
  const spawn = options.spawn ?? browserSpawn();
  if (spawn === undefined) {
    return undefined;
  }
  let worker: ProbeWorker | undefined;
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    const module = new WebAssembly.Module(MODULE);
    const memory = new WebAssembly.Memory({ initial: 1, maximum: 2, shared: true });
    const { exports } = new WebAssembly.Instance(module, { e: { m: memory } }) as unknown as {
      exports: { fill(address: number): number };
    };
    exports.fill(0);
    const w = spawn(SCRIPT);
    worker = w;
    const old = await new Promise<unknown>((resolve) => {
      timer = setTimeout(() => resolve(undefined), options.timeoutMs ?? PROBE_TIMEOUT_MS);
      w.onmessage = (e) => resolve(e.data);
      w.onerror = () => resolve(undefined);
      w.postMessage({ module, memory });
    });
    if (old !== 1) {
      return undefined;
    }
    try {
      exports.fill(PAGE);
    } catch {
      return "a shared WebAssembly memory grown on another thread traps memory.fill on this one, as in WebKit 26";
    }
    return undefined;
  } catch {
    // No shared memory, no nested Worker (blocked by a Content Security
    // Policy, say): nothing learnt.
    return undefined;
  } finally {
    clearTimeout(timer);
    worker?.terminate();
  }
}
